"""scripts/eval_observe.py query / scoring against a FAKE VDM HTTP server (no simulator, no API key)."""

from __future__ import annotations

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("eval_observe", Path(__file__).resolve().parents[1] / "scripts" / "eval_observe.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def write_scene(root: Path, sid: str, cond: str, truth: dict[str, bool]) -> None:
    from PIL import Image

    d = root / sid
    d.mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(d / "image.png")
    d.joinpath("scene.json").write_text(json.dumps({
        "id": sid, "task_id": 6, "instruction": "swap the 2 bowls on their plates using the empty plate",
        "condition": cond, "detail": "", "marks_now": "A = bowl_A (pixel 3, 4) | 1 = plate_A (5, 6)",
        "statements": list(truth), "truth": truth, "aliases": {"bowl_A": "akita_black_bowl_1", "plate_A": "plate_1"},
    }))


class FakeVdm(BaseHTTPRequestHandler):
    """Answers every statement with 1, except a scene whose prompt says 'tilted' (unreadable reply)."""

    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVdm.seen.append(body)
        text = body["messages"][-1]["content"][-1]["text"]
        stmts = re.findall(r"^\(.*\)$", text.split("Statements:")[1], re.M)
        reply = "Observe: " + " | ".join(f"{s} 1" for s in stmts)
        if "(on bowl_A table)" in stmts and len(stmts) == 1:
            reply = "I am not sure."
        data = json.dumps({"choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeVdm)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("OBSERVE_VDM_URL", f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
    monkeypatch.setenv("OBSERVE_VDM_MODEL", "fake-vdm")
    monkeypatch.setenv("OBSERVE_VDM_API_KEY", "x")
    yield srv
    srv.shutdown()


def test_query_and_score_with_fake_server(tmp_path, server):
    scenes = tmp_path / "scenes"
    write_scene(scenes, "s0", "normal", {"(on bowl_A plate_A)": True, "(on bowl_A table)": False})
    write_scene(scenes, "s1", "arm", {"(on bowl_A plate_A)": False, "(handempty)": True})
    write_scene(scenes, "s2", "tilted", {"(on bowl_A table)": True})
    (scenes / "index.json").write_text(json.dumps(["s0", "s1", "s2"]))
    rows = ev.query_scenes(scenes, tmp_path / "out", ev.default_query_fn())
    assert [r["ok"] for r in rows] == [True, True, False]
    prompt = FakeVdm.seen[0]["messages"]
    assert prompt[0]["content"].startswith("You are the visual observer")
    assert "The last code block made no declaration." in prompt[-1]["content"][-1]["text"]
    assert sum(1 for c in prompt[-1]["content"] if c["type"] == "image_url") == 1
    s = ev.score(rows)
    assert s["unreadable"] == 1 and s["replies"] == 3
    assert s["cells"]["on|all|1"] == {"n": 1, "right": 1, "q": 0, "missing": 0}
    assert s["cells"]["on|all|0"] == {"n": 2, "right": 0, "q": 0, "missing": 0}
    assert s["cells"]["handempty|arm|1"]["right"] == 1
    table = (tmp_path / "out" / "table.md").read_text()
    assert "| on | all | 0 | 2 | 0.00 | 0.00 | 0 |" in table
    # Resume: nothing is queried again.
    n = len(FakeVdm.seen)
    ev.query_scenes(scenes, tmp_path / "out", ev.default_query_fn())
    assert len(FakeVdm.seen) == n


def test_scene_statements_and_aliases():
    pos = {"akita_black_bowl_1": np.array([0.5, 0.1, 0.0]), "akita_black_bowl_2": np.array([0.5, -0.1, 0.0]),
           "plate_1": np.array([0.5, 0.2, 0.0]), "plate_2": np.array([0.6, -0.2, 0.0])}
    alias = ev._aliases(list(pos), pos)
    assert alias == {"akita_black_bowl_2": "bowl_A", "akita_black_bowl_1": "bowl_B", "plate_2": "plate_A", "plate_1": "plate_B"}
    keys = ev.scene_statements(alias)
    assert ("on", "akita_black_bowl_1", "plate_1") in keys and ("on", "akita_black_bowl_1", "table") in keys
    assert keys[-1] == ("handempty", "gripper", None) and not any(k[1].startswith("plate") for k in keys)
    cheese = {"cream_cheese_1": np.zeros(3), "basket_1": np.array([0.5, -0.1, 0]), "basket_2": np.array([0.5, 0.1, 0])}
    keys = ev.scene_statements(ev._aliases(list(cheese), cheese))
    assert ("in", "cream_cheese_1", "basket_1") in keys and ("at", "basket_1", "center") in keys
    assert ("in", "basket_1", "basket_2") not in keys
