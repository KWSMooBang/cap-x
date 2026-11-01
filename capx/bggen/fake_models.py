"""FAKE model server for smoke runs and tests: an OpenAI-compatible ``/v1/chat/completions`` endpoint
whose replies are scripted, not model outputs. Route by model name: ``*init*`` -> init JSON, ``*agent*``
-> tool-call JSON (every ``bad_every``-th reply is malformed to exercise the fallback), ``*vdm*`` ->
an ``Observe:`` line.

``FakeBrain`` answers generically (format smoke on real data): goal = first movable onto the first
support / into the first container; Observe answers come from ``answer(statement, image_t)`` (default:
a deterministic hash -> mostly 0, some 1 and ?). Tests subclass it with a truth function.

    python -m capx.bggen.fake_models --port 8199   # then BGGEN_*_URL=http://127.0.0.1:8199/v1
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .vocab import CONTAINER_KEYS, SUPPORT_KEYS

FAKE_MODELS = {"init": "fake-init", "agent": "fake-agent", "vdm": "fake-vdm"}


def _texts(messages: list[dict[str, Any]]) -> str:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        else:
            out += [p.get("text", "") for p in c or [] if p.get("type") == "text"]
    return "\n".join(out)


def _images(messages: list[dict[str, Any]]) -> list[str]:
    return [p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]


def image_t(url: str) -> int | None:
    """Frame index encoded by synthetic test renders in pixel (0, 0): R * 256 + G (None for real images)."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        r, g, b = img.getpixel((0, 0))
        return r * 256 + g if b == 7 else None
    except Exception:
        return None


class FakeBrain:
    def __init__(self, bad_every: int = 0, goal: list[Any] | None = None) -> None:
        self.bad_every = bad_every
        self.goal = goal
        self.n = {"init": 0, "agent": 0, "vdm": 0}
        self.lock = threading.Lock()
        self.prompts: list[tuple[str, str]] = []
        self.max_tokens: int | None = None  # the request's output limit (set by the server before reply)

    # --- roles ------------------------------------------------------------------------------------
    def init(self, messages: list[dict[str, Any]]) -> str:
        text = _texts(messages)
        objs = re.findall(r"^- \S+ = (\S+) \(", text, re.M)
        cats = {a: a.rsplit("_", 1)[0] for a in objs}
        sup = [a for a in objs if cats[a] in SUPPORT_KEYS]
        mov = [a for a in objs if cats[a] not in SUPPORT_KEYS]
        goal = self.goal
        if goal is None:
            goal = []
            if mov and sup:
                cont = [s for s in sup if cats[s] in CONTAINER_KEYS]
                goal = [[["in", mov[0], cont[0]]] if cont else [["on", mov[0], sup[0]]]]
        obs = [f"({'in' if cats[s] in CONTAINER_KEYS else 'on'} {m} {s})" for m in mov for s in sup][:12]
        return json.dumps({"remember": ["initial support of each object"], "goal_branches": goal, "observe": obs})

    def agent(self, messages: list[dict[str, Any]]) -> str:
        with self.lock:
            self.n["agent"] += 1
            n = self.n["agent"]
        if self.bad_every and n % self.bad_every == 0:
            return "I would check the bowl."  # malformed on purpose
        text = _texts(messages)
        m = re.search(r"^\[candidate statements\] (.*)$", text, re.M)
        cands = re.findall(r"\([^()]*\)", m.group(1)) if m else []
        pairs = [s[1:-1].split()[1:3] for s in cands if s.startswith(("(on ", "(in ")) and len(s[1:-1].split()) == 3]
        calls = [{"tool": "observe", "statements": cands[:8]}] if cands else []
        if pairs:
            calls.append({"tool": "geometry", "pairs": [p for p in pairs if p[1] != "table"][:4]})
        return json.dumps({"calls": calls, "why": "check the objects around the event"})

    def answer(self, stmt: str, t: int | None) -> str:
        h = int(hashlib.md5(f"{stmt}{t}".encode()).hexdigest()[:4], 16) % 10
        return "1" if h < 3 else "?" if h == 9 else "0"

    def vdm(self, messages: list[dict[str, Any]]) -> str:
        text = _texts(messages)
        m = re.search(r"Statements:\n(.*?)\nOutput exactly", text, re.S)
        stmts = [s.strip() for s in (m.group(1).splitlines() if m else []) if s.strip()]
        imgs = _images(messages)
        t = image_t(imgs[-1]) if imgs else None
        return "Observe: " + " | ".join(f"{s} {self.answer(s, t)}" for s in stmts)

    def reply(self, model: str, messages: list[dict[str, Any]]) -> str:
        role = next((r for r in ("init", "agent", "vdm") if r in model), "vdm")
        with self.lock:
            self.prompts.append((role, _texts(messages)[:4000]))
            if role != "agent":
                self.n[role] += 1
        return getattr(self, role)(messages)


def fake_usage(messages: list[dict[str, Any]], content: str) -> dict[str, Any]:
    """Text-only token estimate (chars / 4). Images are NOT counted as text (their base64 would inflate the
    count ~1000x); they are reported as ``images`` and their tokens are left to the real API's count."""
    return {"prompt_tokens": len(_texts(messages)) // 4, "completion_tokens": len(content) // 4,
            "images": len(_images(messages)), "fake": True}


def make_handler(brain: FakeBrain) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            brain.max_tokens = body.get("max_tokens")  # a test brain may reply differently per output limit
            content = brain.reply(body.get("model", ""), body.get("messages", []))
            # a reply may be {"content", "finish_reason", "reasoning_content"} (tests: a reasoning model cut by the limit)
            finish, reasoning = "stop", None
            if isinstance(content, dict):
                finish, reasoning = content.get("finish_reason", "stop"), content.get("reasoning_content")
                content = content.get("content") or ""
            msg = {"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})}
            usage = fake_usage(body.get("messages", []), content)
            if reasoning:
                usage["completion_tokens_details"] = {"reasoning_tokens": len(reasoning) // 4}
                usage["completion_tokens"] += len(reasoning) // 4
            data = json.dumps({"model": body.get("model"), "choices": [{"message": msg, "finish_reason": finish}], "usage": usage})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data.encode())

        def log_message(self, *a: Any) -> None:
            pass

    return Handler


class FakeServer:
    def __init__(self, brain: FakeBrain | None = None, port: int = 0) -> None:
        self.brain = brain or FakeBrain()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.brain))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def env(self) -> dict[str, str]:
        return {**{f"BGGEN_{r.upper()}_URL": self.url for r in FAKE_MODELS},
                **{f"BGGEN_{r.upper()}_MODEL": m for r, m in FAKE_MODELS.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--bad-every", type=int, default=5)
    a = ap.parse_args()
    srv = FakeServer(FakeBrain(bad_every=a.bad_every), a.port)
    print(f"FAKE model server at {srv.url} (scripted replies, not models)")
    srv.httpd.serve_forever()


if __name__ == "__main__":
    main()
