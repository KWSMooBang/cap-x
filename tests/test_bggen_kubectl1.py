"""Regression tests for the kubectl 1a / 1b / 2 findings (lead_cap-bg, 2026-11-01):
(b) LIBERO grasps indistinguishable by width -> provisional ``close`` + co-motion, (e) fake token counts, and the
fake server's fallback rate. (a), (c), (d) and the B1K oracle are in test_bggen.py / test_bggen_adapters.py."""

from __future__ import annotations

import json

import numpy as np

from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.fake_models import FakeBrain, FakeServer
from capx.bggen.models import ChatClient, Models, RoleConfig
from capx.bggen.pipeline import run_episode
from tests.bggen_fakes import FakeDetector, SwapWorld, TruthBrain, make_world, write_swap_episode


def _not_carried(n_until: int) -> SwapWorld:
    """The swap script, cut before the 2nd grasp, with the first close NOT holding the bowl: the fingers close and
    the hand lifts and travels to plate_A, but bowl_A stays on plate_B."""
    w = make_world()
    start = w.bowls[0]["bowl_1"].copy()
    for t in range(w.n):
        if w.held[t] == "bowl_1":
            w.held[t] = None
        w.bowls[t]["bowl_1"] = start.copy()
    for name in ("eef", "width", "bowls", "held"):
        setattr(w, name, getattr(w, name)[:n_until])
    return w


def test_close_retracted_when_the_object_stays(tmp_path):
    full = make_world()
    w = _not_carried(full.grasp_t[1] - 5)
    write_swap_episode(tmp_path, w)
    with FakeServer(TruthBrain(w)) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    assert s["events"]["close"] == 1
    assert s["grasp_checks"][0]["obj"] == "bowl_A" and s["grasp_checks"][0]["result"] == "retracted"
    trace = [json.loads(x) for x in (tmp_path / "out" / "trace.jsonl").read_text().splitlines()]
    after = next(t for t in trace if t.get("grasp_verify"))
    assert after["graph"]["(inhand bowl_A)"] == [0.03, "robot"] and after["graph"]["(handempty)"] == [0.97, "robot"]
    rel = next(t for t in trace if t["event"]["kind"] == "release")
    assert rel["event"]["obj"] is None  # nothing to let go of
    segs = json.loads((tmp_path / "out" / "backward_log.json").read_text())["segments"]
    # bggen-fix3 C2: a bowl does not open / close: a retracted grasp stays an object segment (pick / place labels,
    # no Effect), not articulated / manipulate (that merged pick and place into one label)
    assert [(s["kind"], s["retracted"], s["effect"]) for s in segs] == [("object", True, [])]
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows_backward.jsonl").read_text().splitlines()]
    assert not any(r["subtask"].startswith("manipulate(") for r in rows)
    assert any(r["subtask"].startswith("pick(bowl_A") for r in rows)


def test_close_confirmed_by_comotion_in_the_swap(tmp_path):
    w = write_swap_episode(tmp_path)
    with FakeServer(TruthBrain(w)) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    checks = s["grasp_checks"]
    assert [c["obj"] for c in checks] == ["bowl_A", "bowl_B", "bowl_A"]
    assert all(c["result"] in ("confirmed", "confirmed_lifted") and c["decided_t"] < r for c, r in zip(checks, w.release_t, strict=True))
    assert s["goal"]["final_satisfied"]


def test_fake_usage_does_not_count_images_as_text():
    big = "data:image/png;base64," + "A" * 400_000
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x" * 400}, {"type": "image_url", "image_url": {"url": big}},
                                         {"type": "image_url", "image_url": {"url": big}}]}]
    with FakeServer(FakeBrain()) as srv:
        c = ChatClient(RoleConfig("vdm", "fake-vdm", srv.url + "/chat/completions"))
        out = c.query(msgs)
    assert out["usage"]["prompt_tokens"] < 200 and out["usage"]["images"] == 2 and out["usage"]["fake"]
    assert c.usage.images == 2 and c.usage.prompt_tokens < 200


def test_fake_server_fallbacks_are_its_scripted_bad_replies(tmp_path):
    """kubectl 2: libero 3 / b1k 18 fallbacks. FakeBrain(bad_every=5) answers every 5th agent call with prose on
    purpose: the fallback count is agent calls // 5, all 'no JSON object'."""
    w = write_swap_episode(tmp_path)
    with FakeServer(FakeBrain(bad_every=5)) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    calls = s["models"]["agent"]["calls"]
    assert s["agent_fallbacks"] == calls // 5 > 0
    assert s["agent_fallback_reasons"] == {"no JSON object in the reply": calls // 5}
    report = (tmp_path / "out" / "report.md").read_text()
    assert "images sent" in report and "FAKE server counts text only" in report
    assert np.isfinite(s["models"]["vdm"]["images"]) and s["models"]["vdm"]["images"] >= s["models"]["vdm"]["calls"]
