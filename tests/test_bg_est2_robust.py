"""bg_est2 paths that must never end a trial: belief_trace I/O errors and odd Observe replies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capx.belief.memory2 import BgEst2Memory
from tests.belief_fakes import ScriptedModel, WorldEnv, swap_world
from tests.bg_est2_fakes import SWAP2_GOAL, AnyDetector, FakeObserveVdm
from tests.test_bg_est2 import SWAP, A, make_mem, place_code, setup, turn


def test_trace_io_errors_do_not_end_the_trial(tmp_path, capsys):
    # The trace path is a directory: creating the file fails at reset; one warning, state says so.
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    low = WorldEnv()
    low.reset()
    world = low.world
    mem = BgEst2Memory(perception="sam3", shadow=False)
    vdm = FakeObserveVdm(world, lambda: mem.estimator)
    code_env = SimpleNamespace(api_call_wrapper=None)
    mem.reset(low, code_env=code_env, instruction=SWAP, goal_query_fn=ScriptedModel(SWAP2_GOAL), vdm_query_fn=vdm,
              perception_source=AnyDetector(world), trace_path=str(bad))
    world.api = {n: code_env.api_call_wrapper(n, f) for n, f in world.functions().items()}
    world.pick_place(A, "plate_3")
    out = mem.after_turn(1, None, executed=True, code=place_code("bowl_A", "plate_C"))
    assert out.text and mem.state()["trace_error"].startswith("IsADirectoryError")
    assert capsys.readouterr().err.count("belief_trace.jsonl disabled") == 1

    # An append that fails later (e.g. disk full) is caught the same way; the trace is then off.
    mem2, low2, _ = make_mem(tmp_path / "ok", shadow=False)

    def boom(kind):
        raise OSError("No space left on device")

    mem2.estimator.trace = boom
    low2.world.pick_place(A, "plate_3")
    out = mem2.after_turn(1, None, executed=True, code=place_code("bowl_A", "plate_C"))
    assert out.text and mem2.state()["trace_error"] == "OSError: No space left on device" and mem2.trace_path is None
    mem2.after_turn(2, None, executed=True, code=None)  # no second warning, no error
    assert capsys.readouterr().err.count("belief_trace.jsonl disabled") == 1


@pytest.mark.parametrize("content", [["not", "a", "string"], 12345, {"weird": True}, "Observe: (((((", None])
def test_odd_observe_reply_falls_back_to_geometry(content):
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    est.observer.query_fn = lambda prompt: {"content": content}
    world.pick_place(A, "plate_3")
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert r.fallback and est.origin[("on", "bowl_A", "plate_C")] == "geo" and len(r.progress.done) == 1
    assert r.observe["ok"] is False


def test_parse_exception_is_recorded(monkeypatch):
    import capx.belief.observe as obs_mod

    def broken(*a, **k):
        raise TypeError("unexpected reply shape")

    monkeypatch.setattr(obs_mod, "parse_observe", broken)
    res = obs_mod.VdmObserver(lambda p: {"content": "Observe: (on bowl_A plate_B) 1"}).observe(
        "turn", [], [("on", "bowl_A", "plate_B")], ["bowl_A", "plate_B"])
    assert not res.ok and res.error.startswith("parse failed: TypeError")
