"""memory_mode=bg_est2 through the trial loop (capx.envs.trial) with the fake gripper world: M2 / M3 / M5 /
M4 in the right prompts, the declared block reaching the memory, belief_trace.jsonl in the trial
directory, cot_record in all_responses, and M4 only in bg_est2 prompts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import capx.belief.memory2 as memory2_mod
import capx.envs.trial as trial_mod
from capx.belief.prompts_v2 import M4_RULES
from capx.envs.trial import _run_single_trial
from capx.utils.launch_utils import _bg_est_summary_line, _memory_config
from tests.belief_fakes import WorldTruth, pick_place_code
from tests.bg_est2_fakes import SWAP2_GOAL, AnyDetector, FakeObserveVdm
from tests.test_bg_est_trial import ARGS, SIM_NAMES, fakes, make_env, text_of  # noqa: F401 (fixture)
from tests.test_truncation import MULTI_TURN_PROMPT, make_config, reply

VDM_ARGS = SimpleNamespace(**{**vars(ARGS), "visual_differencing_model": "fake-vdm"})


def block(o: str, t: str, oa: str, ta: str, declare: bool = True) -> str:
    head = f"# Subtask: place({oa}, {ta})\n# Effect: (on {oa} {ta}) 0>1 | (inhand {oa}) 1>0\n" if declare else ""
    return head + pick_place_code(o, t)


def run2(tmp_path, monkeypatch, coder_replies, **extra):
    env = make_env()
    world = env.low_level_env.world
    holder: dict = {}
    vdm = FakeObserveVdm(world, lambda: holder["mem"].estimator)
    orig_reset = memory2_mod.BgEst2Memory.reset

    def reset(self, *a, **k):
        holder["mem"] = self
        vdm.world = env.low_level_env.world  # env.reset() made a new world
        return orig_reset(self, *a, **k)

    monkeypatch.setattr(memory2_mod.BgEst2Memory, "reset", reset)
    monkeypatch.setattr(memory2_mod, "Sam3PerceptionSource", lambda: AnyDetector(env.low_level_env.world))
    monkeypatch.setattr(memory2_mod, "LiberoMemSource", WorldTruth)
    prompts: list = []
    count = {"coder": 0}

    def query(a, prompt):
        prompts.append((a.model, prompt))
        if a.model == "fake-vdm":
            return reply(vdm(prompt)["content"])
        count["coder"] += 1
        return coder_replies[min(count["coder"], len(coder_replies)) - 1]

    monkeypatch.setattr(trial_mod, "_query_model", query)
    config = {**make_config(tmp_path), "save_multiturn_prompts": True, "memory_mode": "bg_est2", "record_video": False,
              **extra}
    _run_single_trial(env, 0, VDM_ARGS, config, MULTI_TURN_PROMPT)
    trial_dir = sorted(tmp_path.glob("trial_00_*"))[-1]
    return prompts, json.loads((trial_dir / "all_responses.json").read_text()), trial_dir, env, vdm


def test_bg_est2_trial_swap_prompts_trace_and_logs(tmp_path, monkeypatch):
    replies = [reply(SWAP2_GOAL),
               reply("```python\n" + block("b1", "p3", "bowl_A", "plate_C") + "```"),
               reply("REGENERATE\n```python\n" + block("b2", "p1", "bowl_B", "plate_A") + "```"),
               reply("REGENERATE\n```python\n" + block("b1", "p2", "bowl_A", "plate_B", declare=False) + "```"),
               reply("FINISH")]
    prompts, responses, trial_dir, env, vdm = run2(tmp_path, monkeypatch, replies)
    kinds = [m for m, _ in prompts]
    assert kinds[0] == "fake-vdm"  # M2 before the goal parse (M3 uses its result)
    m2 = text_of(prompts[0][1])
    assert "Each detected object is labeled with a mark at its center:" in m2 and "- A = bowl_A (black bowl)" in m2
    m3 = text_of(prompts[1][1])
    assert "Scene, as seen in the front camera image" in m3 and "- plate_C (plate): empty; right side" in m3
    initial = text_of(prompts[2][1])
    assert "Belief graph (turn 0, ESTIMATED; names match the marks in the camera image;" in initial
    assert initial.rstrip().endswith(M4_RULES.splitlines()[-1])
    m5 = [text_of(p) for m, p in prompts if m == "fake-vdm"][1]
    assert "The last code block declared:\nSubtask: place(bowl_A, plate_C)" in m5
    decisions = [text_of(p) for m, p in prompts if m == "fake" and "Executed:" in text_of(p)]
    assert len(decisions) == 3
    for d in decisions:
        assert d.rstrip().endswith(M4_RULES.splitlines()[-1])
    assert "Subtask (declared): place(bowl_A, plate_C)" in decisions[0]
    assert "Observe (t1): (on bowl_A plate_C) 1" in decisions[0]
    assert "Subtask (declared): none" in decisions[2]  # the third block declared nothing: hooks only
    assert "all steps estimated complete" in decisions[2] and "S5b" in decisions[2]
    # Logs: cot_record per turn, the shadow, belief_trace.jsonl in the trial directory.
    assert [r["decision"] for r in responses] == ["initial", "regenerate", "regenerate", "finish"]
    cot = responses[1]["memory_est"]["cot_record"]
    assert cot["Subtask"] == "place(bowl_A, plate_C)" and cot["Observe"].startswith("(on bowl_A plate_C) 1")
    assert responses[3]["memory_est"]["cot_record"]["declared"] is False
    trace = [json.loads(x) for x in (trial_dir / "belief_trace.jsonl").read_text().splitlines()]
    assert [t["kind"] for t in trace] == ["start", "turn", "turn", "turn"]
    assert trace[1]["keys"]["(on bowl_A plate_C)"]["shadow"] is True
    tot = responses[-1]["memory_state"]["shadow_totals"]
    assert tot["declared"] == 2 and tot["turns_with_code"] == 3 and tot["step_error_abs_sum"] == 0
    assert responses[0]["memory_est"]["shadow"]["goal_match_normalized"] is True
    assert responses[-1]["memory_state"]["calls"]["observe_vdm"] == 4
    # Simulator names never reach a prompt.
    for _, p in prompts:
        t = text_of(p)
        assert not any(n in t for n in SIM_NAMES) and "oracle" not in t.lower()


def test_m4_rules_only_in_bg_est2(tmp_path, monkeypatch, fakes):
    from tests.belief_fakes import SWAP_GOAL_JSON
    from tests.test_bg_est_trial import run, code

    model, responses, _ = run(tmp_path, monkeypatch, [reply(SWAP_GOAL_JSON), reply("```python\n" + code(("b1", "p3")) + "```"),
                                                      reply("FINISH"), reply("FINISH")], env=None)
    for p in model.prompts:
        assert "# Subtask:" not in text_of(p)


def test_memory_config_bg_est2():
    cfg = _memory_config(SimpleNamespace(), {"memory_mode": "bg_est2"})
    assert cfg["memory_signals"] == ["S1", "S2", "S3", "S4", "S5"]
    assert cfg["memory_est_observe_max_statements"] == 10 and cfg["memory_est_event_frames"] is False
    assert cfg["memory_est_perception"] == "sam3"
    cli = SimpleNamespace(memory_mode="bg_est2", memory_est_observe_max_statements=6, memory_est_event_frames=True)
    out = _memory_config(cli, {})
    assert out["memory_est_observe_max_statements"] == 6 and out["memory_est_event_frames"] is True
    with pytest.raises(ValueError):
        _memory_config(SimpleNamespace(), {"memory_mode": "bg_est2", "memory_est_perception": "sim_fake"})
    with pytest.raises(ValueError):
        _memory_config(SimpleNamespace(), {"memory_mode": "bg_est", "memory_signals": ["S3"]})  # S3 is v2 only
    assert "NOT A VALID EVALUATION" in _bg_est_summary_line({"memory_mode": "bg_est2",
                                                              "memory_est_perception": "sim_fake_for_tests"})
    assert "memory_est_observe_max_statements" not in _memory_config(SimpleNamespace(), {"memory_mode": "bg_est"})


def test_event_frames_selected_from_turn_frames():
    from capx.belief.hooks import CallRecord
    from capx.belief.memory2 import select_event_frames

    recs = [CallRecord(0, "goto_pose", "", 100, 150, None, None, None, None),
            CallRecord(1, "close_gripper", "", 150, 180, None, None, None, None),
            CallRecord(2, "open_gripper", "", 300, 330, None, None, None, None)]
    frames = list(range(60))  # one frame every 4 steps from step 100
    ev = select_event_frames(recs, frames, 4)
    assert [(e["index"], e["frame"]) for e in ev] == [(1, 21), (2, 58)]
