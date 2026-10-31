"""memory_mode=bg_est through the trial loop (capx.envs.trial) with a fake gripper world: sandbox hooks,
goal parsing with the coding model, S5a with the estimated progress, the oracle shadow in the logs
only, the sim_fake flag, and the estimator path reading no simulator state."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import capx.envs.trial as trial_mod
from capx.belief import memory as memory_mod
from capx.belief.memory import BgEstMemory
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.envs.trial import _run_single_trial
from capx.utils.launch_utils import _bg_est_summary_line, _memory_config
from tests.belief_fakes import (
    SHORT_IDS,
    SWAP_GOAL_JSON,
    FakeDetector,
    WorldEnv,
    WorldTruth,
    pick_place_code,
    swap_world,
)
from tests.test_truncation import CUT, MULTI_TURN_PROMPT, FakeModel, make_config, reply

GOAL = "swap the 2 bowls on their plates using the empty plate"
SIM_NAMES = ("akita_black_bowl", "plate_1", "plate_2", "plate_3")
ARGS = SimpleNamespace(
    model="fake", server_url="", api_key=None, max_tokens=100, temperature=1.0, reasoning_effort="low",
    debug=False, use_legacy_multi_turn_decision_prompt=False, visual_differencing_model=None,
    visual_differencing_model_server_url="", visual_differencing_model_api_key=None,
)


def code(*moves: tuple[str, str]) -> str:
    return "\n".join(pick_place_code(o, t) for o, t in moves)


def regen(*moves) -> dict:
    return reply("REGENERATE\n```python\n" + code(*moves) + "```")


def make_env() -> CodeExecutionEnvBase:
    cfg = CodeExecEnvConfig(low_level=WorldEnv(), apis=["FakeGripperApi"], privileged=False,
                            prompt="test\nGoal: {libero_environment_goal}")
    env = CodeExecutionEnvBase(cfg)
    env.low_level_env.handle = SimpleNamespace(task_language=GOAL)
    return env


@pytest.fixture
def fakes(monkeypatch):
    """Fake detector over the env's world (instead of SAM3) and the fake ground truth (instead of
    LiberoMemSource, which the shadow and sim_fake_for_tests use)."""
    monkeypatch.setattr(memory_mod, "Sam3PerceptionSource", lambda: _detector_for_current_env())
    monkeypatch.setattr(memory_mod, "LiberoMemSource", WorldTruth)


_CURRENT: dict = {}


def _detector_for_current_env():
    det = FakeDetector(_CURRENT["env"].low_level_env.world)
    _CURRENT["det"] = det
    return det


def run(tmp_path, monkeypatch, replies, env=None, **config_extra):
    env = env or make_env()
    tmp_path.mkdir(parents=True, exist_ok=True)
    _CURRENT["env"] = env
    model = FakeModel(replies)
    monkeypatch.setattr(trial_mod, "_query_model", model)
    config = {**make_config(tmp_path), "save_multiturn_prompts": True, "memory_mode": "bg_est", "record_video": False,
              "memory_est_verify": False, **config_extra}
    _run_single_trial(env, 0, ARGS, config, MULTI_TURN_PROMPT)
    out = sorted(tmp_path.glob("trial_00_*/all_responses.json"))[-1]
    return model, json.loads(out.read_text()), env


def text_of(prompt) -> str:
    return "\n".join(c["text"] for m in prompt if isinstance(m.get("content"), list)
                     for c in m["content"] if c.get("type") == "text")


def test_bg_est_trial_swap_s5a_and_logs(tmp_path, monkeypatch, fakes):
    # goal parse; initial code moves bowl 1; FINISH (rejected: 2 steps remain); the rest; FINISH.
    replies = [reply(SWAP_GOAL_JSON), reply("```python\n" + code(("b1", "p3")) + "```"), reply("FINISH"),
               regen(("b2", "p1"), ("b1", "p2")), reply("FINISH")]
    model, responses, env = run(tmp_path, monkeypatch, replies)
    world = env.low_level_env.world
    assert world.objects["akita_black_bowl_1"]["pos"][1] == pytest.approx(0.0)  # on plate_2
    assert len(model.prompts) == 5
    goal_prompt = text_of(model.prompts[0])
    assert "Task instruction: " + GOAL in goal_prompt and "bowl_A (black bowl) at (0.50, -0.20, 0.02) on plate_A" in goal_prompt
    initial = text_of(model.prompts[1])
    assert "Belief graph (turn 0), ESTIMATED" in initial and "Progress: step 0/3 done" in initial

    dec1 = text_of(model.prompts[2])
    assert "Belief graph (turn 1), ESTIMATED" in dec1 and "step 1/3 done (bowl_A on plate_C)" in dec1
    assert "t1 bowl_A in gripper -> on plate_C (exp+obs)" in dec1
    assert "FINISH rejected" in text_of(model.prompts[3]) and "2 step(s) remain" in text_of(model.prompts[3])

    assert [r["decision"] for r in responses] == ["initial", "regenerate", "finish"]
    first, regen_entry, fin = responses
    assert first["memory_mode"] == "bg_est" and first["memory_est"]["goal"]["source"] == "llm"
    assert first["memory_est"]["shadow"]["goal_match"] is True
    assert first["memory_est"]["shadow"]["mapping"] == {"bowl_A": "akita_black_bowl_1", "bowl_B": "akita_black_bowl_2",
                                                        "plate_A": "plate_1", "plate_B": "plate_2", "plate_C": "plate_3"}
    assert regen_entry["s5_rejected"] is True
    sh = regen_entry["memory_est"]["shadow"]
    assert sh["step_error"] == 0 and sh["goal_pred_correct"] == sh["goal_pred_total"] > 0
    assert sh["finish_check"] == {"s5a_rejected": True, "oracle_complete": False, "right": True}
    assert [r["fn"] for r in regen_entry["memory_est"]["est"]["records"]][:2] == ["goto_pose", "close_gripper"]
    assert [s["id"] for s in fin["memory_signals"]] == ["S5b"]
    assert fin["memory_est"]["shadow"]["oracle_complete"] is True
    assert fin["memory_est"]["shadow"]["signals"] == [{"id": "S5b", "right": True}]
    state = fin["memory_state"]
    assert state["perception"] == "fake" and state["done"] == 3 and state["calls"]["goal_parse"] == 1
    assert state["calls"]["verify_vdm"] == 0 and state["calls"]["perception"] > 0
    assert state["shadow_totals"]["identity_swap_events"] == 0
    # Simulator names and shadow data never reach a prompt.
    for p in model.prompts:
        t = text_of(p)
        assert not any(n in t for n in SIM_NAMES) and "oracle" not in t.lower()


def test_hooks_installed_only_for_bg_est_and_cleared_next_trial(tmp_path, monkeypatch, fakes):
    env = make_env()
    run(tmp_path / "a", monkeypatch, [reply(SWAP_GOAL_JSON), reply("```python\n" + code(("b1", "p3")) + "```"),
                                      reply("FINISH"), reply("FINISH")], env=env)
    assert env.api_call_wrapper is not None
    assert getattr(env._exec_globals["close_gripper"], "__bg_hooked__", False)
    run(tmp_path / "b", monkeypatch, [reply("```python\npass\n```"), reply("FINISH")], env=env, memory_mode="none")
    assert env.api_call_wrapper is None
    assert not getattr(env._exec_globals["close_gripper"], "__bg_hooked__", False)


def test_hooked_call_returns_unchanged_in_sandbox(tmp_path, monkeypatch, fakes):
    replies = [reply(SWAP_GOAL_JSON), reply("```python\nprint(type(obj_pos('b1')).__name__, goto_pose(obj_pos('b1')))\n```"),
               reply("FINISH"), reply("FINISH")]
    model, responses, env = run(tmp_path, monkeypatch, replies)
    assert "ndarray None" in text_of(model.prompts[2])


def test_truncated_turn_observed_without_s1_s2(tmp_path, monkeypatch, fakes):
    replies = [reply(SWAP_GOAL_JSON), reply("```python\n" + code(("b1", "p3")) + "```"),
               reply(CUT, "length"), reply(CUT, "length"), reply("```python\npass\n```".replace("```python", "REGENERATE\n```python")),
               reply("FINISH"), reply("FINISH")]
    model, responses, env = run(tmp_path, monkeypatch, replies)
    decisions = [r["decision"] for r in responses]
    assert decisions[:3] == ["initial", "truncated_skip", "regenerate"]
    skip, noop = responses[1], responses[2]
    assert "Belief graph (turn 2)" in noop["memory_text"] and noop["memory_signals"] == []  # nothing ran in turn 2
    assert [s["id"] for s in responses[3]["memory_signals"]] == ["S2"]  # turn 3 (pass) vs turn 1


def test_sim_fake_needs_explicit_flag_and_is_flagged(tmp_path, monkeypatch, fakes):
    with pytest.raises(ValueError):
        _memory_config(SimpleNamespace(), {"memory_mode": "bg_est", "memory_est_perception": "sim_fake"})
    with pytest.raises(ValueError):
        BgEstMemory(perception="sim")
    cfg = _memory_config(SimpleNamespace(), {"memory_mode": "bg_est", "memory_est_perception": "sim_fake_for_tests"})
    assert cfg["memory_est_perception"] == "sim_fake_for_tests"
    model, responses, env = run(tmp_path, monkeypatch, [reply(SWAP_GOAL_JSON), reply("```python\npass\n```"),
                                                        reply("FINISH"), reply("FINISH")],
                                memory_est_perception="sim_fake_for_tests")
    assert all("sim_fake_for_tests (FAKE" in r["memory_state"]["perception"] for r in responses)
    assert all("sim_fake_for_tests (FAKE" in r["memory_est"]["perception"] for r in responses)
    line = _bg_est_summary_line({"memory_mode": "bg_est", "memory_est_perception": "sim_fake_for_tests"})
    assert "NOT A VALID EVALUATION" in line
    assert _bg_est_summary_line({"memory_mode": "bg_oracle"}) is None


def test_goal_source_bddl_is_diagnostic(tmp_path, monkeypatch, fakes):
    model, responses, env = run(tmp_path, monkeypatch, [reply("```python\npass\n```"), reply("FINISH"), reply("FINISH")],
                                memory_est_goal_source="bddl")
    assert responses[0]["memory_est"]["goal"]["source"] == "bddl"
    assert responses[0]["memory_state"]["goal_source"] == "bddl" and responses[0]["memory_state"]["calls"]["goal_parse"] == 0
    assert "step 0/3 done" in responses[0]["memory_initial_text"]


def test_verify_vdm_called_with_vdm_model(tmp_path, monkeypatch, fakes):
    args = SimpleNamespace(**{**vars(ARGS), "visual_differencing_model": "fake-vdm"})
    env = make_env()
    _CURRENT["env"] = env
    calls = []

    def query(a, prompt):
        calls.append(a.model)
        if a.model == "fake-vdm":
            return reply('{"answers": [{"id": 1, "answer": "true"}]}')
        script = [reply(SWAP_GOAL_JSON), reply("```python\n" + code(("b1", "p3")) + "```"), reply("FINISH")]
        return script[min(sum(m == "fake" for m in calls) - 1, 2)]

    monkeypatch.setattr(trial_mod, "_query_model", query)
    config = {**make_config(tmp_path), "memory_mode": "bg_est", "record_video": False}
    _run_single_trial(env, 0, args, config, MULTI_TURN_PROMPT)
    responses = json.loads(next(tmp_path.glob("trial_00_*/all_responses.json")).read_text())
    assert calls.count("fake-vdm") == 1
    est = responses[1]["memory_est"]["est"]
    assert est["verify"]["ok"] and est["hypotheses"][0]["edge"] == "bowl_A on plate_C"
    assert responses[1]["memory_state"]["calls"]["verify_vdm"] == 1


class StrictLow:
    """Low-level env stand-in: robot observation and step counter only; anything else fails."""

    def __init__(self, world) -> None:
        object.__setattr__(self, "_world", world)

    def get_observation(self):
        return self._world.get_observation()

    @property
    def _sim_step_count(self):
        return self._world.steps

    def __getattr__(self, name):
        raise AssertionError(f"bg_est read simulator state: {name}")


def test_bg_est_memory_reads_no_sim_state():
    world = swap_world()
    low = StrictLow(world)
    code_env = SimpleNamespace(api_call_wrapper=None)
    mem = BgEstMemory(perception="sam3", shadow=False)  # the shadow (eval logs) is the only sim reader
    from tests.belief_fakes import ScriptedModel

    mem.reset(low, code_env=code_env, instruction=GOAL, goal_query_fn=ScriptedModel(SWAP_GOAL_JSON),
              vdm_query_fn=ScriptedModel('{"answers": []}'), perception_source=FakeDetector(world))
    assert mem.shadow is None
    world.api = {n: code_env.api_call_wrapper(n, f) for n, f in world.functions().items()}
    for turn, (o, t) in enumerate([("b1", "p3"), ("b2", "p1"), ("b1", "p2")], 1):
        world.pick_place(SHORT_IDS[o], SHORT_IDS[t])
        out = mem.after_turn(turn, None, executed=True)
    assert [s.id for s in out.signals] == ["S5b"] and mem.check_finish() is None
    assert "all steps estimated complete" in out.text


def test_memory_config_bg_est_defaults_and_signals():
    cfg = _memory_config(SimpleNamespace(), {"memory_mode": "bg_est"})
    assert cfg["memory_signals"] == ["S1", "S2", "S4", "S5"]
    assert (cfg["memory_est_perception"], cfg["memory_est_goal_source"], cfg["memory_est_verify"]) == ("sam3", "llm", True)
    assert cfg["memory_est_gate_dist"] == pytest.approx(0.10) and cfg["memory_est_max_categories"] == 5
    cli = SimpleNamespace(memory_mode="bg_est", memory_signals="S1,S5", memory_est_verify=False, memory_est_gate_dist=0.2)
    out = _memory_config(cli, {"memory_est_verify": True})
    assert out["memory_signals"] == ["S1", "S5"] and out["memory_est_verify"] is False and out["memory_est_gate_dist"] == 0.2
    with pytest.raises(ValueError):
        _memory_config(SimpleNamespace(), {"memory_mode": "bg_oracle", "memory_signals": ["S4"]})
    assert "memory_est_perception" not in _memory_config(SimpleNamespace(), {})


def test_object_positions_in_prompt_are_estimates_not_sim(tmp_path, monkeypatch, fakes):
    """With detector noise the prompt positions differ from the simulator's (sanity: not copied)."""
    env = make_env()
    _CURRENT["env"] = env
    monkeypatch.setattr(memory_mod, "Sam3PerceptionSource",
                        lambda: FakeDetector(env.low_level_env.world, noise=0.01, seed=3))
    model, responses, _ = run(tmp_path, monkeypatch, [reply(SWAP_GOAL_JSON), reply("```python\npass\n```"),
                                                      reply("FINISH"), reply("FINISH")], env=env)
    text = responses[0]["memory_initial_text"]
    assert "bowl_A (black bowl) at (0.50, -0.20, 0.02)" not in text and np.isfinite(len(text))
