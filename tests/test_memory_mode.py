"""memory_mode in the trial loop: none is byte-identical to the harness before memory existed;
vdm_history and bg_oracle add their block to the decision prompt; S5(a) rejects FINISH once."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

import capx.envs.trial as trial_mod
from capx.belief import memory as memory_mod
from capx.envs.trial import _run_single_trial
from capx.utils.launch_utils import _build_multi_turn_decision_prompt, _memory_config
from tests.test_belief import FakeSource
from tests.test_truncation import CUT, MULTI_TURN_PROMPT, STEP_ONCE, FakeModel, make_config, make_env, reply

GOLDEN = Path(__file__).parent / "fixtures" / "memory_none_prompts.json"

# initial code; regenerate; a decision cut off twice (skipped turn); FINISH.
SCENARIO = [
    reply(STEP_ONCE),
    reply("REGENERATE\n```python\nenv.step_once()\nenv.step_once()\n```"),
    reply(CUT, "length"),
    reply(CUT, "length"),
    reply("FINISH"),
]

ARGS = SimpleNamespace(
    model="fake", server_url="", api_key=None, max_tokens=100, temperature=1.0, reasoning_effort="low",
    debug=False, use_legacy_multi_turn_decision_prompt=False, visual_differencing_model="fake-vdm",
    visual_differencing_model_server_url="", visual_differencing_model_api_key=None,
)


def patch_vdm(monkeypatch) -> None:
    """VDM on, without a renderer or a VLM: fixed image and scripted descriptions."""
    monkeypatch.setattr(trial_mod, "VLM_MODELS", list(trial_mod.VLM_MODELS) + ["fake-vdm"])
    monkeypatch.setattr(
        trial_mod, "_get_visual_feedback",
        lambda env, use_wrist_camera=False: ("data:image/png;base64,AAAA", Image.new("RGB", (4, 4))),
    )
    monkeypatch.setattr(trial_mod, "_describe_initial_scene", lambda *a, **k: "Two bowls on two plates.")
    calls = {"n": 0}

    def vdm(*a, **k):
        calls["n"] += 1
        return f"VDM description {calls['n']}: the bowl moved."

    monkeypatch.setattr(trial_mod, "_get_visual_differencing_feedback", vdm)


def run_scenario(tmp_path, monkeypatch, replies=SCENARIO, **config_extra):
    patch_vdm(monkeypatch)
    model = FakeModel(replies)
    monkeypatch.setattr(trial_mod, "_query_model", model)
    env = make_env()
    config = {**make_config(tmp_path), "use_img_differencing": True, "save_multiturn_prompts": True, **config_extra}
    _run_single_trial(env, 0, ARGS, config, MULTI_TURN_PROMPT)
    responses = json.loads(next(tmp_path.glob("trial_00_*/all_responses.json")).read_text())
    return model, responses, env


def test_none_prompts_match_pre_memory_harness(tmp_path, monkeypatch):
    model, _, _ = run_scenario(tmp_path, monkeypatch)
    prompts = json.loads(json.dumps(model.prompts))
    if os.environ.get("CAPX_WRITE_GOLDEN"):
        GOLDEN.write_text(json.dumps(prompts, indent=1))
    assert len(prompts) == 5
    assert prompts == json.loads(GOLDEN.read_text())


# --- bg_oracle / vdm_history through the trial loop -------------------------------------------

SWAP_PLAN = [("akita_black_bowl_1", "plate_3"), ("akita_black_bowl_2", "plate_1"), ("akita_black_bowl_1", "plate_2")]
NOOP = reply("REGENERATE\n```python\npass\n```")
TWO_STEPS = reply("REGENERATE\n```python\nenv.step_once()\nenv.step_once()\n```")


class StepDrivenSource(FakeSource):
    """Swap-task ground truth driven by the fake env: step k completes the k-th swap move."""

    def __init__(self, low) -> None:
        super().__init__()
        self.low = low

    def _sync(self) -> None:
        self.where = {"akita_black_bowl_1": "plate_1", "akita_black_bowl_2": "plate_2"}
        self.done = []
        for bowl, plate in SWAP_PLAN[: self.low.steps]:
            self.move(bowl, plate)

    def eval_predicate(self, pred):
        self._sync()
        return super().eval_predicate(pred)

    def completed_steps(self):
        self._sync()
        return super().completed_steps()


def run_bg(tmp_path, monkeypatch, replies, **config_extra):
    monkeypatch.setattr(memory_mod, "LiberoMemSource", StepDrivenSource)
    return run_scenario(tmp_path, monkeypatch, replies, memory_mode="bg_oracle", **config_extra)


def test_explicit_none_matches_golden_and_logs_mode(tmp_path, monkeypatch):
    model, responses, _ = run_scenario(tmp_path, monkeypatch, memory_mode="none", memory_signals=["S2", "S5"])
    assert json.loads(json.dumps(model.prompts)) == json.loads(GOLDEN.read_text())
    assert {r.get("memory_mode") for r in responses} == {"none"}
    assert all(r.get("memory_text") is None for r in responses if r["decision"] != "initial")


def test_bg_oracle_prompts_and_s5_rejection(tmp_path, monkeypatch):
    # 1 step done; FINISH -> rejected (2 remain); regenerate 2 steps -> complete; FINISH accepted.
    replies = [reply(STEP_ONCE), reply("FINISH"), TWO_STEPS, reply("FINISH")]
    model, responses, env = run_bg(tmp_path, monkeypatch, replies)
    assert env.low_level_env.steps == 3
    assert len(model.prompts) == 4
    initial = json.dumps(model.prompts[0])
    assert "Belief graph (turn 0):" in initial and "remaining: 3" in initial
    assert responses[0]["memory_initial_text"].startswith("Belief graph (turn 0):")

    dec1 = model.prompts[1][-1]["content"][0]["text"]
    vdm = dec1.index("VDM description 1")
    mem = dec1.index("Belief graph (turn 1):")
    instr = dec1.index("Based on the code")
    assert vdm < mem < instr  # memory block after VDM, before the decision instructions
    assert "step 1/3 done" in dec1 and "Belief graph (turn 0)" not in dec1

    rej = model.prompts[2]
    assert rej[:-2] == model.prompts[1]
    assert rej[-2] == {"role": "assistant", "content": "FINISH"}
    rej_text = rej[-1]["content"][0]["text"]
    assert "FINISH rejected" in rej_text and "2 step(s) remain" in rej_text

    assert [r["decision"] for r in responses] == ["initial", "regenerate", "finish"]
    regen, fin = responses[1], responses[2]
    assert regen["s5_rejected"] is True and regen["s5_rejection"]["id"] == "S5a"
    assert regen["s5_first_reply"]["reply"] == "FINISH"
    assert regen["s5_first_reply"]["decision_source"] == "finish_keyword"
    assert "s5_rejected" not in regen["s5_first_reply"]
    assert regen["memory_mode"] == "bg_oracle" and regen["memory_text"] in dec1
    assert fin["s5_rejected"] is False
    assert [s["id"] for s in fin["memory_signals"]] == ["S5b"] and "all steps verified complete" in fin["memory_text"]
    assert fin["memory_state"] == {"done": 3, "remaining": 0, "overshot": False, "known": True, "over_budget": False}


def test_bg_oracle_second_finish_ends(tmp_path, monkeypatch):
    model, responses, _ = run_bg(tmp_path, monkeypatch, [reply(STEP_ONCE), reply("FINISH"), reply("FINISH")])
    assert len(model.prompts) == 3
    assert [r["decision"] for r in responses] == ["initial", "finish"]
    assert responses[1]["s5_rejected"] is True


def test_bg_oracle_skipped_turn_has_no_s2_and_carries_baseline(tmp_path, monkeypatch):
    # turn 1: 1 step done; turn 2: cut twice (skipped, nothing ran); turn 3: no-op code; FINISH twice.
    model, responses, _ = run_bg(tmp_path, monkeypatch, [
        reply(STEP_ONCE), reply(CUT, "length"), reply(CUT, "length"), NOOP, reply("FINISH"), reply("FINISH"),
    ])
    assert [r["decision"] for r in responses] == ["initial", "truncated_skip", "regenerate", "finish"]
    skip, regen, fin = responses[1], responses[2], responses[3]
    assert skip["memory_signals"] == []  # turn 1 made progress
    assert regen["memory_signals"] == []  # turn 2 ran nothing: no S2
    assert "Belief graph (turn 2):" in regen["memory_text"]
    assert [s["id"] for s in fin["memory_signals"]] == ["S2"]  # turn 3 vs turn 1: no progress
    assert "Belief graph (turn 3):" in fin["memory_text"] and fin["s5_rejected"] is True


def test_bg_oracle_initial_prompt_off_and_signals_off(tmp_path, monkeypatch):
    model, responses, _ = run_bg(
        tmp_path, monkeypatch, [reply(STEP_ONCE), reply("FINISH")],
        memory_in_initial_prompt=False, memory_signals=[],
    )
    assert json.loads(json.dumps(model.prompts[0])) == json.loads(GOLDEN.read_text())[0]
    assert responses[0]["memory_initial_text"] is None
    assert len(model.prompts) == 2 and responses[1]["s5_rejected"] is False  # S5 off: FINISH accepted
    assert "Signals: none" in responses[1]["memory_text"]


def test_vdm_history_holds_previous_turns_only(tmp_path, monkeypatch):
    model, responses, _ = run_scenario(tmp_path, monkeypatch, memory_mode="vdm_history")
    golden = json.loads(GOLDEN.read_text())
    assert json.loads(json.dumps(model.prompts[0])) == golden[0]  # no initial memory for vdm_history
    assert responses[1]["memory_text"] is None  # turn 1: no previous turns
    # Turn 3 (prompt 4): turns 1-2 in the history, turn 3's description only in the VDM block.
    last = model.prompts[4][-1]["content"][0]["text"]
    history = last[last.index("VDM history (previous turns, oldest first):"):last.index("Based on the code")]
    assert "turn 1: VDM description 1" in history and "turn 2: VDM description 2" in history
    assert "VDM description 3" not in history and last.count("VDM description 3") == 1
    assert last.index("VDM description 3: the bowl moved.\n") < last.index("VDM history")
    assert {r["memory_mode"] for r in responses} == {"vdm_history"}
    # Apart from the memory block, the prompts are those of memory_mode none.
    for got, base in zip(model.prompts[1:], golden[1:], strict=True):
        text = got[-1]["content"][0]["text"]
        if "VDM history" in text:
            start, end = text.index("VDM history"), text.index("Based on the code")
            text = text[:start] + text[end:]
        assert text == base[-1]["content"][0]["text"]


def test_builder_without_memory_is_unchanged():
    obs = {"full_prompt": [{"role": "system", "content": "s"},
                           {"role": "user", "content": [{"type": "text", "text": "t"}]}]}
    base = _build_multi_turn_decision_prompt(obs, "mt", None, "vdm")
    assert _build_multi_turn_decision_prompt(obs, "mt", None, "vdm", memory_text=None) == base
    with_mem = _build_multi_turn_decision_prompt(obs, "mt", None, "vdm", memory_text="MEM")
    assert with_mem[-1]["content"][0]["text"] == base[-1]["content"][0]["text"].replace("vdm\n", "vdm\nMEM\n", 1)


def test_memory_config_cli_overrides_yaml():
    cli = SimpleNamespace(memory_mode=None, memory_max_chars=None, memory_signals=None,
                          memory_in_initial_prompt=None, memory_max_finish_rejections=None)
    assert _memory_config(cli, {}) == {
        "memory_mode": "none", "memory_max_chars": 3000, "memory_signals": ["S2", "S5"],
        "memory_in_initial_prompt": True, "memory_max_finish_rejections": 1}
    yaml_cfg = {"memory_mode": "vdm_history", "memory_max_chars": 1000, "memory_signals": ["S5"]}
    assert _memory_config(cli, yaml_cfg)["memory_mode"] == "vdm_history"
    cli.memory_mode, cli.memory_signals, cli.memory_in_initial_prompt = "bg_oracle", "none", False
    out = _memory_config(cli, yaml_cfg)
    assert (out["memory_mode"], out["memory_signals"], out["memory_max_chars"]) == ("bg_oracle", [], 1000)
    assert out["memory_in_initial_prompt"] is False
    cli.memory_signals = "S2, S5"
    assert _memory_config(cli, {})["memory_signals"] == ["S2", "S5"]
    for bad in ({"memory_mode": "graph"}, {"memory_signals": ["S1"]}):
        with pytest.raises(ValueError):
            _memory_config(SimpleNamespace(), bad)
