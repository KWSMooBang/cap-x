"""Trial artifacts with numpy values (np.bool_ etc.) are saved, and a failed save does not end the task."""

from __future__ import annotations

import json

import numpy as np

import capx.envs.trial as trial_mod
import capx.utils.launch_utils as launch_utils
from capx.envs.trial import _run_single_trial
from capx.utils.json_utils import json_default
from capx.utils.launch_utils import _save_trial_artifacts
from tests.test_truncation import ARGS, MULTI_TURN_PROMPT, STEP_ONCE, FakeModel, make_config, make_env

MIXED = [{
    "decision": "finish",
    "flag": np.bool_(True), "off": np.bool_(False), "n": np.int64(3), "x": np.float32(0.5),
    "arr": np.arange(3), "nested": {"m": np.array([[1.5, 2.0]]), "l": [np.int32(7), {"b": np.bool_(False)}]},
}]


def save(tmp_path, responses, trial=0):
    return _save_trial_artifacts(make_config(tmp_path), trial, 0, 1.0, True, "code", None, responses,
                                 ["summary line"], [])


def test_numpy_values_saved_as_python_values(tmp_path):
    code_path = save(tmp_path, MIXED)
    trial_dir = tmp_path / "trial_00_sandboxrc_0_reward_1.000_taskcompleted_1"
    assert code_path == str(trial_dir / "code.py")
    got = json.loads((trial_dir / "all_responses.json").read_text())
    assert got == [{"decision": "finish", "flag": True, "off": False, "n": 3, "x": 0.5, "arr": [0, 1, 2],
                    "nested": {"m": [[1.5, 2.0]], "l": [7, {"b": False}]}}]
    assert not (trial_dir / "save_errors.txt").exists()


def test_plain_values_byte_identical(tmp_path):
    plain = [{"decision": "initial", "a": 1, "b": [1.5, None, True], "s": "é\n", "d": {"k": "v"}}]
    save(tmp_path, plain)
    text = next(tmp_path.glob("trial_00_*/all_responses.json")).read_text()
    assert text == json.dumps(plain, indent=2)
    assert json.dumps(plain, default=json_default) == json.dumps(plain)


def test_json_default_fallback_str():
    class Odd:
        def __str__(self):
            return "odd"
    assert json.dumps({"o": Odd()}, default=json_default) == '{"o": "odd"}'


def test_save_error_recorded_summary_kept(tmp_path):
    circular: dict = {"decision": "finish"}
    circular["self"] = circular  # ValueError in json.dumps
    code_path = save(tmp_path, [circular])
    trial_dir = tmp_path / "trial_00_sandboxrc_0_reward_1.000_taskcompleted_1"
    assert code_path == str(trial_dir / "code.py")
    assert (trial_dir / "summary.txt").read_text() == "summary line"
    assert not (trial_dir / "all_responses.json").exists()
    assert "all_responses.json" in (trial_dir / "save_errors.txt").read_text()


def numpy_reply(content):
    return {"content": content, "reasoning": None, "finish_reason": "stop",
            "usage": {"completion_tokens": np.int64(1), "cached": np.bool_(False), "per": np.arange(2)}}


def test_trial_with_numpy_in_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_mod, "_query_model", FakeModel([numpy_reply(STEP_ONCE), numpy_reply("FINISH")]))
    _run_single_trial(make_env(), 0, ARGS, {**make_config(tmp_path), "record_video": False}, MULTI_TURN_PROMPT)
    trial_dir = next(tmp_path.glob("trial_00_*"))
    responses = json.loads((trial_dir / "all_responses.json").read_text())
    assert responses[0]["usage"] == {"completion_tokens": 1, "cached": False, "per": [0, 1]}
    assert (trial_dir / "trial_end.json").exists() and (trial_dir / "summary.txt").exists()


def test_save_error_then_next_trial_runs(tmp_path, monkeypatch):
    def broken_default(o):
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    monkeypatch.setattr(launch_utils, "json_default", broken_default)
    monkeypatch.setattr(trial_mod, "json_default", broken_default)
    config = {**make_config(tmp_path), "record_video": False}
    for trial in (0, 1):
        monkeypatch.setattr(trial_mod, "_query_model",
                            FakeModel([numpy_reply(STEP_ONCE), numpy_reply("FINISH")]))
        _run_single_trial(make_env(), trial, ARGS, config, MULTI_TURN_PROMPT)
    for trial in (0, 1):
        trial_dir = next(tmp_path.glob(f"trial_{trial:02d}_*"))
        assert (trial_dir / "summary.txt").read_text().strip()
        assert "all_responses.json" in (trial_dir / "save_errors.txt").read_text()
        assert json.loads((trial_dir / "trial_end.json").read_text())["trial"] == trial
