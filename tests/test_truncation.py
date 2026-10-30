"""Replies cut off at max_tokens are never executed; timed-out trials keep their video."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import capx.envs.trial as trial_mod
from capx.envs.base import BaseEnv
from capx.envs.runner import _run_single_trial_with_timeout
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.envs.trial import MULTITURN_LIMIT, TRUNCATED_REPLY_FEEDBACK, _run_single_trial

MULTI_TURN_PROMPT = "Executed:\n{executed_code}\nstdout:\n{console_stdout}\nstderr:\n{console_stderr}\n"
STEP_ONCE = "```python\nenv.step_once()\n```"
# Cut-off code: it would take 5 steps if it were (wrongly) run as an unclosed fence.
CUT = "REGENERATE\n```python\nfor _ in range(5):\n    env.step_once()\n"


class FakeEnv(BaseEnv):
    """Low-level env that records one video frame per step_once()."""

    def __init__(self, sleep_s: float = 0.0) -> None:
        super().__init__()
        self.steps = 0
        self.sleep_s = sleep_s
        self._recording = False
        self._frames: list[np.ndarray] = []

    def step_once(self) -> None:
        import time

        self.steps += 1
        if self._recording:
            self._frames.append(np.full((32, 32, 3), self.steps % 255, dtype=np.uint8))
        time.sleep(self.sleep_s)

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._recording = enabled
        if clear:
            self._frames = []

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = list(self._frames)
        if clear:
            self._frames = []
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frames)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.steps = 0
        return self.get_observation(), {}

    def step(self, action: Any):
        return self.get_observation(), 0.0, False, False, {}

    def get_observation(self) -> dict[str, Any]:
        return {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False


def reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {"content": content, "reasoning": None, "finish_reason": finish_reason, "usage": {"completion_tokens": 1}}


class FakeModel:
    """Scripted replies; the last one repeats. Records every prompt."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.prompts: list[Any] = []

    def __call__(self, args, prompt):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


def make_env(sleep_s: float = 0.0) -> CodeExecutionEnvBase:
    cfg = CodeExecEnvConfig(low_level=FakeEnv(sleep_s), apis=[], privileged=True, prompt="test")
    return CodeExecutionEnvBase(cfg)


def make_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "output_dir": str(tmp_path), "use_parallel_ensemble": False, "use_multimodel": False,
        "use_visual_feedback": False, "use_img_differencing": False, "use_video_differencing": False,
        "use_wrist_camera": False, "record_video": True, "use_oracle_code": False, "save_multiturn_prompts": False,
    }


ARGS = SimpleNamespace(
    model="fake", server_url="", api_key=None, max_tokens=100, temperature=1.0, reasoning_effort="low",
    debug=False, use_legacy_multi_turn_decision_prompt=False, visual_differencing_model="fake",
    visual_differencing_model_server_url="", visual_differencing_model_api_key=None,
)


def run(tmp_path, monkeypatch, replies):
    model = FakeModel(replies)
    monkeypatch.setattr(trial_mod, "_query_model", model)
    env = make_env()
    summary = _run_single_trial(env, 0, ARGS, make_config(tmp_path), MULTI_TURN_PROMPT)
    responses = __import__("json").loads(next(tmp_path.glob("trial_00_*/all_responses.json")).read_text())
    return env.low_level_env, model, responses, summary


def prompt_text(prompt) -> str:
    return str(prompt)


def test_decision_cut_twice_is_skipped(tmp_path, monkeypatch):
    low, model, responses, _ = run(tmp_path, monkeypatch, [
        reply(STEP_ONCE), reply(CUT, "length"), reply(CUT, "length"), reply("FINISH"),
    ])
    assert low.steps == 1  # the cut-off code never ran
    assert [r["decision"] for r in responses] == ["initial", "truncated_skip", "finish"]
    skip = responses[1]
    assert (skip["decision_source"], skip["truncation"], skip["code_blocks"]) == ("truncated_skip", "skip", [])
    assert "unclosed_fence" not in skip["decision_flags"]
    assert len(model.prompts) == 4
    # The prompt after the skipped turn carries the fixed feedback; earlier prompts do not.
    assert TRUNCATED_REPLY_FEEDBACK in prompt_text(model.prompts[3])
    assert TRUNCATED_REPLY_FEEDBACK not in prompt_text(model.prompts[1])


def test_decision_cut_then_normal_uses_retry(tmp_path, monkeypatch):
    low, model, responses, _ = run(tmp_path, monkeypatch, [
        reply(STEP_ONCE), reply(CUT, "length"), reply("REGENERATE\n```python\nenv.step_once()\nenv.step_once()\n```"),
        reply("FINISH"),
    ])
    assert low.steps == 3
    assert [r["decision"] for r in responses] == ["initial", "regenerate", "finish"]
    assert responses[1]["truncation"] == "retry" and responses[1]["finish_reason"] == "stop"
    assert responses[1]["truncated_first_usage"] == {"completion_tokens": 1}
    assert "truncation" not in responses[2]
    assert len(model.prompts) == 4


def test_initial_code_cut_twice_goes_to_first_decision(tmp_path, monkeypatch):
    low, model, responses, summary = run(tmp_path, monkeypatch, [
        reply(CUT, "length"), reply(CUT, "length"), reply("FINISH"),
    ])
    assert low.steps == 0
    initial = responses[0]
    assert (initial["decision"], initial["truncation"], initial["decision_source"]) == ("initial", "skip", "truncated_skip")
    assert initial["code_blocks"] == []
    assert [r["decision"] for r in responses] == ["initial", "finish"]
    assert len(model.prompts) == 3
    assert TRUNCATED_REPLY_FEEDBACK in prompt_text(model.prompts[2])
    code = next(tmp_path.glob("trial_00_*/code.py")).read_text()
    assert "for _ in range(5)" not in code and "nothing executed" in code


def test_initial_code_cut_then_normal_runs_retry(tmp_path, monkeypatch):
    low, model, responses, _ = run(tmp_path, monkeypatch, [reply(CUT, "length"), reply(STEP_ONCE), reply("FINISH")])
    assert low.steps == 1
    assert responses[0]["truncation"] == "retry" and responses[0]["code_blocks"] == ["env.step_once()"]


def test_repeated_cuts_end_at_turn_limit(tmp_path, monkeypatch):
    low, model, responses, _ = run(tmp_path, monkeypatch, [reply(CUT, "length")])
    assert low.steps == 0
    skips = [r for r in responses if r["decision"] == "truncated_skip"]
    assert len(skips) == MULTITURN_LIMIT + 1  # same turn budget as executed code: at most MULTITURN_LIMIT + 1 turns
    assert len(model.prompts) == 2 + 2 * len(skips)


def test_empty_cut_reply_is_not_finish(tmp_path, monkeypatch):
    # Nebius returns empty content when the cut lands in the reasoning; that used to end the trial.
    low, model, responses, _ = run(tmp_path, monkeypatch, [
        reply(STEP_ONCE), reply("", "length"), reply("", "length"), reply(STEP_ONCE.replace("```python", "REGENERATE\n```python")),
        reply("FINISH"),
    ])
    assert [r["decision"] for r in responses] == ["initial", "truncated_skip", "regenerate", "finish"]
    assert low.steps == 2


@pytest.mark.parametrize("code, expect_video", [
    ("```python\nwhile True:\n    env.step_once()\n```", True),
    ("```python\nwhile True:\n    pass\n```", False),  # no frames recorded: no video
])
def test_timeout_trial_saves_video(tmp_path, monkeypatch, code, expect_video):
    monkeypatch.setattr(trial_mod, "_query_model", FakeModel([reply(code)]))
    env = make_env(sleep_s=0.01)
    config = {**make_config(tmp_path), "trial_timeout_s": 2}
    summary = _run_single_trial_with_timeout(env, 0, ARGS, config, MULTI_TURN_PROMPT, timeout_s=2)
    assert summary.sandbox_rc == 1
    trial_dir = next(tmp_path.glob("trial_00_sandboxrc_1_*"))
    videos = sorted(p.name for p in trial_dir.glob("video_*.mp4"))
    if expect_video:
        assert videos == ["video_combined.mp4", "video_turn_00.mp4"]
        assert (trial_dir / "video_combined.mp4").stat().st_size > 0
    else:
        assert videos == []
