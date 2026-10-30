"""Step cap: a low-level env stops at max_steps even when generated code swallows exceptions."""

from typing import Any

from capx.envs.base import BaseEnv, EpisodeTruncated
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase


class FakeEnv(BaseEnv):
    """Minimal low-level env with the same step-budget check as FrankaLiberoEnv."""

    def __init__(self, max_steps: int = 50) -> None:
        super().__init__()
        self.max_steps = max_steps
        self._sim_step_count = 0

    def step_once(self) -> None:
        if self._sim_step_count >= self.max_steps:
            raise EpisodeTruncated(f"Episode truncated: reached max_steps={self.max_steps} sim steps")
        self._sim_step_count += 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._sim_step_count = 0
        return self.get_observation(), {}

    def step(self, action: Any):
        return self.get_observation(), 0.0, False, False, {}

    def get_observation(self) -> dict[str, Any]:
        return {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False


def make_env(max_steps: int = 50) -> CodeExecutionEnvBase:
    cfg = CodeExecEnvConfig(low_level=FakeEnv(max_steps), apis=[], privileged=True, prompt="test")
    env = CodeExecutionEnvBase(cfg)
    env.reset()
    return env


def test_swallowing_loop_stops_at_max_steps():
    env = make_env(50)
    code = "while True:\n    try:\n        env.step_once()\n    except Exception:\n        pass\n"
    _, _, _, truncated, info = env.step(code)
    assert truncated
    assert env.low_level_env._sim_step_count == 50
    assert info["sandbox_rc"] == 0
    assert "Episode truncated" in info["stderr"]


def test_under_budget_is_not_truncated():
    env = make_env(50)
    _, _, _, truncated, info = env.step("for _ in range(10):\n    env.step_once()\n")
    assert not truncated and env.low_level_env._sim_step_count == 10 and info["stderr"] == ""
