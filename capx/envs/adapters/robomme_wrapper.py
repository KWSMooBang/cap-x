from __future__ import annotations

from typing import Any

from capx.integrations.robomme import RoboMMEHandle, load_robomme_task


class RoboMMEWrapper:
    """Thin wrapper around one RoboMME episode (mirrors ``LiberoWrapper``).

    ``memory_state()`` returns RoboMME's task goal, oracle subgoals, status and reset frames.
    """

    def __init__(
        self, task_id: str, episode_idx: int = 0, dataset: str = "test", action_space: str = "joint_angle"
    ) -> None:
        self.handle: RoboMMEHandle = load_robomme_task(
            task_id=task_id, episode_idx=episode_idx, dataset=dataset, action_space=action_space
        )
        self._success = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._success = False
        return self.handle.reset()

    def step_action(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.handle.step(action)
        self._success = self._success or info.get("status") == "success"
        return obs, reward, terminated or truncated, info

    def memory_state(self) -> dict[str, Any]:
        return self.handle.memory_state()

    def success(self) -> bool:
        return self._success

    def set_success(self, value: bool) -> None:
        self._success = value
