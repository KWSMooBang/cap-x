from __future__ import annotations

from typing import Any

from capx.integrations.libero_mem import LiberoMemHandle, load_libero_mem_task


class LiberoMemWrapper:
    """Thin wrapper around a LIBERO-Mem task (mirrors ``LiberoWrapper``).

    ``memory_state()`` returns the env's sequential-subgoal progress.
    """

    def __init__(self, task_id: int, cam_w: int = 128, cam_h: int = 128) -> None:
        self.handle: LiberoMemHandle = load_libero_mem_task(task_id=task_id, cam_w=cam_w, cam_h=cam_h)
        self._success = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._success = False
        obs, info = self.handle.reset(seed=seed)
        return obs, info

    def step_action(
        self, action: list[float]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, done, info = self.handle.step(action)
        self._success = self._success or done
        return obs, reward, done, info

    def memory_state(self) -> dict[str, Any]:
        return self.handle.memory_state()

    def success(self) -> bool:
        return self._success

    def set_success(self, value: bool) -> None:
        self._success = value
