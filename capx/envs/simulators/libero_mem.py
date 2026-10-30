"""LIBERO-Mem (https://libero-mem.github.io) simulator.

Same Franka/MuJoCo stack and API surface as :class:`FrankaLiberoEnv`; only the task source
differs. Install with ``uv sync --extra libero-mem`` (exclusive with ``--extra libero``).
"""

from __future__ import annotations

from typing import Any

from capx.integrations.libero_mem import (
    SUITE_NAME,
    ensure_libero_mem_importable,
    load_libero_mem_task,
    patch_robosuite_egl_context,
)

ensure_libero_mem_importable()
patch_robosuite_egl_context()

from .libero import FrankaLiberoEnv  # noqa: E402


class FrankaLiberoMemEnv(FrankaLiberoEnv):
    """Franka LIBERO-Mem task env.

    ``memory_state()`` (also ``info["memory"]`` on reset and after each code step) reports
    LIBERO-Mem's sequential subgoal tracker: satisfied subgoals in order, how many are required,
    and whether the final goal was overshot. It is ground-truth task progress, so it is kept out
    of ``get_observation()``, which generated code can read.
    """

    def __init__(
        self,
        task_id: int = 0,
        privileged: bool = True,
        max_steps: int = 4000,
        seed: int | None = None,
        enable_render: bool = False,
        control_freq: int = 20,
        viser_debug: bool = False,
    ) -> None:
        super().__init__(
            suite_name=SUITE_NAME,
            task_id=task_id,
            privileged=privileged,
            max_steps=max_steps,
            seed=seed,
            enable_render=enable_render,
            control_freq=control_freq,
            viser_debug=viser_debug,
        )

    def _load_handle(self, *, suite_name: str, **kwargs: Any) -> Any:
        return load_libero_mem_task(**kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # Upstream eval resets progress after placing the init state and settling.
        self.handle.reset_progress()
        obs = self.get_observation()
        info["memory"] = self.memory_state()
        info["init_source"] = self.handle.init_source
        return obs, info

    def task_completed(self) -> bool:
        return self.handle.task_success()

    def memory_state(self) -> dict[str, Any]:
        return self.handle.memory_state()


__all__ = ["FrankaLiberoMemEnv"]
