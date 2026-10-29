"""LIBERO-Mem (https://libero-mem.github.io) simulator.

Same Franka/MuJoCo stack and API surface as :class:`FrankaLiberoEnv`; only the task source
differs. Install with ``uv sync --extra libero-mem`` (exclusive with ``--extra libero``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.integrations.libero_mem import (
    SUITE_NAME,
    ensure_libero_mem_importable,
    load_libero_mem_task,
)

ensure_libero_mem_importable()

from .libero import FrankaLiberoEnv  # noqa: E402


class FrankaLiberoMemEnv(FrankaLiberoEnv):
    """Franka LIBERO-Mem task env.

    On top of the FrankaLiberoEnv observation, ``obs["memory"]`` (and ``info["memory"]`` on
    reset) exposes LIBERO-Mem's sequential subgoal tracker: satisfied subgoals in order,
    how many are required, and whether the final goal was overshot.
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
        init_states = self.handle.init_states
        if init_states is not None and len(init_states) > 0:
            # FrankaLiberoEnv.reset() calls env.reset() after set_init_state(), and the hard reset
            # re-samples the BDDL placement. Re-apply the init state so upstream layouts are used.
            idx = 0 if seed is None else (seed - 1) % len(init_states)
            self._current_obs = self.handle.env.set_init_state(init_states[idx])
            self._current_joints = self.handle.env.sim.data.qpos[:7].copy()
            self.home_joint_position = np.array(self._current_obs["robot0_joint_pos"], dtype=np.float64)
            for _ in range(10):
                self._step_once()
        # Upstream eval resets progress after placing the init state and settling.
        self.handle.reset_progress()
        obs = self.get_observation()
        info["memory"] = obs["memory"]
        return obs, info

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        obs["memory"] = self.handle.memory_state()
        return obs


__all__ = ["FrankaLiberoMemEnv"]
