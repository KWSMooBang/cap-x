"""RoboMME (https://robomme.github.io) simulator — ManiSkill3/SAPIEN Panda tasks.

Install with ``uv sync --extra robomme`` (Python >= 3.11). Requires a Vulkan device for SAPIEN
(NVIDIA ICD, or lavapipe from ``mesa-vulkan-drivers`` for CPU rendering).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robomme  # noqa: F401  # type: ignore[import-not-found]  # fail early if the extra is missing

from capx.envs.base import BaseEnv
from capx.integrations.robomme import RoboMMEHandle, load_robomme_task

EPISODES_PER_SPLIT = {"train": 100, "val": 50, "test": 50}

# RoboMME obs list key -> cap-x camera name
_CAMERAS = {"front": "front", "wrist": "wrist"}


class RoboMMEEnv(BaseEnv):
    """One RoboMME task in ``joint_angle`` action space.

    Actions are absolute ``[7 joint positions, gripper]`` with gripper ``1`` open / ``-1`` closed.
    ``obs["memory"]`` exposes RoboMME's task goal, oracle online subgoals, status and the frame
    lists returned at reset (the conditioning video for Video* tasks). ``obs["obs_lists"]`` holds
    the full frame lists from the latest step.

    ``reset(seed=k)`` selects episode ``(k - 1) % episodes`` of the split (fixed benchmark seeds),
    mirroring how FrankaLiberoEnv maps trial seeds to init states.
    """

    def __init__(
        self,
        task_id: str = "PickXtimes",
        episode_idx: int = 0,
        dataset: str = "test",
        privileged: bool = False,
        max_steps: int = 1000,
        include_depth: bool = True,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.dataset = dataset
        self.privileged = privileged
        self.max_steps = max_steps
        self._include = {
            "include_front_depth": include_depth,
            "include_wrist_depth": include_depth,
            "include_front_camera_extrinsic": True,
            "include_wrist_camera_extrinsic": True,
            "include_front_camera_intrinsic": True,
            "include_wrist_camera_intrinsic": True,
        }
        self.handle: RoboMMEHandle = self._load(episode_idx)
        self._gripper = 1.0
        self._reward = 0.0
        self._step_count = 0
        self.reset()

    def _load(self, episode_idx: int) -> RoboMMEHandle:
        return load_robomme_task(
            task_id=self.task_id,
            episode_idx=episode_idx,
            dataset=self.dataset,
            action_space="joint_angle",
            max_steps=self.max_steps,
            **self._include,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            episode_idx = (seed - 1) % EPISODES_PER_SPLIT[self.dataset]
            if episode_idx != self.handle.episode_idx:
                self.handle.env.close()
                self.handle = self._load(episode_idx)
        self.handle.reset()
        self._gripper = 1.0
        self._reward = 0.0
        self._step_count = 0
        obs = self.get_observation()
        info = {"task_prompt": self.handle.task_goal(), "memory": obs["memory"]}
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        _, self._reward, terminated, truncated, info = self.handle.step(np.asarray(action, dtype=np.float64))
        self._step_count += 1
        truncated = truncated or self._step_count >= self.max_steps
        return self.get_observation(), self._reward, terminated, truncated, info

    # ----------------------- joint-control interface (as FrankaLiberoEnv) -----------------------

    def _set_gripper(self, fraction: float) -> None:
        """0.0 (closed) to 1.0 (open)."""
        self._gripper = float(np.clip(fraction, 0.0, 1.0))

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 120
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        action = np.concatenate([target, [self._gripper * 2.0 - 1.0]])
        for _ in range(max_steps):
            self.step(action)
            current = np.asarray(self.handle.last_obs["joint_state_list"][-1], dtype=np.float64)[:7]
            if np.linalg.norm(current - target) < tolerance or self.handle.last_info.get("status") not in (
                None,
                "ongoing",
            ):
                break

    # ----------------------------------- observation -----------------------------------

    def get_observation(self) -> dict[str, Any]:
        lists = self.handle.last_obs
        info = self.handle.last_info
        obs: dict[str, Any] = {}
        for src, cam in _CAMERAS.items():
            if f"{src}_rgb_list" not in lists:
                continue
            obs[cam] = {"images": {"rgb": np.asarray(lists[f"{src}_rgb_list"][-1])}}
            if f"{src}_depth_list" in lists:
                obs[cam]["images"]["depth"] = np.asarray(lists[f"{src}_depth_list"][-1])
            if f"{src}_camera_extrinsic_list" in lists:
                obs[cam]["extrinsics"] = np.asarray(lists[f"{src}_camera_extrinsic_list"][-1])
            if f"{src}_camera_intrinsic" in info:
                obs[cam]["intrinsics"] = np.asarray(info[f"{src}_camera_intrinsic"])
        joints = np.asarray(lists["joint_state_list"][-1], dtype=np.float64).reshape(-1)[:7]
        gripper = np.asarray(lists["gripper_state_list"][-1], dtype=np.float64).reshape(-1)
        obs["robot_joint_pos"] = np.concatenate([joints, gripper[:1]])
        # World frame [x, y, z, roll, pitch, yaw]; Panda base at (-0.615, 0, 0), identity rotation.
        obs["robot_eef_state"] = np.asarray(lists["eef_state_list"][-1], dtype=np.float64)
        obs["obs_lists"] = lists
        obs["memory"] = self.handle.memory_state()
        return obs

    def compute_reward(self) -> float:
        return self._reward

    def task_completed(self) -> bool:
        return self.handle.last_info.get("status") == "success"

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        return np.asarray(self.handle.last_obs["front_rgb_list"][-1])

    def close(self) -> None:
        self.handle.env.close()


__all__ = ["RoboMMEEnv"]
