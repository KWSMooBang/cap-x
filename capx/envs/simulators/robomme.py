"""RoboMME (https://robomme.github.io) simulator — ManiSkill3/SAPIEN Panda tasks.

Install with ``uv sync --extra robomme`` (Python >= 3.11). Requires a Vulkan device for SAPIEN
(NVIDIA ICD; lavapipe cannot render the cameras RoboMME needs).

Conventions (see :mod:`capx.integrations.robomme.geometry`): everything this env reports is in the
**robot base frame** (``panda_link0``; RoboMME world shifted by +0.615 m in x), like FrankaLiberoEnv.
Camera ``pose_mat`` is camera -> base with OpenCV camera axes, depth is in metres.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv, EpisodeTruncated
from capx.integrations.robomme import (
    CONTROL_FREQ_HZ,
    DEFAULT_MAX_STEPS,
    STICK_TASKS,
    RoboMMEHandle,
    load_robomme_task,
)
from capx.integrations.robomme.geometry import (
    DEFAULT_GRIPPER_TCP_OFFSET,
    camera_pose_in_base,
    depth_to_meters,
    eef_state_to_base_pose,
    matrix_to_wxyz,
    measure_tcp_offset,
    pose_matrix,
    uniform_frame_indices,
)

# Environment variable that overrides ``condition_frames`` without regenerating configs.
CONDITION_FRAMES_ENV = "CAPX_ROBOMME_CONDITION_FRAMES"

EPISODES_PER_SPLIT = {"train": 100, "val": 50, "test": 50}

# RoboMME camera -> cap-x camera name. The LIBERO names are reused so the API docs (and code written for
# LIBERO) apply unchanged: "agentview" is RoboMME's front camera, "robot0_eye_in_hand" its wrist camera.
CAMERAS = {"front": "agentview", "wrist": "robot0_eye_in_hand"}

# RoboMME status values that end an episode (DemonstrationWrapper._step_batch); "error" is a failed step.
TERMINAL_STATUSES = ("success", "fail", "timeout", "error")

logger = logging.getLogger(__name__)


class RoboMMEResetError(RuntimeError):
    """RoboMME reported a finished episode right after reset (stale state from an earlier episode)."""


# ManiSkill Panda finger joint range (metres per finger); 0.04 = fully open.
GRIPPER_OPEN_WIDTH = 0.04


class RoboMMEEnv(BaseEnv):
    """One RoboMME task in ``joint_angle`` action space, driven through cap-x's joint interface.

    Actions sent to RoboMME are absolute ``[7 joint positions (rad), gripper]`` with gripper ``+1`` open
    and ``-1`` closed (``_set_gripper`` fraction ``f`` -> ``2f - 1``). One ``step`` is one ManiSkill
    control step (``pd_joint_pos`` at 20 Hz). ``move_to_joints_blocking`` interpolates in joint space
    (at most ``max_joint_step`` rad per step) and then waits for convergence.

    Episode end: the first step whose RoboMME ``status`` is terminal (success / fail / timeout / error) or
    that would exceed ``max_steps`` latches ``episode_end_reason`` and raises :class:`EpisodeTruncated`,
    so generated code stops there (the code env then reports ``truncated``). ``task_completed()`` is the
    latched ``status == "success"``.

    Privileged information: ``get_observation()`` holds only cameras and robot state. RoboMME's oracle
    subgoals, status and goal list stay in ``memory_state()`` (evaluation logs only). The conditioning
    video returned at reset is exposed through :meth:`conditioning_frames` for the first prompt.

    ``reset(seed=k)`` selects episode ``(k - 1) % episodes`` of the split, so trial ``k`` always plays the
    same test episode (trials 1-5 -> episodes 0-4).

    Every ``reset`` runs on an env on which RoboMME ``env.reset()`` has not been called yet: if the current
    env was already reset (e.g. at construction), it is closed and the episode is loaded again. RoboMME's
    second reset of the same env keeps the previous success state (``status == "success"`` right after
    reset), which ended trial 1 of the ensemble pilot after one control step. ``reset`` also raises
    :class:`RoboMMEResetError` if the reset info is not ``status == "ongoing"`` with ``success`` false.
    """

    def __init__(
        self,
        task_id: str = "PickXtimes",
        episode_idx: int = 0,
        dataset: str = "test",
        privileged: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
        include_depth: bool = True,
        condition_frames: int = 32,
        max_joint_step: float = 0.06,
        video_subsample: int = 2,
        enable_render: bool = True,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.dataset = dataset
        self.privileged = privileged
        self.max_steps = int(max_steps)
        # CAPX_ROBOMME_CONDITION_FRAMES overrides the config (fallback, e.g. 16, if the model limits images).
        override = os.environ.get(CONDITION_FRAMES_ENV)
        self.condition_frames = int(override) if override else int(condition_frames)
        self.max_joint_step = float(max_joint_step)
        self.is_stick = task_id in STICK_TASKS
        self.viser_debug = False  # not supported; kept for CodeExecutionEnvBase
        self._control_freq = CONTROL_FREQ_HZ
        self._subsample_rate = max(1, int(video_subsample))
        self._include = {
            "include_front_depth": include_depth,
            "include_wrist_depth": include_depth,
            "include_front_camera_extrinsic": True,
            "include_wrist_camera_extrinsic": True,
            "include_front_camera_intrinsic": True,
            "include_wrist_camera_intrinsic": True,
        }
        self._record_frames = False
        self._record_wrist_camera = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self.handle: RoboMMEHandle = self._load(episode_idx)
        self._init_episode_state()
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

    def _init_episode_state(self) -> None:
        self._gripper = 1.0
        self._reward = 0.0
        self._sim_step_count = 0
        self.episode_end_reason: str | None = None
        self._outcome: str | None = None
        self.home_joint_position: np.ndarray | None = None
        self._target_joints: np.ndarray | None = None
        self.tcp_offset_hand: np.ndarray = DEFAULT_GRIPPER_TCP_OFFSET.copy()
        self._tool_rot_in_hand = np.eye(3)

    # ----------------------------------- episode -----------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            episode_idx = (seed - 1) % EPISODES_PER_SPLIT[self.dataset]
        else:
            episode_idx = self.handle.episode_idx
        if episode_idx != self.handle.episode_idx or self.handle.reset_count > 0:
            # Fresh env for every reset: RoboMME env.reset() exactly once per episode env.
            self.handle.env.close()
            self.handle = self._load(episode_idx)
        _, reset_info = self.handle.reset()
        self._check_reset_info(reset_info)
        self._init_episode_state()
        self.home_joint_position = self._current_joints()
        self._target_joints = self.home_joint_position.copy()
        obs = self.get_observation()
        # Tool centre point in the panda_hand frame (fingertips, or the stick tip for the stick tasks).
        tcp_pos, tcp_wxyz = eef_state_to_base_pose(self.handle.last_obs["eef_state_list"][-1])
        measured = measure_tcp_offset(self.home_joint_position, tcp_pos, tcp_wxyz)
        if measured is not None:
            self.tcp_offset_hand, self._tool_rot_in_hand = measured
            obs = self.get_observation()
        info = {
            "task_prompt": self.handle.task_goal(),
            "episode_idx": self.handle.episode_idx,
            "tcp_offset_hand": self.tcp_offset_hand.tolist(),
            "memory": self.memory_state(),
        }
        return obs, info

    def _check_reset_info(self, info: dict[str, Any]) -> None:
        status = info.get("status")
        status = status[0] if isinstance(status, (list, tuple)) and status else status
        success = info.get("success", False)
        success = bool(np.any(np.asarray(success.detach().cpu() if hasattr(success, "detach") else success)))
        if (status is not None and status != "ongoing") or success:
            msg = (
                f"RoboMME {self.task_id} episode {self.handle.episode_idx}: reset returned status={status!r} "
                f"success={success} (expected ongoing/False; stale state from an earlier reset?)"
            )
            logger.error(msg)
            raise RoboMMEResetError(msg)

    @property
    def episode_over(self) -> bool:
        return self.episode_end_reason is not None

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """One RoboMME control step. Raises EpisodeTruncated once the episode has ended."""
        if self.episode_over:
            raise EpisodeTruncated(f"Episode already ended ({self.episode_end_reason}); no further actions")
        if self._sim_step_count >= self.max_steps:
            self.episode_end_reason = "step_cap"
            raise EpisodeTruncated(f"Episode truncated: reached max_steps={self.max_steps} control steps")
        _, reward, terminated, truncated, info = self.handle.step(np.asarray(action, dtype=np.float64))
        self._sim_step_count += 1
        status = info.get("status")
        self._reward = 1.0 if status == "success" else 0.0
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        if status in TERMINAL_STATUSES or terminated or truncated:
            self._outcome = status if status in TERMINAL_STATUSES else ("fail" if terminated else "timeout")
            self.episode_end_reason = self._outcome
            if self._record_frames:
                self._record_frame()
            raise EpisodeTruncated(f"Episode ended after {self._sim_step_count} control steps")
        if self._sim_step_count >= self.max_steps:
            # Budget used up on this step: end now, so code stops here instead of at its next action.
            self.episode_end_reason = "step_cap"
            if self._record_frames:
                self._record_frame()
            raise EpisodeTruncated(f"Episode truncated: reached max_steps={self.max_steps} control steps")
        return self.get_observation(), self._reward, bool(terminated), bool(truncated), info

    # ----------------------- joint-control interface (as FrankaLiberoEnv) -----------------------

    def _current_joints(self) -> np.ndarray:
        return np.asarray(self.handle.last_obs["joint_state_list"][-1], dtype=np.float64).reshape(-1)[:7]

    def _action(self, joints: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(joints, dtype=np.float64).reshape(7), [self._gripper * 2.0 - 1.0]])

    def _set_gripper(self, fraction: float) -> None:
        """0.0 (closed) to 1.0 (open); sent to RoboMME as 2*fraction - 1 (-1 closed, +1 open)."""
        self._gripper = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        """Hold the last commanded joint target for one control step (applies the gripper command)."""
        target = self._target_joints if self._target_joints is not None else self._current_joints()
        self.step(self._action(target))

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.01, max_steps: int = 120
    ) -> None:
        """Move to absolute joint angles: interpolate (<= max_joint_step rad/step), then settle.

        Uses at most ``max_steps`` control steps; stops early once within ``tolerance`` (L2, rad).
        """
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        start = self._target_joints if self._target_joints is not None else self._current_joints()
        n_interp = int(np.ceil(np.max(np.abs(target - start)) / max(self.max_joint_step, 1e-6)))
        steps = 0
        for i in range(1, n_interp + 1):
            if steps >= max_steps:
                break
            waypoint = start + (target - start) * (i / n_interp)
            self._target_joints = waypoint
            self.step(self._action(waypoint))
            steps += 1
        self._target_joints = target
        while steps < max_steps:
            if np.linalg.norm(self._current_joints() - target) < tolerance and steps > 0:
                break
            self.step(self._action(target))
            steps += 1

    # ----------------------------------- observation -----------------------------------

    def get_observation(self) -> dict[str, Any]:
        """Cameras (base frame, OpenCV axes, depth in metres) and robot state. No oracle information."""
        lists = self.handle.last_obs
        info = self.handle.last_info
        obs: dict[str, Any] = {}
        for src, cam in CAMERAS.items():
            if f"{src}_rgb_list" not in lists:
                continue
            obs[cam] = {"images": {"rgb": np.asarray(lists[f"{src}_rgb_list"][-1], dtype=np.uint8)}}
            if f"{src}_depth_list" in lists:
                obs[cam]["images"]["depth"] = depth_to_meters(lists[f"{src}_depth_list"][-1])
            if f"{src}_camera_extrinsic_list" in lists:
                pose_mat = camera_pose_in_base(lists[f"{src}_camera_extrinsic_list"][-1])
                obs[cam]["pose_mat"] = pose_mat
            if f"{src}_camera_intrinsic" in info:
                obs[cam]["intrinsics"] = np.asarray(info[f"{src}_camera_intrinsic"], dtype=np.float64)
        joints = self._current_joints()
        finger = np.asarray(lists["gripper_state_list"][-1], dtype=np.float64).reshape(-1)
        gripper = 0.0 if self.is_stick else float(np.clip(finger[0] / GRIPPER_OPEN_WIDTH, 0.0, 1.0))
        tcp_pos, tcp_wxyz = eef_state_to_base_pose(lists["eef_state_list"][-1])
        # Report the panda_hand orientation (what solve_ik / goto_pose take) at the tool centre point.
        hand_rot = pose_matrix(tcp_pos, tcp_wxyz)[:3, :3] @ self._tool_rot_in_hand.T
        tcp_wxyz = matrix_to_wxyz(hand_rot)
        obs["robot_joint_pos"] = np.concatenate([joints, [gripper]])
        obs["robot_cartesian_pos"] = np.concatenate([tcp_pos, tcp_wxyz, [gripper]])
        return obs

    def conditioning_frames(self, k: int | None = None) -> list[tuple[int, np.ndarray]]:
        """Up to ``k`` (default ``condition_frames``) front-camera frames of the conditioning video,
        uniformly spaced with first and last included, as ``(frame index, image)``. Empty for tasks whose
        reset returns no video. Frame ``i`` was recorded at control step ``i`` (20 Hz)."""
        k = self.condition_frames if k is None else int(k)
        frames = self.handle.conditioning_frames("front")
        return [(i, frames[i]) for i in uniform_frame_indices(len(frames), k)]

    def num_conditioning_frames(self) -> int:
        return self.handle.num_conditioning_frames()

    def memory_state(self) -> dict[str, Any]:
        """Oracle task progress for evaluation logs (never in a prompt or the code sandbox)."""
        state = self.handle.memory_state()
        state["episode_idx"] = self.handle.episode_idx
        state["sim_steps"] = self._sim_step_count
        state["episode_end_reason"] = self.episode_end_reason
        return state

    def compute_reward(self) -> float:
        return 1.0 if self._outcome == "success" else 0.0

    def task_completed(self) -> bool:
        return self._outcome == "success"

    # ----------------------------------- video -----------------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            # Conditioning video first (red border, as RoboMME's own recorder marks demonstration frames).
            frames = self.handle.conditioning_frames("front")
            for frame in frames[:: self._subsample_rate]:
                self._frame_buffer.append(_red_border(frame))
                if wrist_camera:
                    self._wrist_frame_buffer.append(np.zeros_like(frame))
            self._record_frame()

    def _record_frame(self) -> None:
        self._frame_buffer.append(self.render())
        if self._record_wrist_camera:
            wrist = self.render_wrist()
            if wrist is not None:
                self._wrist_frame_buffer.append(wrist)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if clear:  # hand the frames over (no second copy of a whole trial's frames)
            frames, self._frame_buffer = self._frame_buffer, []
            return frames
        return [f.copy() for f in self._frame_buffer]

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if clear:
            frames, self._wrist_frame_buffer = self._wrist_frame_buffer, []
            return frames
        return [f.copy() for f in self._wrist_frame_buffer]

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._wrist_frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        return np.asarray(self.handle.last_obs["front_rgb_list"][-1], dtype=np.uint8)

    def render_wrist(self) -> np.ndarray | None:
        frames = self.handle.last_obs.get("wrist_rgb_list")
        return np.asarray(frames[-1], dtype=np.uint8) if frames else None

    def close(self) -> None:
        self.handle.env.close()


def _red_border(frame: np.ndarray, width: int = 4) -> np.ndarray:
    out = np.array(frame, dtype=np.uint8, copy=True)
    out[:width], out[-width:], out[:, :width], out[:, -width:] = [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]
    return out


__all__ = ["RoboMMEEnv", "RoboMMEResetError", "CAMERAS", "TERMINAL_STATUSES"]
