"""CaP-X control API for RoboMME, mirroring the LIBERO CaP-Agent0 API.

:class:`FrankaRoboMMEApiReducedSkillLibrary` subclasses
:class:`~capx.integrations.franka.libero_reduced_skill_library.FrankaLiberoApiReducedSkillLibrary`, so the
coding model sees the same function names, signatures and (almost all) the same docstrings as on LIBERO /
LIBERO-Mem. Only RoboMME-specific behaviour is overridden, and those docstrings say what differs:

- ``get_observation``: same keys; ``agentview`` is RoboMME's front camera and ``robot0_eye_in_hand`` its
  wrist camera; ``robot_cartesian_pos`` is the tool centre point (between the fingertips, or the stick tip).
- ``solve_ik`` / ``goto_pose``: the target position is the tool centre point (TCP). The TCP offset from
  ``panda_hand`` is measured from the simulator at reset (about 0.103 m for the gripper; the stick tip for
  PatternLock / RouteStick) instead of LIBERO's fixed 0.1 m. Workspace clamp widened to RoboMME's table.
- ``open_gripper`` / ``close_gripper``: 15 control steps (20 Hz) instead of 30; no-ops for the stick tasks.
- ``goto_home_joint_position``: RoboMME's reset pose (``[0, 0, 0, -pi/2, 0, pi/2, pi/4]``, or the swing
  pose for the stick tasks), not LIBERO's.

All poses are in the robot base frame (``panda_link0``), as on LIBERO. IK uses the same pyroki server
(``panda_description``, ``panda_hand``); ManiSkill's Panda has the same kinematics, and the solution is sent
to RoboMME as a ``joint_angle`` action.
"""


import logging
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.franka.common import (
    apply_tcp_offset,
    extract_arm_joints,
    solve_ik_with_convergence,
)
from capx.integrations.franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary
from capx.integrations.robomme.geometry import DEFAULT_GRIPPER_TCP_OFFSET

logger = logging.getLogger(__name__)

# Reachable workspace in base coordinates (table top is z = 0; objects spawn around x = 0.615).
WORKSPACE_LOW = np.array([-0.1, -0.6, 0.0])
WORKSPACE_HIGH = np.array([0.95, 0.6, 0.9])
GRIPPER_STEPS = 15


class FrankaRoboMMEApiReducedSkillLibrary(FrankaLiberoApiReducedSkillLibrary):
    """Robot control helpers for the Franka Panda in RoboMME (same functions as the LIBERO API)."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def _is_stick(self) -> bool:
        return bool(getattr(self._env, "is_stick", False))

    def _tcp_offset(self) -> np.ndarray:
        """TCP position in the panda_hand frame, measured by the env at reset."""
        offset = getattr(self._env, "tcp_offset_hand", None)
        return np.asarray(DEFAULT_GRIPPER_TCP_OFFSET if offset is None else offset, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # overridden API functions
    # ------------------------------------------------------------------ #
    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment. All poses are in the robot base frame.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current front camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current front camera depth image in meters as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix (camera-to-robot-base, OpenCV camera axes: x right, y down, z forward) as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image in meters as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix (camera-to-robot-base, OpenCV axes) as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current tool center point pose (between the fingertips; the stick tip if the robot holds a stick) in the robot base frame as a numpy array of shape (8,), dtype float64. The first 3 elements are XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (8,), dtype float64: 7 arm joint angles, then the gripper position normalized, 0 (closed) to 1 (open).
                The table top is the plane z = 0 in the robot base frame.
        """
        obs = self._env.get_observation()
        for cam in (self.camera_name, self.wrist_camera_name):
            if cam in obs and "depth" in obs[cam]["images"]:
                depth = obs[cam]["images"]["depth"]
                if depth.ndim == 3:
                    obs[cam]["images"]["depth"] = depth[..., 0]
        return obs

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics so that the tool center point (between the fingertips) reaches a target pose.

        Args:
            position:
                Target tool center point position in the robot base frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation of the panda_hand frame as a unit quaternion in the robot base frame.
                Shape: (4,), [w, x, y, z], dtype float64. [0, 1, 0, 0] points the gripper straight down.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.6, 0.0, 0.1])
            >>> target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # top-down, wxyz
            >>> joints = solve_ik(target_pos, target_quat)
            >>> move_to_joints(joints)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        pos = np.clip(pos, WORKSPACE_LOW, WORKSPACE_HIGH)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_wxyz = quat_wxyz / max(np.linalg.norm(quat_wxyz), 1e-9)
        offset = -self._tcp_offset()  # panda_hand target = TCP target minus the TCP offset (hand frame)

        orientations = [
            ("requested", quat_wxyz),
            ("top-down", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)),
            ("45-tilt", np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)),
            ("side-approach", np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64)),
        ]
        for label, quat in orientations:
            try:
                hand_pos = apply_tcp_offset(pos, quat, offset)
                self.cfg = solve_ik_with_convergence(self.ik_solve_fn, quat, hand_pos, self.cfg)
                if label != "requested":
                    logger.info("IK solved with fallback orientation: %s", label)
                return extract_arm_joints(self.cfg)
            except Exception:
                logger.warning("IK failed with orientation '%s', trying next fallback", label)
                continue
        raise RuntimeError(f"IK failed for position {pos} with all orientation fallbacks")

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using Inverse Kinematics with optional approach motion.

        Combines solve_ik + move_to_joints into a single call. If z_approach > 0,
        first moves to the position raised by z_approach along the robot base Z axis,
        then moves to the final position. The position is the tool center point (between the fingertips).

        Args:
            position: (3,) XYZ target in meters (robot base frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (robot base frame).
            z_approach: Z offset for approach motion (meters). Default 0.0.

        Example:
            >>> goto_pose(np.array([0.6, 0.0, 0.1]), np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.075)
        """
        super().goto_pose(position, quaternion_wxyz, z_approach=z_approach)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        if self._is_stick():
            print("[open_gripper] This robot holds a stick and has no gripper; nothing to do.")
            return
        self._env._set_gripper(1.0)
        for _ in range(GRIPPER_STEPS):
            self._env._step_once()

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        if self._is_stick():
            print("[close_gripper] This robot holds a stick and has no gripper; nothing to do.")
            return
        self._env._set_gripper(0.0)
        for _ in range(GRIPPER_STEPS):
            self._env._step_once()

    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration (the pose at the start of the episode)."""
        super().goto_home_joint_position()


__all__ = ["FrankaRoboMMEApiReducedSkillLibrary"]
