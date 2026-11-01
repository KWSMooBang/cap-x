"""RoboMME frame conventions and Panda kinematics (pure numpy, no simulator needed).

Conventions checked against RoboMME (``doc/env_format.md``, ``env_record_wrapper/DemonstrationWrapper.py``)
and ManiSkill3 (``mani_skill/sensors/camera.py``):

- World frame: +x forward (robot -> objects), +y left, +z up; table top is ``z = 0``. The Panda base
  (``panda_link0``) is fixed at world ``(-0.615, 0, 0)`` with identity rotation, so robot base coordinates
  are world coordinates shifted by ``+0.615`` in x (``base_from_world``).
- ``*_camera_extrinsic_list`` is ManiSkill's ``extrinsic_cv`` (SAPIEN ``get_extrinsic_matrix``): a 3x4
  **world -> camera** transform in the OpenCV camera convention (x right, y down, z forward).
  ``*_camera_intrinsic`` is ``intrinsic_cv`` (3x3 pinhole).
- ``*_depth_list`` is ManiSkill depth cast to ``int16`` **millimetres** (0 where nothing is rendered).
- ``eef_state_list`` is the ManiSkill TCP pose in the world frame: ``[x, y, z, roll, pitch, yaw]`` with
  extrinsic XYZ Euler angles (unwrapped over time, so they may exceed +-pi).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Panda base (panda_link0) pose in the RoboMME world frame (identity rotation).
ROBOT_BASE_IN_WORLD = np.array([-0.615, 0.0, 0.0], dtype=np.float64)


def base_from_world() -> np.ndarray:
    """4x4 transform from RoboMME world coordinates to robot base coordinates."""
    T = np.eye(4)
    T[:3, 3] = -ROBOT_BASE_IN_WORLD
    return T


def world_to_base_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) - ROBOT_BASE_IN_WORLD


def camera_pose_in_base(extrinsic_cv: np.ndarray) -> np.ndarray:
    """Camera -> robot base 4x4 transform (OpenCV camera axes) from a world -> camera ``extrinsic_cv``."""
    E = np.asarray(extrinsic_cv, dtype=np.float64)
    T_cam_world = np.eye(4)
    T_cam_world[: E.shape[0], :4] = E[:3, :4] if E.shape[0] >= 3 else E
    T_world_cam = np.linalg.inv(T_cam_world)
    return base_from_world() @ T_world_cam


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """RoboMME depth (int16 mm) -> float32 metres, shape (H, W, 1). Float input is taken as metres."""
    d = np.asarray(depth)
    if d.ndim == 2:
        d = d[..., None]
    if np.issubdtype(d.dtype, np.integer):
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


def eef_state_to_base_pose(eef_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``eef_state_list`` entry (world xyz + extrinsic-XYZ rpy) -> (base xyz, quaternion wxyz)."""
    s = np.asarray(eef_state, dtype=np.float64).reshape(-1)
    pos = world_to_base_points(s[:3])
    xyzw = Rotation.from_euler("xyz", s[3:6]).as_quat()
    return pos, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(R, dtype=np.float64)[:3, :3]).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


# Franka Panda modified DH parameters (Craig), as in the Franka docs and the franka_description URDF that
# both ManiSkill (panda_v2.urdf) and robot_descriptions' panda_description use. Joint i: (a, d, alpha).
_PANDA_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
_FLANGE_D = 0.107  # panda_link7 -> panda_link8
_HAND_YAW = -np.pi / 4  # panda_link8 -> panda_hand (rpy 0 0 -pi/4)


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def panda_hand_fk(joints: np.ndarray) -> np.ndarray:
    """``panda_hand`` pose in the robot base frame (4x4) for 7 arm joint angles."""
    q = np.asarray(joints, dtype=np.float64).reshape(-1)[:7]
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_PANDA_DH, q, strict=True):
        T = T @ _dh(a, d, alpha, theta)
    T = T @ _dh(0.0, _FLANGE_D, 0.0, 0.0)
    Rz = np.eye(4)
    Rz[:3, :3] = Rotation.from_euler("z", _HAND_YAW).as_matrix()
    return T @ Rz


# ManiSkill panda_wristcam TCP (panda_hand_tcp): 0.1034 m along panda_hand z, no rotation.
DEFAULT_GRIPPER_TCP_OFFSET = np.array([0.0, 0.0, 0.1034])


def measure_tcp_offset(
    joints: np.ndarray, tcp_pos: np.ndarray, tcp_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Tool frame relative to ``panda_hand`` from joint angles and the reported TCP pose (base frame).

    Returns ``(offset, R)``: the TCP position in the hand frame and the TCP rotation relative to the hand
    (identity for the gripper; the stick's tool frame may be rotated). None if the offset is not 1-60 cm.
    """
    rel = np.linalg.inv(panda_hand_fk(joints)) @ pose_matrix(tcp_pos, tcp_wxyz)
    offset = rel[:3, 3]
    if not (0.01 < np.linalg.norm(offset) < 0.6):
        return None
    return offset, rel[:3, :3]


def uniform_frame_indices(n: int, k: int) -> list[int]:
    """``k`` indices spread uniformly over ``range(n)``, first and last included (all if ``n <= k``)."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
