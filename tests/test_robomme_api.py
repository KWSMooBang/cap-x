"""RoboMME control API on a stub ManiSkill/RoboMME env (CPU, no SAPIEN / Vulkan).

The stub reproduces RoboMME's ``DemonstrationWrapper`` I/O: list-valued observations, int16-millimetre depth,
world->camera ``extrinsic_cv``, TCP pose as world xyz + extrinsic-XYZ rpy, ``info["status"]`` and oracle
subgoals, and a conditioning video returned by ``reset()``.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.modules.setdefault("robomme", types.ModuleType("robomme"))  # real package only on GPU/Vulkan hosts

import capx.envs.simulators.robomme as sim_mod  # noqa: E402
import capx.envs.trial as trial_mod  # noqa: E402
import capx.integrations.franka.libero_reduced as libero_reduced  # noqa: E402
from capx.envs.base import EpisodeTruncated  # noqa: E402
from capx.envs.tasks.base import CodeExecEnvConfig  # noqa: E402
from capx.envs.tasks.robomme.robomme_env import RoboMMECodeEnv  # noqa: E402
from capx.envs.trial import _run_single_trial  # noqa: E402
from capx.integrations.franka.libero_reduced_skill_library import FrankaLiberoApiReducedSkillLibrary  # noqa: E402
from capx.integrations.robomme import RoboMMEHandle  # noqa: E402
from capx.integrations.robomme.api import FrankaRoboMMEApiReducedSkillLibrary  # noqa: E402
from capx.integrations.robomme.geometry import (  # noqa: E402
    ROBOT_BASE_IN_WORLD,
    matrix_to_wxyz,
    panda_hand_fk,
    pose_matrix,
    uniform_frame_indices,
)

HOME = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4])
TCP = 0.1034  # ManiSkill panda_hand_tcp offset along panda_hand z
H = W = 64
K = np.array([[32.0, 0.0, 32.0], [0.0, 32.0, 32.0], [0.0, 0.0, 1.0]])  # fov 90 deg at 64 px
SECRETS = ("SECRET_SIMPLE_SUBGOAL", "SECRET_GROUNDED_SUBGOAL", "SECRET_STATUS_WORD")


def look_at_extrinsic_cv(eye, target) -> np.ndarray:
    """World -> camera (OpenCV axes) 3x4, built like SAPIEN: camera link x forward, y left, z up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    x = (target - eye) / np.linalg.norm(target - eye)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    T_world_link = np.eye(4)
    T_world_link[:3, :3] = np.stack([x, y, z], axis=1)
    T_world_link[:3, 3] = eye
    T_link_cv = np.eye(4)
    T_link_cv[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    return np.linalg.inv(T_world_link @ T_link_cv)[:3, :4]


FRONT_EYE, FRONT_TARGET = np.array([0.3, 0.0, 0.4]), np.array([0.0, 0.0, -0.2])  # PickXtimes base_camera


class FakeBenchmarkEnv:
    """Stand-in for ``BenchmarkEnvBuilder.make_env_for_episode(...)`` in joint_angle action space."""

    def __init__(self, task_id: str = "PickXtimes", n_demo: int = 0, success_at: int | None = None,
                 fail_at: int | None = None) -> None:
        self.task_id = task_id
        self.stick = task_id in ("PatternLock", "RouteStick")
        self.n_demo = n_demo
        self.success_at, self.fail_at = success_at, fail_at
        self.actions: list[np.ndarray] = []
        self.depth_mm = np.full((H, W, 1), 1000, dtype=np.int16)
        self.closed = False

    def _frame(self, value: int) -> np.ndarray:
        return np.full((H, W, 3), value % 256, dtype=np.uint8)

    def _obs(self, frame_value: int) -> dict[str, Any]:
        T_tcp = panda_hand_fk(self.q) @ pose_matrix([0, 0, TCP], [1, 0, 0, 0])
        pos_world = T_tcp[:3, 3] + ROBOT_BASE_IN_WORLD
        rpy = Rotation.from_matrix(T_tcp[:3, :3]).as_euler("xyz")
        return {
            "front_rgb_list": self._frame(frame_value),
            "wrist_rgb_list": self._frame(frame_value + 100),
            "front_depth_list": self.depth_mm.copy(),
            "wrist_depth_list": self.depth_mm.copy(),
            "front_camera_extrinsic_list": look_at_extrinsic_cv(FRONT_EYE, FRONT_TARGET).astype(np.float32),
            "wrist_camera_extrinsic_list": look_at_extrinsic_cv(pos_world, pos_world - [0, 0, 1e-3] + [1e-4, 0, 0]),
            "joint_state_list": self.q.copy(),
            "gripper_state_list": np.zeros(2) if self.stick else np.array([self.g, self.g]),
            "eef_state_list": np.concatenate([pos_world, rpy]),
        }

    def _info(self, status: str) -> dict[str, Any]:
        return {
            "task_goal": ["pick up the red cube twice, then press the button", "alternative goal"],
            "simple_subgoal_online": SECRETS[0],
            "grounded_subgoal_online": SECRETS[1] + " at <12, 34>",
            "status": status,
            "success": status == "success",
            "front_camera_intrinsic": K.astype(np.float32),
            "wrist_camera_intrinsic": K.astype(np.float32),
        }

    def reset(self):
        self.q, self.g, self.n = HOME.copy(), 0.04, 0
        frames = [self._obs(i + 1) for i in range(self.n_demo)] + [self._obs(200)]
        obs = {k: [f[k] for f in frames] for k in frames[0]}
        return obs, self._info("ongoing")

    def step(self, action):
        a = np.asarray(action, dtype=float)
        self.actions.append(a)
        self.q = self.q + np.clip(a[:7] - self.q, -0.3, 0.3)
        if not self.stick:
            self.g = 0.04 if a[7] > 0 else 0.0
        self.n += 1
        status = "ongoing"
        if self.success_at is not None and self.n >= self.success_at:
            status = "success"
        if self.fail_at is not None and self.n >= self.fail_at:
            status = "fail"
        obs = {k: [v] for k, v in self._obs(200 + self.n).items()}
        terminated = status in ("success", "fail")
        return obs, 0.0, terminated, False, self._info(status)

    def close(self):
        self.closed = True


def numeric_ik(target_pose_wxyz_xyz, prev_cfg=None):
    """Stand-in for the pyroki server: panda_hand IK with this repo's FK. Returns 7 joints + 1 (gripper)."""
    t = np.asarray(target_pose_wxyz_xyz, dtype=float)
    T_goal = pose_matrix(t[4:], t[:4])

    def residual(q):
        T = panda_hand_fk(q)
        rot = Rotation.from_matrix(T_goal[:3, :3].T @ T[:3, :3]).as_rotvec()
        return np.concatenate([T[:3, 3] - T_goal[:3, 3], 0.3 * rot])

    q0 = HOME if prev_cfg is None else np.asarray(prev_cfg, dtype=float)[:7]
    q = least_squares(residual, q0, xtol=1e-12, ftol=1e-12).x
    return np.concatenate([q, [0.04]])


@pytest.fixture
def patched_models(monkeypatch):
    """No perception / IK servers: perception inits return dummies, IK is solved numerically."""
    for name in ("init_sam3", "init_sam3_point_prompt", "init_molmo", "init_contact_graspnet",
                 "init_contact_graspnet_point_clouds"):
        monkeypatch.setattr(libero_reduced, name, lambda *a, **k: (lambda *a2, **k2: []))
    monkeypatch.setattr(libero_reduced, "init_pyroki", lambda *a, **k: numeric_ik)


def make_env(monkeypatch, task_id="PickXtimes", n_demo=0, condition_frames=8, **fake_kw) -> sim_mod.RoboMMEEnv:
    def fake_load(task_id, episode_idx=0, dataset="test", action_space="joint_angle", max_steps=1300, **flags):
        assert action_space == "joint_angle"
        assert flags["include_front_depth"] and flags["include_front_camera_extrinsic"]
        return RoboMMEHandle(env=FakeBenchmarkEnv(task_id, n_demo=n_demo, **fake_kw), task_id=task_id,
                             episode_idx=episode_idx, dataset=dataset)

    monkeypatch.setattr(sim_mod, "load_robomme_task", fake_load)
    return sim_mod.RoboMMEEnv(task_id=task_id, condition_frames=condition_frames)


def make_code_env(monkeypatch, task_id="PickXtimes", n_demo=0, condition_frames=8, **fake_kw) -> RoboMMECodeEnv:
    low = make_env(monkeypatch, task_id, n_demo, condition_frames, **fake_kw)
    cfg = CodeExecEnvConfig(low_level=low, apis=["FrankaRoboMMEApiReducedSkillLibrary"], privileged=False,
                            prompt="You are controlling a Franka Emika robot with API described below.\n"
                                   "Goal: {libero_environment_goal}\n")
    return RoboMMECodeEnv(cfg)


# --------------------------------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------------------------------

def test_api_functions_match_libero(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    ours = FrankaRoboMMEApiReducedSkillLibrary(env).functions()
    libero = FrankaLiberoApiReducedSkillLibrary(env).functions()
    assert list(ours) == list(libero)
    for name in libero:
        assert inspect.signature(ours[name]) == inspect.signature(libero[name]), name
    for name in ("get_observation", "segment_sam3_text_prompt", "segment_sam3_point_prompt", "point_prompt_molmo",
                 "plan_grasp", "plan_grasp_from_point_clouds", "get_oriented_bounding_box_from_3d_points",
                 "solve_ik", "move_to_joints", "open_gripper", "close_gripper", "goto_pose",
                 "goto_home_joint_position", "subsample_point_cloud", "filter_noise", "mask_to_world_points",
                 "select_top_down_grasp"):
        assert name in ours


def test_api_registered():
    from capx.integrations.base_api import list_apis

    assert "FrankaRoboMMEApiReducedSkillLibrary" in list_apis()


def test_uniform_frame_indices():
    assert uniform_frame_indices(0, 8) == []
    assert uniform_frame_indices(5, 8) == [0, 1, 2, 3, 4]
    idx = uniform_frame_indices(100, 8)
    assert len(idx) == 8 and idx[0] == 0 and idx[-1] == 99 and idx == sorted(idx)
    assert uniform_frame_indices(100, 1) == [99]


# --------------------------------------------------------------------------------------------------
# Coordinate conventions
# --------------------------------------------------------------------------------------------------

def test_observation_keys_and_base_frame(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    obs = FrankaRoboMMEApiReducedSkillLibrary(env).get_observation()
    assert set(obs) == {"agentview", "robot0_eye_in_hand", "robot_joint_pos", "robot_cartesian_pos"}
    cam = obs["agentview"]
    assert cam["images"]["rgb"].shape == (H, W, 3) and cam["images"]["rgb"].dtype == np.uint8
    assert cam["images"]["depth"].shape == (H, W) and cam["images"]["depth"].dtype == np.float32
    assert cam["images"]["depth"][0, 0] == pytest.approx(1.0)  # 1000 mm -> 1 m
    assert cam["pose_mat"].shape == (4, 4) and cam["intrinsics"].shape == (3, 3)
    # Camera position in the base frame: world eye shifted by +0.615 m in x.
    np.testing.assert_allclose(cam["pose_mat"][:3, 3], FRONT_EYE - ROBOT_BASE_IN_WORLD, atol=1e-6)
    # Robot state: home joints, open gripper, TCP straight down at the FK position.
    np.testing.assert_allclose(obs["robot_joint_pos"], np.concatenate([HOME, [1.0]]))
    T_tcp = panda_hand_fk(HOME) @ pose_matrix([0, 0, TCP], [1, 0, 0, 0])
    np.testing.assert_allclose(obs["robot_cartesian_pos"][:3], T_tcp[:3, 3], atol=1e-6)
    q = obs["robot_cartesian_pos"][3:7]
    assert abs(abs(q[1]) - 1.0) < 1e-6  # [0, +-1, 0, 0]: pointing down, as LIBERO's top-down quaternion
    np.testing.assert_allclose(env.tcp_offset_hand, [0, 0, TCP], atol=1e-6)


def test_depth_pixel_to_base_point(monkeypatch, patched_models):
    """Known camera: the view-centre pixel deprojects onto the camera's view ray; image up is world up."""
    env = make_env(monkeypatch)
    fake = env.handle.env
    d = 0.5
    fake.depth_mm[32, 32, 0] = int(d * 1000)
    fake.depth_mm[10, 32, 0] = int(d * 1000)
    env.handle.reset()  # re-render observation with the new depth
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    obs = api.get_observation()
    cam = obs["agentview"]
    ray = (FRONT_TARGET - FRONT_EYE) / np.linalg.norm(FRONT_TARGET - FRONT_EYE)
    expected_world = FRONT_EYE + d * ray  # depth is the z (forward) distance; centre pixel lies on the axis
    p = api.pixel_to_world_point(32, 32, cam["images"]["depth"][32, 32], cam["intrinsics"], cam["pose_mat"])
    np.testing.assert_allclose(p, expected_world - ROBOT_BASE_IN_WORLD, atol=1.5e-3)
    mask = np.zeros((H, W), dtype=bool)
    mask[32, 32] = mask[10, 32] = True
    pts = api.mask_to_world_points(mask, cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"])
    upper = pts[np.argmin([np.linalg.norm(x - p) for x in pts]) ^ 1]
    assert upper[2] > p[2]  # a row nearer the top of the image is higher in the world
    assert abs(upper[1] - p[1]) < 1e-6  # same column, camera has no roll: same y


# --------------------------------------------------------------------------------------------------
# IK and motion
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", [[0.6, 0.0, 0.10], [0.55, 0.15, 0.05], [0.7, -0.2, 0.2]])
def test_ik_round_trip_tcp(monkeypatch, patched_models, target):
    env = make_env(monkeypatch)
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    joints = api.solve_ik(np.array(target), quat)
    T_tcp = panda_hand_fk(joints) @ pose_matrix(env.tcp_offset_hand, [1, 0, 0, 0])
    np.testing.assert_allclose(T_tcp[:3, 3], target, atol=1e-4)
    assert abs(abs(matrix_to_wxyz(T_tcp[:3, :3])[1]) - 1.0) < 1e-4


def test_goto_pose_moves_robot_to_target(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    target = np.array([0.62, 0.05, 0.08])
    api.goto_pose(target, np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.05)
    obs = api.get_observation()
    np.testing.assert_allclose(obs["robot_cartesian_pos"][:3], target, atol=5e-3)
    actions = np.stack(env.handle.env.actions)
    assert actions.shape[1] == 8
    assert np.all(np.max(np.abs(np.diff(actions[:, :7], axis=0)), axis=1) <= env.max_joint_step + 1e-9)
    assert env._sim_step_count == len(actions)


def test_pyroki_matches_panda_fk():
    """The pyroki server's panda_description agrees with the Panda kinematics used for RoboMME."""
    pytest.importorskip("pyroki")
    try:
        import capx.serving.launch_pyroki_server as srv

        srv.init_pyroki_server("panda_description", "panda_hand")
    except Exception as exc:  # robot_descriptions needs network on first use
        pytest.skip(f"pyroki / panda_description unavailable: {exc}")
    T_goal = panda_hand_fk([0.1, -0.3, 0.05, -2.0, 0.02, 1.8, 0.7])
    target = np.concatenate([matrix_to_wxyz(T_goal[:3, :3]), T_goal[:3, 3]])
    q = np.asarray(srv._do_solve_ik(target, None))[:7]
    np.testing.assert_allclose(panda_hand_fk(q)[:3, 3], T_goal[:3, 3], atol=5e-3)


def test_gripper_actions(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    fake = env.handle.env
    api.close_gripper()
    closed = np.stack(fake.actions)
    assert np.all(closed[:, 7] == -1.0)
    np.testing.assert_allclose(closed[:, :7], np.tile(HOME, (len(closed), 1)))  # holds the arm still
    assert api.get_observation()["robot_joint_pos"][7] == 0.0
    n = len(fake.actions)
    api.open_gripper()
    assert np.all(np.stack(fake.actions[n:])[:, 7] == 1.0)
    assert api.get_observation()["robot_joint_pos"][7] == 1.0


def test_stick_task_gripper_is_noop(monkeypatch, patched_models, capsys):
    env = make_env(monkeypatch, task_id="PatternLock")
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    api.close_gripper()
    api.open_gripper()
    assert env.handle.env.actions == []
    assert "stick" in capsys.readouterr().out
    assert api.get_observation()["robot_joint_pos"][7] == 0.0


def test_home_is_reset_pose(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    api = FrankaRoboMMEApiReducedSkillLibrary(env)
    api.move_to_joints(HOME + 0.2)
    api.goto_home_joint_position()
    np.testing.assert_allclose(env.get_observation()["robot_joint_pos"][:7], HOME, atol=0.01)


# --------------------------------------------------------------------------------------------------
# Episode end, budget, seeds
# --------------------------------------------------------------------------------------------------

def test_success_latches_and_stops_code(monkeypatch, patched_models):
    code_env = make_code_env(monkeypatch, success_at=5)
    code_env.reset(seed=1)
    _, reward, _, truncated, info = code_env.step("for _ in range(20):\n    close_gripper()\nprint('after')")
    low = code_env.low_level_env
    assert truncated and info["episode_end_reason"] == "success"
    assert low._sim_step_count == 5 and "after" not in info["stdout"]
    assert low.task_completed() and reward == 1.0
    assert "Episode ended" in info["stderr"]
    with pytest.raises(EpisodeTruncated):
        low.step(np.zeros(8))


def test_fail_and_step_cap(monkeypatch, patched_models):
    code_env = make_code_env(monkeypatch, fail_at=3)
    code_env.reset(seed=1)
    _, reward, _, truncated, info = code_env.step("close_gripper()")
    assert truncated and info["episode_end_reason"] == "fail" and reward == 0.0
    assert not code_env.low_level_env.task_completed()

    code_env = make_code_env(monkeypatch)
    code_env.low_level_env.max_steps = 7
    code_env.reset(seed=1)
    _, _, _, truncated, info = code_env.step("close_gripper()\nclose_gripper()")
    assert truncated and info["episode_end_reason"] == "step_cap"
    assert code_env.low_level_env._sim_step_count == 7


def test_seed_selects_test_episode(monkeypatch, patched_models):
    env = make_env(monkeypatch)
    for seed, episode in ((1, 0), (2, 1), (5, 4), (51, 0)):
        _, info = env.reset(seed=seed)
        assert info["episode_idx"] == episode == env.handle.episode_idx


# --------------------------------------------------------------------------------------------------
# Conditioning video and privileged information
# --------------------------------------------------------------------------------------------------

def _images(prompt) -> list[str]:
    return [c["image_url"]["url"] for c in prompt[-1]["content"] if c.get("type") == "image_url"]


def _texts(prompt) -> str:
    return "\n".join(c["text"] for m in prompt for c in (m["content"] if isinstance(m["content"], list)
                     else [{"type": "text", "text": m["content"]}]) if c.get("type") == "text")


def test_conditioning_frames_in_first_prompt_only(monkeypatch, patched_models):
    code_env = make_code_env(monkeypatch, task_id="VideoUnmaskSwap", n_demo=40)
    obs, info = code_env.reset(seed=1)
    images = _images(obs["full_prompt"])
    assert len(images) == 8 and all(u.startswith("data:image/png;base64,") for u in images)
    assert info["condition_frame_indices"] == uniform_frame_indices(40, 8)
    text = _texts(obs["full_prompt"])
    assert "40 frames (2.0 s" in text and "Video frame 8/8 (t = 1.95 s, frame 40 of 40)" in text
    # The frames are the conditioning video (values 1..40), not the current scene (200).
    import base64
    import io

    from PIL import Image

    first = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[0].split(",", 1)[1]))))
    last = np.asarray(Image.open(io.BytesIO(base64.b64decode(images[-1].split(",", 1)[1]))))
    assert first[0, 0, 0] == 1 and last[0, 0, 0] == 40
    # Decision prompts are built from the env's own prompt, which never holds the video.
    obs2, *_ = code_env.step("x = 1")
    assert _images(obs2["full_prompt"]) == []
    obs3, _ = code_env.reset(seed=1)
    assert len(_images(obs3["full_prompt"])) == 8  # no build-up across trials


def test_condition_frames_option_and_no_video_task(monkeypatch, patched_models):
    code_env = make_code_env(monkeypatch, task_id="VideoRepick", n_demo=40, condition_frames=3)
    obs, _ = code_env.reset(seed=1)
    assert len(_images(obs["full_prompt"])) == 3
    code_env = make_code_env(monkeypatch, task_id="PickXtimes", n_demo=0)
    obs, info = code_env.reset(seed=1)
    assert _images(obs["full_prompt"]) == [] and info["condition_frame_indices"] == []
    assert "video" not in _texts(obs["full_prompt"]).lower()


def _contains_secret(x, depth=0) -> bool:
    if depth > 6:
        return False
    if isinstance(x, str):
        return any(s in x for s in SECRETS)
    if isinstance(x, dict):
        return any(_contains_secret(k, depth + 1) or _contains_secret(v, depth + 1) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return any(_contains_secret(v, depth + 1) for v in x)
    return False


def test_no_privileged_info_in_prompt_or_sandbox(monkeypatch, patched_models):
    code_env = make_code_env(monkeypatch, task_id="VideoUnmaskSwap", n_demo=10)
    obs, info = code_env.reset(seed=1)
    assert "memory" not in obs and "obs_lists" not in obs
    assert not _contains_secret(obs["full_prompt"])
    assert info["memory"]["simple_subgoal_online"] == SECRETS[0]  # still logged for evaluation
    g = code_env._exec_globals
    assert "env" not in g
    assert not _contains_secret({k: v for k, v in g.items() if k != "__builtins__"})
    _, _, _, _, step_info = code_env.step(
        "o = get_observation()\nprint(sorted(o))\nprint(sorted(obs))\n"
        "try:\n    env\nexcept NameError:\n    print('NO_ENV')\n"
    )
    out = step_info["stdout"]
    assert "NO_ENV" in out and "memory" not in out and not _contains_secret(out)
    assert not _contains_secret(code_env._exec_globals.get("o"))
    json.dumps(step_info["memory"])  # evaluation log stays JSON-serializable (no frames)


# --------------------------------------------------------------------------------------------------
# Full trial with a scripted model
# --------------------------------------------------------------------------------------------------

def _reply(content: str) -> dict[str, Any]:
    return {"content": content, "reasoning": None, "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


ARGS = SimpleNamespace(
    model="fake", server_url="", api_key=None, max_tokens=100, temperature=1.0, reasoning_effort="low",
    debug=False, use_legacy_multi_turn_decision_prompt=False, visual_differencing_model="Qwen/Qwen3.8-27B",
    visual_differencing_model_server_url="", visual_differencing_model_api_key=None,
)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "output_dir": str(tmp_path), "use_parallel_ensemble": False, "use_multimodel": False,
        "use_visual_feedback": False, "use_img_differencing": True, "use_video_differencing": False,
        "use_wrist_camera": False, "record_video": True, "use_oracle_code": False, "save_multiturn_prompts": True,
    }


def test_trial_success_ends_with_reason_and_logs(monkeypatch, patched_models, tmp_path):
    prompts: list[Any] = []

    def fake_model(args, prompt):
        prompts.append((args.model, prompt))
        if args.model != "fake":  # VDM
            return _reply("The robot moved.")
        return _reply("```python\nclose_gripper()\nopen_gripper()\nclose_gripper()\n```")

    monkeypatch.setattr(trial_mod, "_query_model", fake_model)
    monkeypatch.setattr(trial_mod, "_write_video", lambda frames, base_dir, suffix="": None)
    code_env = make_code_env(monkeypatch, task_id="VideoUnmaskSwap", n_demo=12, success_at=20)
    summary = _run_single_trial(code_env, 1, ARGS, _config(tmp_path), "Executed:\n{executed_code}\n{console_stdout}\n{console_stderr}")
    trial_dir = next(tmp_path.glob("trial_01_*"))
    end = json.loads((trial_dir / "trial_end.json").read_text())
    assert end["reason"] == "success" and end["task_completed"] is True and end["sim_steps"] == 20
    assert end["turns"] == 1 and summary.task_completed
    responses = json.loads((trial_dir / "all_responses.json").read_text())
    assert responses[0]["vdm_usage"] == [{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}]
    assert responses[-1]["decision"] == "truncated" and responses[-1]["reason"] == "success"
    coding_prompt = [p for m, p in prompts if m == "fake"][0]
    assert len(_images(coding_prompt)) == 8
    assert not _contains_secret([p for _, p in prompts])
    # Video buffer: conditioning frames (subsampled, red border) come first.
    assert "taskcompleted_1" in trial_dir.name
