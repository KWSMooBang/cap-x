"""RoboMME (arXiv 2603.04639) task loading.

RoboMME wraps ManiSkill3 (SAPIEN) tasks in ``BenchmarkEnvBuilder``. Every observation value is
a *list* of frames: ``reset()`` returns the whole conditioning sequence (e.g. the demonstration
video in Video* tasks) and each ``step()`` returns the frames produced by that action. Oracle
subgoals and status are reported in ``info``. See ``capx/third_party/RoboMME/doc/env_format.md``.

SAPIEN needs a Vulkan device even to build a scene (``obs_mode`` is fixed to
``rgb+depth+segmentation`` by the builder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ROBOMME_TASKS = [
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill",
    "VideoUnmaskSwap", "VideoUnmask", "ButtonUnmaskSwap", "ButtonUnmask",
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight",
    "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
]  # BenchmarkEnvBuilder.get_task_list() order

# info keys that carry task progress. Oracle / evaluation-only: they go to the trial logs (info["memory"]),
# never into a prompt or the generated-code sandbox (task_goal reaches the prompt only as the goal line).
MEMORY_INFO_KEYS = ("task_goal", "simple_subgoal_online", "grounded_subgoal_online", "status")

# Suites (README of RoboMME) in benchmark task order.
ROBOMME_SUITES = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "Permanence": ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
TASK_SUITE = {task: suite for suite, tasks in ROBOMME_SUITES.items() for task in tasks}
# These two hold a stick instead of the gripper: 7-D actions (the wrapper drops the 8th value).
STICK_TASKS = ("PatternLock", "RouteStick")
# Step budget used for the MME-VLA results in the RoboMME paper (scripts/evaluation.py); the CVPR 2026
# challenge uses 1500. One step = one ManiSkill control step (pd_joint_pos, 20 Hz).
DEFAULT_MAX_STEPS = 1300
CONTROL_FREQ_HZ = 20


def _json_safe(x: Any) -> Any:
    x = to_numpy(x)
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 16 else f"<array {x.shape}>"
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return x


def to_numpy(x: Any) -> Any:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


def _lists_to_numpy(obs: dict[str, list]) -> dict[str, list]:
    return {k: [to_numpy(v) for v in vs] if isinstance(vs, list) else to_numpy(vs) for k, vs in obs.items()}


@dataclass
class RoboMMEHandle:
    env: Any
    task_id: str
    episode_idx: int
    dataset: str
    reset_obs: dict[str, list] = field(default_factory=dict)
    last_obs: dict[str, list] = field(default_factory=dict)
    last_info: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> tuple[dict[str, list], dict[str, Any]]:
        obs, info = self.env.reset()
        self.reset_obs = _lists_to_numpy(obs)
        self.last_obs, self.last_info = self.reset_obs, info
        return self.reset_obs, info

    def step(self, action: np.ndarray | dict) -> tuple[dict[str, list], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs, self.last_info = _lists_to_numpy(obs), info
        return self.last_obs, float(to_numpy(reward)), bool(to_numpy(terminated)), bool(to_numpy(truncated)), info

    def task_goal(self) -> str:
        goal = self.last_info.get("task_goal", "")
        return goal[0] if isinstance(goal, (list, tuple)) and goal else str(goal)

    @property
    def task_language(self) -> str:
        # Read by capx.envs.trial._patch_libero_goal to fill {libero_environment_goal} in prompts.
        return self.task_goal()

    def num_conditioning_frames(self) -> int:
        """Frames returned at reset before the current one (the conditioning video; 0 for most tasks).

        RoboMME's own evaluation marks every reset frame but the last as demonstration
        (``scripts/evaluation.py``: ``is_video_demo = i < len(rgb_list) - 1``).
        """
        return max(0, len(self.reset_obs.get("front_rgb_list", [])) - 1)

    def conditioning_frames(self, camera: str = "front") -> list[np.ndarray]:
        frames = self.reset_obs.get(f"{camera}_rgb_list", [])
        return [np.asarray(f) for f in frames[:-1]]

    def memory_state(self) -> dict[str, Any]:
        """RoboMME task-progress info (oracle subgoals, status) for evaluation logs; JSON-safe, no frames."""
        state = {k: _json_safe(self.last_info.get(k)) for k in MEMORY_INFO_KEYS}
        state["num_conditioning_frames"] = self.num_conditioning_frames()
        return state


def load_robomme_task(
    task_id: str,
    episode_idx: int = 0,
    dataset: str = "test",
    action_space: str = "joint_angle",
    max_steps: int = DEFAULT_MAX_STEPS,
    **include_flags: bool,
) -> RoboMMEHandle:
    """Build one RoboMME episode env. ``include_flags`` are the builder's ``include_*`` switches."""
    try:
        from robomme.env_record_wrapper import BenchmarkEnvBuilder  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("RoboMME not available; run `uv sync --extra robomme`.") from e

    builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset=dataset,
        action_space=action_space,
        gui_render=False,
        max_steps=max_steps,
    )
    env = builder.make_env_for_episode(episode_idx, max_steps=max_steps, **include_flags)
    return RoboMMEHandle(env=env, task_id=task_id, episode_idx=episode_idx, dataset=dataset)
