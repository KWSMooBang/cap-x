from __future__ import annotations

import base64
import io
from typing import Any, SupportsFloat

import numpy as np
from PIL import Image

from capx.envs.tasks.base import CodeExecutionEnvBase

CONTROL_FREQ_HZ = 20  # capx.integrations.robomme.CONTROL_FREQ_HZ (not imported: keeps this module robomme-free)


def _png_data_url(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def conditioning_video_content(
    frames: list[tuple[int, np.ndarray]], total: int, control_freq: float = CONTROL_FREQ_HZ
) -> list[dict[str, Any]]:
    """Prompt content items for the conditioning video: one intro text, then a caption and image per frame."""
    if not frames:
        return []
    duration = total / control_freq
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Before you take control, this task showed a video from the front camera (the 'agentview' camera) "
                f"of {total} frames ({duration:.1f} s at {control_freq:g} frames per second). "
                f"Below are {len(frames)} frames sampled uniformly from it in time order, including the first and "
                "last frame. The video is shown only here, once; the current scene is what get_observation() returns."
            ),
        }
    ]
    for n, (idx, frame) in enumerate(frames, start=1):
        content.append(
            {"type": "text", "text": f"Video frame {n}/{len(frames)} (t = {idx / control_freq:.2f} s, frame {idx + 1} of {total}):"}
        )
        content.append({"type": "image_url", "image_url": {"url": _png_data_url(frame)}})
    return content


CONDITION_VIDEO_VDM_SYSTEM = (
    "You describe a robot task's demonstration video for another model that cannot see it. That model will "
    "control the robot afterwards and will only have your text, so it must learn from you everything the video "
    "shows that could matter for the task. Report only what is visible. Do not say what the answer is, which "
    "object is the target, what the robot should do next or what the subgoals are. Do *NOT* write any code."
)

CONDITION_VIDEO_VDM_INSTRUCTIONS = (
    "Describe the video in time order, as a numbered list of events with their times (t = ... s). For each event "
    "say which object moved (color, shape, size, any marking, and which of several similar objects it is), from "
    "where to where (relative to the other objects, and left / right / front / back as seen from this camera), "
    "and what touched or moved it. Track hidden objects: say which container or cover is over which object, "
    "every time one is lifted, put down or swapped, and where each container is at the end. Count repeated "
    "actions and give their order. For movements that trace a path (a stick, a button press sequence, a route), "
    "describe the path's shape and its waypoints in order. Mention highlights, lights or anything that appears "
    "or disappears, and when. Then describe the last frame: where every object is. If something is unclear "
    "(occluded, between sampled frames, too small), say so instead of guessing. Do not infer the goal's answer, "
    "the target object or a plan; describe only what happened."
)


def condition_video_vdm_prompt(
    goal: str | None, frames: list[tuple[int, np.ndarray]], total: int, control_freq: float = CONTROL_FREQ_HZ
) -> list[dict[str, Any]]:
    """Prompt for the VDM to describe the conditioning video once (condition_video_mode vdm_text).

    The frames go in exactly as in the first prompt of the images mode (conditioning_video_content).
    """
    content: list[dict[str, Any]] = []
    if goal:
        content.append({"type": "text", "text": f"The robot's task (for context only; do not solve it): {goal}"})
    content.append({"type": "text", "text": CONDITION_VIDEO_VDM_INSTRUCTIONS})
    content.append({
        "type": "text",
        "text": (
            f"The video is from the front camera (the 'agentview' camera) and has {total} frames "
            f"({total / control_freq:.1f} s at {control_freq:g} frames per second). Below are {len(frames)} frames "
            "sampled uniformly from it in time order, including the first and last frame."
        ),
    })
    content.extend(conditioning_video_content(frames, total, control_freq)[1:])  # captions + images, no intro
    content.append({"type": "text", "text": CONDITION_VIDEO_VDM_INSTRUCTIONS})
    return [
        {"role": "system", "content": CONDITION_VIDEO_VDM_SYSTEM},
        {"role": "user", "content": content},
    ]


def condition_video_text_content(
    description: str, n_frames: int, total: int, control_freq: float = CONTROL_FREQ_HZ
) -> list[dict[str, Any]]:
    """First-prompt content for the conditioning video as VDM text (replaces conditioning_video_content)."""
    duration = total / control_freq
    return [
        {
            "type": "text",
            "text": (
                f"Before you take control, this task showed a video from the front camera (the 'agentview' camera) "
                f"of {total} frames ({duration:.1f} s at {control_freq:g} frames per second). You cannot see it. "
                f"A vision model watched {n_frames} frames sampled uniformly from it in time order (including the "
                "first and last frame) and described it as below. The video is described only here, once; the "
                "current scene is what get_observation() returns.\n"
                f"<conditioning_video_description>\n{description.strip()}\n</conditioning_video_description>"
            ),
        }
    ]


class RoboMMECodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for RoboMME tasks.

    Prompt and APIs come from the YAML config; the task is selected by ``task_id`` /
    ``episode_idx`` on the low-level env (``RoboMMEEnv``).

    Conditioning video (Video* / Imitation tasks): ``reset()`` appends ``condition_frames`` front-camera frames
    (low-level env option, default 32; env var CAPX_ROBOMME_CONDITION_FRAMES overrides it), uniformly sampled with their times, to the **first** prompt only
    (the returned observation's ``full_prompt``). Multi-turn decision prompts are built from the env's own
    prompt, which never holds them: the memory-less baseline sees the video once.

    ``condition_video`` (after reset): the sampled frames, the video length and how many content items they
    took at the end of the first prompt, or None. ``use_condition_video_text`` swaps those items for a VDM
    description (condition_video_mode vdm_text, set in capx/envs/trial.py).

    Episode end: when RoboMME reports a terminal status (or the step cap is reached) the low-level env
    latches ``episode_end_reason``; ``step()`` then reports ``truncated`` so the trial stops.
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        low = self.low_level_env
        frames = low.conditioning_frames() if hasattr(low, "conditioning_frames") else []
        self.condition_video = None
        if frames:
            total = low.num_conditioning_frames()
            content = conditioning_video_content(frames, total)
            prompt = [dict(m) for m in obs["full_prompt"]]
            prompt[-1] = {**prompt[-1], "content": list(prompt[-1]["content"]) + content}
            obs["full_prompt"] = prompt
            self.condition_video = {"frames": frames, "total": total, "n_items": len(content)}
        info["condition_frame_indices"] = [i for i, _ in frames]
        info.setdefault("memory", {})
        if isinstance(info["memory"], dict):
            info["memory"]["condition_frame_indices"] = [i for i, _ in frames]
        return obs, info

    def condition_video_vdm_prompt(self, goal: str | None) -> list[dict[str, Any]] | None:
        """VDM prompt describing this episode's conditioning video, or None if there is none."""
        cv = getattr(self, "condition_video", None)
        return condition_video_vdm_prompt(goal, cv["frames"], cv["total"]) if cv else None

    def use_condition_video_text(self, obs: dict[str, Any], description: str) -> None:
        """Replace the conditioning-video images at the end of ``obs``'s first prompt with ``description``."""
        cv = self.condition_video
        prompt = [dict(m) for m in obs["full_prompt"]]
        content = list(prompt[-1]["content"])
        video = content[len(content) - cv["n_items"]:]
        if len(video) != cv["n_items"] or not any(c.get("type") == "image_url" for c in video):
            raise RuntimeError("conditioning video is no longer at the end of the first prompt")
        text = condition_video_text_content(description, len(cv["frames"]), cv["total"])
        prompt[-1] = {**prompt[-1], "content": content[: len(content) - cv["n_items"]] + text}
        obs["full_prompt"] = prompt

    def step(self, action: str) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        reason = getattr(self.low_level_env, "episode_end_reason", None)
        if reason is not None:
            truncated = True
            info["episode_end_reason"] = reason
        return obs, reward, terminated, truncated, info


__all__ = [
    "RoboMMECodeEnv",
    "condition_video_text_content",
    "condition_video_vdm_prompt",
    "conditioning_video_content",
]
