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


class RoboMMECodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for RoboMME tasks.

    Prompt and APIs come from the YAML config; the task is selected by ``task_id`` /
    ``episode_idx`` on the low-level env (``RoboMMEEnv``).

    Conditioning video (Video* / Imitation tasks): ``reset()`` appends ``condition_frames`` front-camera frames
    (low-level env option, default 8), uniformly sampled with their times, to the **first** prompt only
    (the returned observation's ``full_prompt``). Multi-turn decision prompts are built from the env's own
    prompt, which never holds them: the memory-less baseline sees the video once.

    Episode end: when RoboMME reports a terminal status (or the step cap is reached) the low-level env
    latches ``episode_end_reason``; ``step()`` then reports ``truncated`` so the trial stops.
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        low = self.low_level_env
        frames = low.conditioning_frames() if hasattr(low, "conditioning_frames") else []
        if frames:
            content = conditioning_video_content(frames, low.num_conditioning_frames())
            prompt = [dict(m) for m in obs["full_prompt"]]
            prompt[-1] = {**prompt[-1], "content": list(prompt[-1]["content"]) + content}
            obs["full_prompt"] = prompt
        info["condition_frame_indices"] = [i for i, _ in frames]
        info.setdefault("memory", {})
        if isinstance(info["memory"], dict):
            info["memory"]["condition_frame_indices"] = [i for i, _ in frames]
        return obs, info

    def step(self, action: str) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        reason = getattr(self.low_level_env, "episode_end_reason", None)
        if reason is not None:
            truncated = True
            info["episode_end_reason"] = reason
        return obs, reward, terminated, truncated, info


__all__ = ["RoboMMECodeEnv", "conditioning_video_content"]
