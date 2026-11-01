from __future__ import annotations

import base64
import io
import re
import time
from collections.abc import Callable
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
    description: str, n_frames: int, total: int, control_freq: float = CONTROL_FREQ_HZ, how: str | None = None
) -> list[dict[str, Any]]:
    """First-prompt content for the conditioning video as VDM text (replaces conditioning_video_content).

    ``how``: how the frames were watched (default: the single-call wording, ``n_frames`` sampled uniformly).
    """
    duration = total / control_freq
    if how is None:
        how = (f"{n_frames} frames sampled uniformly from it in time order (including the first and last frame)")
    return [
        {
            "type": "text",
            "text": (
                f"Before you take control, this task showed a video from the front camera (the 'agentview' camera) "
                f"of {total} frames ({duration:.1f} s at {control_freq:g} frames per second). You cannot see it. "
                f"A vision model watched {how} and described it as below. The video is described only here, once; "
                "the current scene is what get_observation() returns.\n"
                f"<conditioning_video_description>\n{description.strip()}\n</conditioning_video_description>"
            ),
        }
    ]


# ---------------------------------------------------------------------------------------------------------------
# Segmented description (condition_video_segment_frames > 0): every ``stride``-th frame, in overlapping parts, one VDM
# call per part; each part gets the object state at the end of the previous part and continues from it.
# ---------------------------------------------------------------------------------------------------------------

CONDITION_VIDEO_SEGMENT_DEFAULTS = {"stride": 1, "segment_frames": 20, "overlap": 4, "max_segments": 12}

CONDITION_VIDEO_SEGMENT_SYSTEM = (
    "You describe a robot task's demonstration video for another model that cannot see it. The video is shown "
    "to you in consecutive parts. That model will control the robot afterwards and will only have your text, so "
    "it must learn from you everything the video shows that could matter for the task. Report only what is "
    "visible. Do not say what the answer is, which object is the target, what the robot should do next or what "
    "the subgoals are. Do *NOT* write any code."
)

CONDITION_VIDEO_SEGMENT_INSTRUCTIONS = (
    "Track every object individually through the frames: its identity (color and kind, e.g. 'red cube', 'blue "
    "cup', 'left button') and its position (relative to the other objects, and left / right / front / back as "
    "seen from this camera) in each frame. Objects may move without the robot touching them (containers can "
    "slide past each other and trade places), so compare each object's position frame by frame instead of "
    "describing the scene as a whole. Report events with their times (t = ... s): an object covered or "
    "uncovered, containers swapped (say which container went from where to where), an object lifted, carried, "
    "placed (where, and in which order relative to earlier placements), a highlight or light appearing or "
    "disappearing, a button pressed, a path traced by the gripper or a stick (its shape and waypoints in order). "
    "Count repeated actions. If something cannot be seen (occluded, too fast, between frames), say so instead "
    "of guessing. Do not infer the goal's answer, the target object or a plan."
)

CONDITION_VIDEO_SEGMENT_FORMAT = (
    "Reply in exactly two sections:\n"
    "EVENTS:\n"
    "a numbered list of the events in this part, in time order, each with its time (t = ... s); write 'none' if "
    "nothing happens. Do not repeat events from the frames this part shares with the previous part.\n"
    "STATE:\n"
    "the state at the last frame of this part: each object with its identity and position; which container is "
    "over which object (or 'unknown'); objects placed so far and their order; highlights or buttons so far."
)

_STATE_HEADER = re.compile(r"^\s*[*#]*\s*STATE\s*[*#]*\s*:?\s*[*#]*\s*$", re.MULTILINE)
_EVENTS_HEADER = re.compile(r"^\s*[*#]*\s*EVENTS\s*[*#]*\s*:?\s*[*#]*\s*$", re.MULTILINE)


def strided_frame_indices(total: int, stride: int) -> list[int]:
    """Every ``stride``-th frame index of ``range(total)``, with the last frame always included."""
    if total <= 0:
        return []
    idx = list(range(0, total, max(1, int(stride))))
    if idx[-1] != total - 1:
        idx.append(total - 1)
    return idx


def split_into_segments(indices: list[int], segment_frames: int, overlap: int) -> list[list[int]]:
    """Consecutive parts of ``segment_frames`` indices; each part starts with the last ``overlap`` of the previous.

    If fewer than half a step of new frames would be left for a last part, the current part takes them instead.
    """
    if not indices:
        return []
    size = max(1, int(segment_frames))
    overlap = min(max(0, int(overlap)), size - 1)
    segments, start = [], 0
    step = size - overlap
    while True:
        end = start + size
        # A last part with only a few new frames is merged into this one instead of costing its own call.
        if len(indices) - end < max(1, step // 2):
            end = len(indices)
        segments.append(indices[start:end])
        if end >= len(indices):
            return segments
        start += step


def plan_condition_video_segments(
    total: int, stride: int, segment_frames: int, overlap: int, max_segments: int
) -> tuple[int, list[list[int]]]:
    """(stride used, parts as frame-index lists). The stride grows until there are at most ``max_segments`` parts."""
    stride = max(1, int(stride))
    while True:
        segments = split_into_segments(strided_frame_indices(total, stride), segment_frames, overlap)
        if len(segments) <= max(1, int(max_segments)) or stride >= total:
            return stride, segments
        stride += 1


def split_segment_reply(text: str) -> tuple[str, str]:
    """(events, state) of a segment reply; without a STATE section both are the whole reply."""
    text = (text or "").strip()
    m = _STATE_HEADER.search(text)
    if m is None:
        return text, text
    events, state = text[: m.start()].strip(), text[m.end():].strip()
    events = _EVENTS_HEADER.sub("", events, count=1).strip()
    return events, state


def condition_video_segment_prompt(
    goal: str | None,
    frames: list[tuple[int, np.ndarray]],
    part: int,
    n_parts: int,
    total: int,
    stride: int,
    shared_with_previous: int,
    state_so_far: str | None,
    control_freq: float = CONTROL_FREQ_HZ,
) -> list[dict[str, Any]]:
    """VDM prompt for part ``part`` (1-based) of the conditioning video."""
    first, last = frames[0][0], frames[-1][0]
    every = "every frame" if stride == 1 else f"every {stride}th frame"
    content: list[dict[str, Any]] = []
    if goal:
        content.append({"type": "text", "text": f"The robot's task (for context only; do not solve it): {goal}"})
    content.append({"type": "text", "text": CONDITION_VIDEO_SEGMENT_INSTRUCTIONS})
    intro = (
        f"The video is from the front camera (the 'agentview' camera) and has {total} frames "
        f"({total / control_freq:.1f} s at {control_freq:g} frames per second). This is part {part} of {n_parts}: "
        f"frames {first + 1}-{last + 1} (t = {first / control_freq:.2f}-{last / control_freq:.2f} s), {every}, "
        "in time order."
    )
    if part > 1 and shared_with_previous:
        intro += (f" Its first {shared_with_previous} frames are the last frames of part {part - 1}; events in them "
                  "are already described.")
    content.append({"type": "text", "text": intro})
    if state_so_far:
        content.append({
            "type": "text",
            "text": ("State at the end of the previous part, as you described it. Continue from it and keep each "
                     f"object's identity consistent:\n<state_so_far>\n{state_so_far}\n</state_so_far>"),
        })
    for idx, frame in frames:
        content.append({"type": "text", "text": f"Frame {idx + 1} of {total} (t = {idx / control_freq:.2f} s):"})
        content.append({"type": "image_url", "image_url": {"url": _png_data_url(frame)}})
    content.append({"type": "text", "text": CONDITION_VIDEO_SEGMENT_INSTRUCTIONS})
    content.append({"type": "text", "text": CONDITION_VIDEO_SEGMENT_FORMAT})
    return [
        {"role": "system", "content": CONDITION_VIDEO_SEGMENT_SYSTEM},
        {"role": "user", "content": content},
    ]


TRUNCATED_MARK = "[truncated: the description of this part was cut off by the output token limit]"


def assemble_condition_video_description(parts: list[dict[str, Any]], control_freq: float = CONTROL_FREQ_HZ) -> str:
    """The text that goes into the first prompt: each part's events in order, then the final state.

    A part whose reply was cut off at max_tokens is marked TRUNCATED_MARK. The final state is the last complete
    part's STATE; if that is not the last part, the heading says from which part it is.
    """
    lines = []
    for p in parts:
        lines.append(f"Part {p['part']}/{len(parts)} (t = {p['first'] / control_freq:.2f}-"
                     f"{p['last'] / control_freq:.2f} s):")
        lines.append(p["events"] or "none")
        if p.get("truncated"):
            lines.append(TRUNCATED_MARK)
        lines.append("")
    complete = [p for p in parts if not p.get("truncated")]
    if not complete:
        lines.append("Final state: unknown (every part was cut off).")
    elif complete[-1] is parts[-1]:
        lines.append("Final state (end of the video):")
        lines.append(complete[-1]["state"])
    else:
        last = complete[-1]
        lines.append(f"Final state: unknown (the description of the last part(s) was cut off). State at the end of "
                     f"part {last['part']} (t = {last['last'] / control_freq:.2f} s):")
        lines.append(last["state"])
    return "\n".join(lines).strip()


def describe_condition_video_segmented(
    query: Callable[[Any, list[dict[str, Any]]], dict[str, Any]],
    vdm_args: Any,
    goal: str | None,
    all_frames: list[np.ndarray],
    *,
    stride: int,
    segment_frames: int,
    overlap: int,
    max_segments: int,
    control_freq: float = CONTROL_FREQ_HZ,
) -> tuple[str, dict[str, Any]]:
    """Describe the whole conditioning video part by part. Returns (description, log).

    ``query(vdm_args, prompt)`` returns a query_model-style dict. An empty or cut-off reply for a part is asked once
    more; empty twice raises RuntimeError.
    """
    total = len(all_frames)
    stride_used, segments = plan_condition_video_segments(total, stride, segment_frames, overlap, max_segments)
    parts: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    state: str | None = None
    prev: list[int] = []
    for k, seg in enumerate(segments, start=1):
        shared = len(set(seg) & set(prev))
        prompt = condition_video_segment_prompt(
            goal, [(i, all_frames[i]) for i in seg], k, len(segments), total, stride_used, shared, state, control_freq)
        out: dict[str, Any] = {}
        for attempt in range(2):
            start = time.time()
            out = query(vdm_args, prompt)
            calls.append({"part": k, "attempt": attempt + 1, "frames": [seg[0], seg[-1]], "n_frames": len(seg),
                          "latency_s": round(time.time() - start, 2), "finish_reason": out.get("finish_reason"),
                          "usage": out.get("usage")})
            if (out.get("content") or "").strip() and out.get("finish_reason") != "length":
                break
        reply = (out.get("content") or "").strip()
        if not reply:
            raise RuntimeError(f"VDM gave no description for conditioning-video part {k}/{len(segments)} (twice)")
        truncated = out.get("finish_reason") == "length"  # cut off on both attempts
        events, part_state = split_segment_reply(reply)
        parts.append({"part": k, "first": seg[0], "last": seg[-1], "n_frames": len(seg), "shared": shared,
                      "events": events, "state": part_state, "reply": reply, "reasoning": out.get("reasoning"),
                      "truncated": truncated, "state_in": state})
        if not truncated:  # a cut-off STATE is never passed on: the next part gets the last complete one
            state = part_state
        prev = seg
    description = assemble_condition_video_description(parts, control_freq)
    used = sorted({i for seg in segments for i in seg})
    log = {
        "segmentation": {"stride": stride_used, "requested_stride": int(stride), "segment_frames": int(segment_frames),
                         "overlap": int(overlap), "max_segments": int(max_segments), "n_segments": len(segments),
                         "segments": [[seg[0], seg[-1], len(seg)] for seg in segments]},
        "n_frames": len(used),
        "frame_indices": used,
        "vdm_calls": calls,
        "parts": parts,
        "truncated_parts": [p["part"] for p in parts if p["truncated"]],
        "how": (f"{len(used)} of its frames ({'every frame' if stride_used == 1 else f'every {stride_used}th frame'}, "
                f"in {len(segments)} overlapping parts, in time order)"),
    }
    return description, log


def describe_condition_video_single(
    query: Callable[[Any, list[dict[str, Any]]], dict[str, Any]],
    vdm_args: Any,
    goal: str | None,
    frames: list[tuple[int, np.ndarray]],
    total: int,
    control_freq: float = CONTROL_FREQ_HZ,
) -> tuple[str, dict[str, Any]]:
    """The first version (condition_video_segment_frames 0): one VDM call with the sampled ``frames``.

    An empty or cut-off reply is asked once more; empty twice raises RuntimeError, cut off twice is used with
    TRUNCATED_MARK appended (``truncated`` in the log).
    """
    prompt = condition_video_vdm_prompt(goal, frames, total, control_freq)
    calls: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    for attempt in range(2):
        start = time.time()
        out = query(vdm_args, prompt)
        calls.append({"attempt": attempt + 1, "n_frames": len(frames), "latency_s": round(time.time() - start, 2),
                      "finish_reason": out.get("finish_reason"), "usage": out.get("usage")})
        if (out.get("content") or "").strip() and out.get("finish_reason") != "length":
            break
    description = (out.get("content") or "").strip()
    if not description:
        raise RuntimeError(f"VDM {getattr(vdm_args, 'model', '?')} gave no conditioning-video description (twice)")
    truncated = out.get("finish_reason") == "length"
    if truncated:
        description += "\n" + TRUNCATED_MARK
    return description, {"vdm_calls": calls, "vdm_reasoning": out.get("reasoning"), "truncated": truncated,
                         "n_frames": len(frames), "frame_indices": [i for i, _ in frames]}


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
            self._condition_low = low
        info["condition_frame_indices"] = [i for i, _ in frames]
        info.setdefault("memory", {})
        if isinstance(info["memory"], dict):
            info["memory"]["condition_frame_indices"] = [i for i, _ in frames]
        return obs, info

    def condition_video_vdm_prompt(self, goal: str | None) -> list[dict[str, Any]] | None:
        """VDM prompt describing this episode's conditioning video, or None if there is none."""
        cv = getattr(self, "condition_video", None)
        return condition_video_vdm_prompt(goal, cv["frames"], cv["total"]) if cv else None

    def condition_video_all_frames(self) -> list[np.ndarray]:
        """Every front-camera frame of this episode's conditioning video (empty if there is none)."""
        if not getattr(self, "condition_video", None):
            return []
        return list(self._condition_low.handle.conditioning_frames("front"))

    def use_condition_video_text(self, obs: dict[str, Any], description: str, how: str | None = None) -> None:
        """Replace the conditioning-video images at the end of ``obs``'s first prompt with ``description``."""
        cv = self.condition_video
        prompt = [dict(m) for m in obs["full_prompt"]]
        content = list(prompt[-1]["content"])
        video = content[len(content) - cv["n_items"]:]
        if len(video) != cv["n_items"] or not any(c.get("type") == "image_url" for c in video):
            raise RuntimeError("conditioning video is no longer at the end of the first prompt")
        text = condition_video_text_content(description, len(cv["frames"]), cv["total"], how=how)
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
    "CONDITION_VIDEO_SEGMENT_DEFAULTS",
    "TRUNCATED_MARK",
    "RoboMMECodeEnv",
    "describe_condition_video_segmented",
    "describe_condition_video_single",
    "plan_condition_video_segments",
    "split_segment_reply",
    "condition_video_text_content",
    "condition_video_vdm_prompt",
    "conditioning_video_content",
]
