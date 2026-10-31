"""Verification VDM for the estimated Belief Graph (``bg_est``).

Separate from cap-x's free-text VDM description: once per turn, the VDM gets the agentview image
before and after the turn plus a numbered list of hypotheses (this turn's expected effects and
goal-related predicates the geometry is unsure about) and answers each with true / false /
not_visible as JSON. Objects are named by alias with their pixel position in the images. A reply
that does not parse is ignored (the graph is then updated from geometry alone).
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .goal import GoalParseError, _content, _extract_json

ANSWERS = ("true", "false", "not_visible")

VERIFY_SYSTEM = (
    "You check statements about a robot tabletop scene against camera images. Judge only what the "
    "images show. Answer with JSON only."
)


@dataclass
class Hypothesis:
    id: int
    key: tuple[str, str, str | None]  # edge key (predicate, subject, target)
    text: str
    why: str  # expected | uncertain


@dataclass
class VerifyResult:
    answers: dict[int, str] = field(default_factory=dict)  # hypothesis id -> true/false/not_visible
    ok: bool = False
    reply: str = ""
    error: str | None = None
    usage: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "answers": {str(k): v for k, v in self.answers.items()},
                "error": self.error, "reply": self.reply}


def pred_text(key: tuple[str, str, str | None]) -> str:
    pred, subj, target = key
    if pred == "inhand":
        return f"{subj} is held in the robot gripper"
    if pred == "in":
        return f"{subj} is inside {target}"
    if target == "table":
        return f"{subj} stands directly on the table (not on another object)"
    return f"{subj} is on top of {target}"


def encode_png(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def verify_prompt(
    instruction: str,
    before_image: str | None,
    after_image: str,
    hypotheses: Sequence[Hypothesis],
    object_refs: Sequence[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": f"Task: {instruction}"}]
    if before_image is not None:
        content += [{"type": "text", "text": "Before the robot's last action:"},
                    {"type": "image_url", "image_url": {"url": before_image}}]
    content += [{"type": "text", "text": "Now (after the action):"},
                {"type": "image_url", "image_url": {"url": after_image}}]
    statements = "\n".join(f"{h.id}. {h.text}" for h in hypotheses)
    refs = "\n".join(object_refs)
    content.append({"type": "text", "text": (
        "Objects (pixel x right, y down in the images; positions are estimates):\n"
        f"{refs}\n\n"
        "For each statement, say whether it is true NOW (in the 'Now' image): \"true\", \"false\", or "
        "\"not_visible\" if the image does not let you tell.\n"
        f"{statements}\n\n"
        'Answer JSON only: {"answers": [{"id": 1, "answer": "true"}, ...]}'
    )})
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": content}]


def parse_verify_reply(text: str, ids: Sequence[int]) -> dict[int, str]:
    data = _extract_json(text)
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, list):
        raise GoalParseError("missing 'answers' list")
    out: dict[int, str] = {}
    for a in answers:
        if not isinstance(a, dict):
            continue
        try:
            i = int(a.get("id"))
        except (TypeError, ValueError):
            continue
        ans = str(a.get("answer", "")).strip().lower().replace(" ", "_")
        if i in ids and ans in ANSWERS:
            out[i] = ans
    if not out:
        raise GoalParseError("no usable answers")
    return out


class VdmVerifier:
    """``query_fn(prompt) -> str | {"content", "usage"}`` is the VDM (``--visual-differencing-model``)."""

    def __init__(self, query_fn: Callable[[list[dict[str, Any]]], Any], max_hypotheses: int = 8) -> None:
        self.query_fn = query_fn
        self.max_hypotheses = max_hypotheses
        self.calls = 0

    def verify(
        self,
        instruction: str,
        before_image: str | None,
        after_image: str,
        hypotheses: Sequence[Hypothesis],
        object_refs: Sequence[str],
    ) -> VerifyResult:
        hyps = list(hypotheses)[: self.max_hypotheses]
        result = VerifyResult()
        if not hyps:
            return result
        self.calls += 1
        try:
            result.reply, result.usage = _content(self.query_fn(
                verify_prompt(instruction, before_image, after_image, hyps, object_refs)))
            result.answers = parse_verify_reply(result.reply, [h.id for h in hyps])
            result.ok = True
        except Exception as exc:  # unusable reply or server error: geometry only
            result.error = str(exc)
        return result
