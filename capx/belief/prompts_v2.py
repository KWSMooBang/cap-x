"""Prompt texts of the estimated Belief Graph v2 (``bg_est2``), in one place.

Source: /outputs/cap-bg/tasks/bg-est2-design.md (the 「추정 BG v2 설계」 design; that file wins over
the artifact). Only the ``{...}`` slots are filled in here; the wording is the design's.

* M2 -- initial Observe (once per trial, the VDM): one marked image, all candidate statements.
* M3 -- goal parsing (once per trial, the goal model, text only): a viewpoint-explicit scene
  description; each step lists only the predicates that BECOME true.
* M4 -- Subtask / Effect declaration rules appended to the first prompt and every decision prompt
  (``bg_est2`` only).
* M5 -- turn Observe (once per turn, the VDM; replaces v1's verification VDM): marked images before
  and now, the declared Subtask / Effect, and the statements to answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

OBSERVE_SYSTEM = (
    "You are the visual observer for a robot's belief graph. Report only what the image shows about the "
    "given statements. Use the exact output format."
)

VIEW_TEXT = (
    "The image is the front camera view. The robot stands at the top of the image, so the robot's left is "
    "the image's right."
)

OUTPUT_RULES = (
    "Output exactly one line, in the same order:\n"
    "Observe: (statement) value | (statement) value | ...\n"
    'Then, only if needed, one line starting with "Notes:" for things the statements miss (an object tilted, '
    "fallen, or a mark on the wrong object)."
)

OUTPUT_RULES_TURN = (
    "Output exactly one line, in the same order:\n"
    "Observe: (statement) value | ...\n"
    'Then, only if needed, one line starting with "Notes:".'
)

GOAL_SYSTEM_V2 = "You turn a robot task instruction into goal steps over the named objects. Answer with JSON only."

GOAL_RULES_V2 = """Write the goal as ordered steps. Each step lists only the predicates that BECOME true at that step; do not repeat predicates from earlier steps.
- ["on", X, Y]: X rests on Y (Y may be "table"); ["in", X, Y]: X is inside container Y.
- "N times" is one step with "repeat": N. Putting an object back counts only after it was picked up.
- If several orders or choices satisfy the instruction, give each as a branch.
- If direction words (left, right, nearest) can be read more than one way, give each reading as a branch.
Format: {"goal": {"type": "or", "branches": [{"steps": [{"preds": [["on", "bowl_A", "plate_A"]], "repeat": 1}, ...]}, ...]}}"""

M4_RULES = """Start every code block with two comment lines that state what the block is meant to achieve, using the object names from the belief graph:
# Subtask: <op>(<args>)      op is one of: pick, place, put_in, place_region, look
# Effect: (<predicate>) <before>><after> | ...
These lines are not executed. After the block runs, the camera checks them, and any mismatch is reported back to you."""

NO_DECLARATION = "The last code block made no declaration."


def _image(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def mark_list_lines(marks: Sequence[tuple[str, str, str, tuple[float, float]]]) -> str:
    """M2 mark table: ``(mark, alias, category prompt, (u, v))`` -> ``- A = bowl_A (black bowl), mark at pixel (u, v)``."""
    return "\n".join(f"- {m} = {a} ({c}), mark at pixel ({u:.0f}, {v:.0f})" for m, a, c, (u, v) in marks)


def marks_now_text(shown: Sequence[tuple[str, str, tuple[float, float]]], missing: Sequence[str]) -> str:
    """M5 ``Marks now:`` value: ``A = bowl_A (pixel 250, 205) | 1 = plate_A (250, 214) | B, 2: not detected``.
    (The design writes the first entry with ``pixel``, the rest without.)"""
    parts = []
    for i, (m, a, (u, v)) in enumerate(shown):
        parts.append(f"{m} = {a} ({'pixel ' if i == 0 else ''}{u:.0f}, {v:.0f})")
    if missing:
        parts.append(f"{', '.join(missing)}: not detected")
    return " | ".join(parts) if parts else "none detected"


def observe_initial_prompt(
    instruction: str,
    image: str,
    marks: Sequence[tuple[str, str, str, tuple[float, float]]],
    statements: Sequence[str],
) -> list[dict[str, Any]]:
    """M2: the first marked image and every candidate statement."""
    before = f"Task: {instruction}"
    after = (
        f"{VIEW_TEXT}\n"
        "Each detected object is labeled with a mark at its center:\n"
        f"{mark_list_lines(marks)}\n"
        "For each statement, answer 1 if it is true in the image, 0 if it is false, ? if the image does not let "
        "you tell (hidden, out of view, unclear).\n"
        "Statements:\n" + "\n".join(statements) + "\n" + OUTPUT_RULES
    )
    return [
        {"role": "system", "content": OBSERVE_SYSTEM},
        {"role": "user", "content": [_text(before), _image(image), _text(after)]},
    ]


def observe_turn_prompt(
    instruction: str,
    image_before: str | None,
    image_now: str,
    marks_now: str,
    declared: tuple[str, str] | None,
    statements: Sequence[str],
) -> list[dict[str, Any]]:
    """M5: images before / now, marks now, the declaration (``(Subtask, Effect)`` text or None)."""
    head = (
        f"Task: {instruction}\n"
        "Image 1: before the last code block. Image 2: now. Front camera; the robot is at the top of the image. "
        "Marks show each object's name at its estimated position (marks for objects not detected now are omitted "
        "from image 2)."
    )
    content: list[dict[str, Any]] = [_text(head)]
    if image_before is not None:
        content.append(_image(image_before))
    content.append(_image(image_now))
    decl = (f"The last code block declared:\nSubtask: {declared[0]}\nEffect: {declared[1]}"
            if declared is not None else NO_DECLARATION)
    tail = (
        f"Marks now: {marks_now}\n"
        f"{decl}\n"
        "For each statement, answer for NOW (image 2): 1 true, 0 false, ? cannot tell.\n"
        "Statements:\n" + "\n".join(statements) + "\n" + OUTPUT_RULES_TURN
    )
    content.append(_text(tail))
    return [{"role": "system", "content": OBSERVE_SYSTEM}, {"role": "user", "content": content}]


def goal_prompt_v2(instruction: str, scene_lines: Sequence[str]) -> list[dict[str, Any]]:
    """M3: task instruction and the viewpoint-explicit scene description (``- alias (category): ...``)."""
    user = (
        f"Task instruction: {instruction}\n"
        'Scene, as seen in the front camera image (the robot is at the top of the image; "left" and "right" are '
        "image left and right):\n" + "\n".join(scene_lines) + "\n" + GOAL_RULES_V2
    )
    return [
        {"role": "system", "content": GOAL_SYSTEM_V2},
        {"role": "user", "content": [_text(user)]},
    ]
