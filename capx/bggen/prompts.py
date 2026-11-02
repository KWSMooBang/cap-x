"""Prompt texts of the BG generator.

* ``init_prompt``: instruction + marked first image + detected objects (alias, mark, 3D, viewpoint-explicit
  position words) -> JSON {remember, goal_branches, observe}.
* ``agent_prompt``: event + graph summary + goal + remember + re-checks + marked image -> JSON tool calls.
* ``observe_prompt``: the VDM Observe in bg_est2's M5 format (images before / now, ``Marks now``,
  statements, one ``Observe:`` line) keeping its "on = resting flat" sentence (``prompts_v2.ON_RULE``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from capx.belief.prompts_v2 import OBSERVE_SYSTEM, ON_RULE, OUTPUT_RULES_TURN

VIEWS = {
    # LIBERO-Mem agentview faces the robot: the robot's left is the image right.
    "front": "The image is the front camera view. The robot stands at the top of the image, so the robot's left is the image's right.",
    # B1K head camera looks forward from the robot: image left = robot's left.
    "head": "The image is the robot's head camera view (the robot looks into the scene), so image left is the robot's left.",
    "wrist": "The image is a wrist camera view, close to the gripper.",
}


def _image(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


# --- init ------------------------------------------------------------------------------------------

INIT_SYSTEM = ("You initialize a robot's belief graph from a task instruction and the first camera image. "
               "Answer with JSON only.")

INIT_RULES = """Return JSON with three fields:
"remember": short notes of what must be remembered for this task (e.g. "original plate of each bowl", "which plate was empty at the start").
"goal_branches": the goal as ordered steps over the object names above (and "table"). Each step lists only the relations that BECOME true at that step. If the instruction allows several orders or readings, give each as a separate branch.
"observe": statements to check in the first image now (where each movable object is, what is empty, open or on), at most {max_obs}.
Relations: {relations}.
Format:
{{"remember": ["..."], "goal_branches": [[["on", "bowl_A", "plate_A"]], [["on", "bowl_B", "plate_A"]]], "observe": ["(on bowl_A plate_B)", "(open fridge_A)"]}}
A branch is a list of steps; a step is one relation [pred, args...] or a list of relations that become true together."""

RELATIONS_TEXT = {
    "on": '["on", X, Y] X rests flat on Y (Y may be "table")',
    "in": '["in", X, Y] X is inside container Y',
    "at": '["at", X, "center"] X is in the named table region',
    "open": '["open", X] door / lid of X is open',
    "toggled_on": '["toggled_on", X] appliance X is switched on',
    "cooked": '["cooked", X] food X is cooked',
}


EXPECTED_TEXT = ("Objects the task names that are NOT visible yet (out of view or behind a door): {lines}. Use these names "
                 "for them; each name is bound to the object when it is first detected. Another not-yet-visible object of "
                 "one of these categories takes the next free letter (e.g. {example}).\n")


def init_prompt(instruction: str, image: str, object_lines: Sequence[str], view: str, relations: Sequence[str],
                max_obs: int = 20, expected: Sequence[str] = ()) -> list[dict[str, Any]]:
    """``expected``: "alias (category)" of objects the task names that were not detected in the first image."""
    rel = "; ".join(RELATIONS_TEXT[r] for r in relations if r in RELATIONS_TEXT)
    head = f"Task: {instruction}\n{VIEWS.get(view, view)}"
    exp = ""
    if expected:
        first = expected[0].split(" ")[0]
        example = first[:-1] + chr(ord(first[-1]) + 1) if first[-1:].isupper() and first[-1] != "Z" else first + "_B"
        exp = EXPECTED_TEXT.format(lines=", ".join(expected), example=example)
    tail = ("Detected objects (each has a mark at its center in the image; xyz in m):\n"
            + ("\n".join(object_lines) if object_lines else "(none)") + "\n" + exp
            + INIT_RULES.format(max_obs=max_obs, relations=rel))
    return [{"role": "system", "content": INIT_SYSTEM},
            {"role": "user", "content": [_text(head), _image(image), _text(tail)]}]


# --- agent -----------------------------------------------------------------------------------------

AGENT_SYSTEM = ("You keep a robot's belief graph up to date. At each event you choose which evidence to collect "
                "with the tools. You never write probabilities; a fixed verifier sets them. Answer with JSON only.")

TOOLS_TEXT = """Tools (robot_rel -- grippers holding / empty / lifted -- is always computed from the robot state):
- {{"tool": "observe", "statements": ["(on bowl_A plate_A)", ...]}}  the visual model answers 1 / 0 / ? from the marked image (at most {max_obs} statements)
- {{"tool": "geometry", "pairs": [["bowl_A", "plate_A"], ...]}}  3D / mask check of whether X rests flat on / inside Y
- {{"tool": "detect", "categories": ["{cat0}"]}}  detect a category again with its next text prompt (use when an object is missing; only these categories: {categories})
- {{"tool": "look", "camera": "<one of: {cameras}>"}}  use this camera's image for this event's observe
{hindsight}Optional "expect": [{{"statement": "(on bowl_A plate_A)", "value": 1}}] -- what this event should have caused (used only where the camera cannot tell).
Format: {{"calls": [...], "expect": [...], "why": "one sentence"}}"""

HINDSIGHT_TEXT = ('- {{"tool": "hindsight", "statements": [...], "after_s": 1.0}}  OFFLINE ONLY: observe a later frame of the '
                  "same quiet interval (used only to clean single-frame errors, never to fill unknowns)\n")


def agent_prompt(event_text: str, graph_text: str, goal_text: str, remember: Sequence[str], recheck: Sequence[str],
                 marks_text: str, suggestions: Sequence[str], image: str | None, cameras: Sequence[str],
                 offline: bool, max_obs: int = 10, categories: Sequence[str] = ()) -> list[dict[str, Any]]:
    """``categories``: the only categories ``detect`` accepts (the instruction's / init's / existing tracks')."""
    cats = list(categories) or ["bowl"]
    tools = TOOLS_TEXT.format(max_obs=max_obs, cameras=", ".join(cameras), cat0=cats[0], categories=", ".join(cats),
                              hindsight=HINDSIGHT_TEXT.format() if offline else "")
    lines = [f"[event] {event_text}", f"[graph] {graph_text}", f"[goal] {goal_text}",
             f"[remember] {' | '.join(remember) if remember else 'none'}",
             f"[re-check requested by the verifier] {' '.join(recheck) if recheck else 'none'}",
             f"[marks now] {marks_text}",
             f"[candidate statements] {' '.join(suggestions) if suggestions else 'none'}",
             tools, "Choose the tool calls that check the result of this event. Answer with JSON only."]
    content: list[dict[str, Any]] = [_text("\n".join(lines[:6]))]
    if image is not None:
        content.append(_image(image))
    content.append(_text("\n".join(lines[6:])))
    return [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": content}]


# --- observe (VDM, bg_est2 M5 format) -------------------------------------------------------------

def observe_prompt(instruction: str, image_before: str | None, image_now: str, marks_now: str, event_text: str,
                   statements: Sequence[str], view: str) -> list[dict[str, Any]]:
    head = (f"Task: {instruction}\n"
            + ("Image 1: the previous keyframe. Image 2: now. " if image_before else "Image: now. ")
            + VIEWS.get(view, view) + " Marks show each object's name at its estimated position (marks for objects "
            "not detected now are omitted).")
    content: list[dict[str, Any]] = [_text(head)]
    if image_before is not None:
        content.append(_image(image_before))
    content.append(_image(image_now))
    tail = (f"Marks now: {marks_now}\n"
            f"What just happened: {event_text}\n"
            f"For each statement, answer for NOW ({'image 2' if image_before else 'the image'}): 1 true, 0 false, ? cannot tell.\n"
            f"{ON_RULE}\n"
            "Statements:\n" + "\n".join(statements) + "\n" + OUTPUT_RULES_TURN)
    content.append(_text(tail))
    return [{"role": "system", "content": OBSERVE_SYSTEM}, {"role": "user", "content": content}]


# --- identity check of a first binding (bggen-fix5 B) --------------------------------------------------------------

IDENTITY_SYSTEM = "You check one object detection in a robot's camera image. Answer with one character."


# bggen-fix6 B: one line telling a category from the look-alike the detector confuses it with (B1K kitchen: the built-in
# oven was detected as the microwave; batch4 010 / 020 the real microwave got its one 'no' too). B1K categories only.
IDENTITY_CUES = {
    "microwave": ("A microwave is a small box-shaped appliance (about half a metre wide) with one side-hinged door with a "
                  "window and a narrow button / keypad panel beside it; a built-in oven is larger, with a door that opens "
                  "downward and knobs / dials above it."),
    "fridge": ("A fridge is a tall cabinet-like appliance (taller than the counter) with one or two large vertical doors; "
               "a kitchen cabinet or an oven is not one."),
}


def identity_prompt(category_text: str, image: str, view: str, cue: str | None = None) -> list[dict[str, Any]]:
    """Is the object inside the red box (mark 1) a ``category_text``? -> one line ``Answer: 1 | 0 | ?``. ``cue``
    (fix6 B, ``IDENTITY_CUES``): one line on how the category looks next to its look-alike."""
    head = f"{VIEWS.get(view, view)} A red box with the mark 1 surrounds one detected object."
    tail = (f"Is the object in the red box (mark 1) a {category_text}? Look at the object itself, not at what is near it "
            f"(another appliance or a cabinet door that only resembles a {category_text} is not one).\n"
            + (f"{cue}\n" if cue else "")
            + "Answer 1 if it is, 0 if it is not, ? if you cannot tell. Output exactly one line: Answer: <1|0|?>")
    return [{"role": "system", "content": IDENTITY_SYSTEM},
            {"role": "user", "content": [_text(head), _image(image), _text(tail)]}]


def parse_identity(text: str) -> str:
    """The answer of ``identity_prompt``: the value after the last ``Answer:``, else the last 1 / 0 / ? token; ? if none."""
    import re

    m = re.findall(r"answer\s*[:=]\s*\**\s*([01?])", text or "", re.I)
    if m:
        return m[-1]
    m = re.findall(r"(?<![\w.])([01?])(?![\w.])", text or "")
    return m[-1] if m else "?"


def text_only(messages: Sequence[dict[str, Any]]) -> str:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(f"[{m.get('role')}] {c}")
        else:
            out.append(f"[{m.get('role')}] " + "\n".join(p.get("text", "") if p.get("type") == "text" else "<image>"
                                                         for p in c or []))
    return "\n".join(out)
