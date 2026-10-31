"""Goal parsing for the estimated Belief Graph (``bg_est``): task language -> ordered goal steps.

Once per trial, the task language and the initial object list (aliases, positions, relations from
the first perception) go to a text-only LLM, which returns the goal as JSON in our predicate form
(``on`` / ``in`` / ``inhand`` over object aliases, ``table`` as a support). The result is the same
``list[list[Step]]`` (one list per alternative order) that ``progress.Progress`` uses for the oracle.

JSON accepted from the model::

    {"goal": {"type": "sequence", "steps": [{"preds": [["on", "bowl_A", "plate_C"]], "repeat": 1}, ...]}}
    {"goal": {"type": "or", "branches": [{"steps": [...]}, {"steps": [...]}]}}

``repeat: N`` ("... N times") expands to the same step N times. A reply that does not parse or
validate is asked once more with the error; if that fails too the goal is None and progress is
reported as unknown.

``goal_source: bddl`` (diagnostics only) builds the steps from the BDDL goal instead, with simulator
names mapped to aliases by the caller; it is logged as such.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .progress import Step, canonical_step, goal_branches

PREDICATES = {"on": 3, "in": 3, "inhand": 2}
TABLE = "table"
MAX_REPEAT = 20

GOAL_SYSTEM = (
    "You turn a robot task instruction into a goal specification over the objects in the scene. "
    "Answer with JSON only, no code and no explanation."
)

GOAL_INSTRUCTIONS = """Write the goal as an ordered list of steps. Each step is a set of predicates that must all hold at that point:
- ["on", X, Y]: object X rests on object Y (Y may be "table")
- ["in", X, Y]: object X is inside container Y
- ["inhand", X]: object X is held by the robot gripper
Use only the object names listed above (and "table").
Rules:
- One step per intermediate arrangement the instruction asks for, in order. Do not add steps for grasping or lifting.
- "N times" means the same step N times: use "repeat": N. Putting an object back where it was counts as a step only after it was picked up, so "place it back on the plate 3 times" is the step [["on", obj, plate]] with "repeat": 3.
- If the instruction allows several orders or choices (e.g. which object goes first, which of two containers), give each complete alternative as a branch.
Format (JSON only):
{"goal": {"type": "sequence", "steps": [{"preds": [["on", "apple_A", "box_B"]], "repeat": 1}]}}
or
{"goal": {"type": "or", "branches": [{"steps": [...]}, {"steps": [...]}]}}"""


class GoalParseError(ValueError):
    pass


def _extract_json(text: str) -> Any:
    """First JSON object in ``text`` (code fences and surrounding prose are ignored)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise GoalParseError("no JSON object in the reply")


def _pred(raw: Any, names: set[str]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise GoalParseError(f"predicate {raw!r} is not a list")
    p = [str(x).strip() for x in raw]
    p[0] = p[0].lower().replace("_", "")
    if p[0] not in PREDICATES or len(p) != PREDICATES[p[0]]:
        raise GoalParseError(f"predicate {raw!r}: use on/in with 2 objects or inhand with 1")
    for i, arg in enumerate(p[1:], 1):
        if arg not in names and not (arg == TABLE and p[0] == "on" and i == 2):
            raise GoalParseError(f"unknown object {arg!r} in {raw!r}")
    if p[0] == "on" and p[1] == p[2]:
        raise GoalParseError(f"predicate {raw!r} relates an object to itself")
    return tuple(p)


def _steps(raw_steps: Any, names: set[str]) -> list[Step]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise GoalParseError("steps must be a non-empty list")
    out: list[Step] = []
    for s in raw_steps:
        if isinstance(s, dict):
            preds, repeat = s.get("preds"), s.get("repeat", 1)
        else:  # a bare list of predicates
            preds, repeat = s, 1
        if not isinstance(preds, list) or not preds:
            raise GoalParseError(f"step {s!r} has no predicates")
        if preds and isinstance(preds[0], str):  # a single predicate without the outer list
            preds = [preds]
        try:
            repeat = int(repeat)
        except (TypeError, ValueError) as e:
            raise GoalParseError(f"repeat {repeat!r} is not a number") from e
        if not 1 <= repeat <= MAX_REPEAT:
            raise GoalParseError(f"repeat {repeat} out of range 1..{MAX_REPEAT}")
        out += [canonical_step([_pred(p, names) for p in preds])] * repeat
    return out


def parse_goal_json(text: str, names: Sequence[str]) -> list[list[Step]]:
    """Model reply -> goal branches; raises GoalParseError."""
    data = _extract_json(text)
    goal = data.get("goal", data) if isinstance(data, dict) else None
    if not isinstance(goal, dict):
        raise GoalParseError("missing 'goal' object")
    known = set(names)
    kind = str(goal.get("type", "sequence")).lower()
    if kind == "or" or "branches" in goal:
        branches = goal.get("branches")
        if not isinstance(branches, list) or not branches:
            raise GoalParseError("'or' goal needs a non-empty 'branches' list")
        return [_steps(b.get("steps") if isinstance(b, dict) else b, known) for b in branches]
    return [_steps(goal.get("steps"), known)]


def goal_preds(branches: Sequence[Sequence[Step]]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for b in branches:
        for s in b:
            for p in s:
                if p not in out:
                    out.append(p)
    return out


@dataclass
class GoalResult:
    branches: list[list[Step]] | None  # None: could not be parsed -> progress unknown
    source: str  # llm | bddl
    attempts: int = 0
    replies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    usage: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "branches": None if self.branches is None else [[list(map(list, s)) for s in b] for b in self.branches],
            "attempts": self.attempts, "errors": self.errors, "replies": self.replies,
        }


QueryFn = Callable[[list[dict[str, Any]]], Any]  # prompt -> str or {"content", "usage", ...}


def _content(out: Any) -> tuple[str, Any]:
    if isinstance(out, dict):
        return out.get("content") or "", out.get("usage")
    return str(out or ""), None


def goal_prompt(instruction: str, objects_text: str) -> list[dict[str, Any]]:
    user = (
        f"Task instruction: {instruction}\n\n"
        f"Objects detected in the scene (estimated; xyz in the robot base frame, m):\n{objects_text}\n\n"
        f"{GOAL_INSTRUCTIONS}"
    )
    return [
        {"role": "system", "content": GOAL_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def parse_goal_llm(query_fn: QueryFn, instruction: str, objects_text: str, names: Sequence[str],
                   retries: int = 1) -> GoalResult:
    """Ask the model; on a bad reply ask once more with the error (``retries``)."""
    result = GoalResult(None, "llm")
    prompt = goal_prompt(instruction, objects_text)
    for _ in range(retries + 1):
        result.attempts += 1
        try:
            text, usage = _content(query_fn(prompt))
        except Exception as exc:  # server error: counts as a failed attempt
            result.errors.append(f"query failed: {exc}")
            continue
        result.replies.append(text)
        result.usage.append(usage)
        try:
            result.branches = parse_goal_json(text, names)
            return result
        except GoalParseError as exc:
            result.errors.append(str(exc))
            prompt = prompt + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"That reply could not be used: {exc}. Answer again with JSON only, in the format above.")}]},
            ]
    return result


def goal_from_bddl(goal_state: Sequence[Any], sim_to_alias: dict[str, str]) -> GoalResult:
    """Diagnostics: the BDDL goal with simulator names mapped to aliases. ``X_contain_region`` maps
    to X's alias; a name without an alias (e.g. a table region) is kept and never holds."""

    def name(n: str) -> str:
        base = n[: -len("_contain_region")] if n.endswith("_contain_region") else n
        return sim_to_alias.get(base, n)

    branches = [
        [canonical_step([(p[0],) + tuple(name(a) for a in p[1:]) for p in step]) for step in b]
        for b in goal_branches(goal_state)
    ]
    return GoalResult(branches, "bddl", attempts=0)
