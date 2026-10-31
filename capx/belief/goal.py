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

_UNKNOWN = re.compile(r"unknown object '([^']*)'")


def unknown_objects(result: GoalResult) -> list[str]:
    """Names a failed goal parse rejected as not in the detected objects (empty: it failed otherwise)."""
    if result.branches is not None:
        return []
    out: list[str] = []
    for e in result.errors:
        for m in _UNKNOWN.finditer(e):
            if m.group(1) not in out:
                out.append(m.group(1))
    return out


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


# --- v2 (bg_est2): M3 prompt, regions, per-step "newly true" normalization ---------------------------

V2_PREDICATES = {"on": 3, "in": 3, "at": 3}
REGION_WORDS = ("center",)  # table regions an instruction may name; at(o, r)


def regions_in(instruction: str, regions: Sequence[str] = REGION_WORDS) -> list[str]:
    text = instruction.lower()
    return [r for r in regions if re.search(rf"\b{re.escape(r)}\b", text)]


def _pred_v2(raw: Any, names: set[str], regions: set[str]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise GoalParseError(f"predicate {raw!r} is not a list")
    p = [str(x).strip() for x in raw]
    p[0] = p[0].lower().replace("_", "")
    lower = {n.lower(): n for n in names}
    if p[0] not in V2_PREDICATES or len(p) != V2_PREDICATES[p[0]]:
        raise GoalParseError(f"predicate {raw!r}: use on/in with 2 objects")
    subj = lower.get(p[1].lower())
    if subj is None:
        raise GoalParseError(f"unknown object {p[1]!r} in {raw!r}")
    tgt_raw = p[2].lower()
    tgt = lower.get(tgt_raw)
    if tgt is None:
        if tgt_raw == TABLE:
            if p[0] != "on":
                raise GoalParseError(f"{raw!r}: only 'on' the table")
            return ("on", subj, TABLE)
        region = next((r for r in regions if r in tgt_raw), None)  # "center", "the center of the table"
        if region is None:
            raise GoalParseError(f"unknown object {p[2]!r} in {raw!r}")
        return ("at", subj, region)
    if p[0] == "at":
        raise GoalParseError(f"{raw!r}: 'at' takes a region")
    if subj == tgt:
        raise GoalParseError(f"predicate {raw!r} relates an object to itself")
    return (p[0], subj, tgt)


def _steps_v2(raw_steps: Any, names: set[str], regions: set[str]) -> list[Step]:
    if not isinstance(raw_steps, list) or not raw_steps:
        raise GoalParseError("steps must be a non-empty list")
    out: list[Step] = []
    for s in raw_steps:
        if isinstance(s, dict):
            preds, repeat = s.get("preds"), s.get("repeat", 1)
        else:
            preds, repeat = s, 1
        if not isinstance(preds, list) or not preds:
            raise GoalParseError(f"step {s!r} has no predicates")
        if isinstance(preds[0], str):
            preds = [preds]
        try:
            repeat = int(repeat if repeat is not None else 1)
        except (TypeError, ValueError) as e:
            raise GoalParseError(f"repeat {repeat!r} is not a number") from e
        if not 1 <= repeat <= MAX_REPEAT:
            raise GoalParseError(f"repeat {repeat} out of range 1..{MAX_REPEAT}")
        out += [canonical_step([_pred_v2(p, names, regions) for p in preds])] * repeat
    return out


def parse_goal_json_v2(text: str, names: Sequence[str], regions: Sequence[str] = REGION_WORDS) -> list[list[Step]]:
    """M3 reply -> goal branches (not yet normalized); raises GoalParseError. Accepts on / in over the
    aliases and "table", and a region word (e.g. center) as target -> ("at", X, region)."""
    data = _extract_json(text)
    goal = data.get("goal", data) if isinstance(data, dict) else None
    if not isinstance(goal, dict):
        raise GoalParseError("missing 'goal' object")
    known, regs = set(names), {r.lower() for r in regions}
    if str(goal.get("type", "sequence")).lower() == "or" or "branches" in goal:
        branches = goal.get("branches")
        if not isinstance(branches, list) or not branches:
            raise GoalParseError("'or' goal needs a non-empty 'branches' list")
        return [_steps_v2(b.get("steps") if isinstance(b, dict) else b, known, regs) for b in branches]
    return [_steps_v2(goal.get("steps"), known, regs)]


def _apply(state: dict[str, frozenset], pred: tuple[str, ...]) -> None:
    if pred[0] in ("on", "in", "at"):
        state[pred[1]] = frozenset({(pred[0], pred[2])})


def _holds(state: dict[str, frozenset], pred: tuple[str, ...]) -> bool:
    return (pred[0], pred[2] if len(pred) > 2 else None) in state.get(pred[1], frozenset())


def normalize_steps(branches: Sequence[Sequence[Step]], init: dict[str, frozenset]) -> list[list[Step]]:
    """Each step keeps only the predicates that BECOME true at it.

    Walking each branch from ``init`` (alias -> {(pred, target)}), a predicate that already holds before
    its step is dropped when the step has another, new predicate (a cumulative "state so far" step). A
    step none of whose predicates is new is kept whole: it is a put-back / repeat ("place it back")."""
    out = []
    for b in branches:
        state = dict(init)
        steps: list[Step] = []
        for s in b:
            new = [p for p in s if not _holds(state, p)]
            step = canonical_step(new if new else list(s))
            for p in step:
                _apply(state, p)
            steps.append(step)
        out.append(steps)
    return out


def dedupe_branches(branches: Sequence[Sequence[Step]]) -> list[list[Step]]:
    out: list[list[Step]] = []
    for b in branches:
        if list(b) not in out:
            out.append(list(b))
    return out


def parse_goal_llm_v2(query_fn: QueryFn, instruction: str, scene_lines: Sequence[str], names: Sequence[str],
                      init: dict[str, frozenset], regions: Sequence[str] = REGION_WORDS,
                      retries: int = 1) -> GoalResult:
    """M3; on a bad reply ask once more with the error. Branches are normalized to newly-true predicates."""
    from .prompts_v2 import goal_prompt_v2

    result = GoalResult(None, "llm")
    prompt = goal_prompt_v2(instruction, scene_lines)
    for _ in range(retries + 1):
        result.attempts += 1
        try:
            text, usage = _content(query_fn(prompt))
        except Exception as exc:
            result.errors.append(f"query failed: {exc}")
            continue
        result.replies.append(text)
        result.usage.append(usage)
        try:
            raw = parse_goal_json_v2(text, names, regions)
            result.branches = dedupe_branches(normalize_steps(raw, init))
            return result
        except GoalParseError as exc:
            result.errors.append(str(exc))
            prompt = prompt + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"That reply could not be used: {exc}. Answer again with JSON only, in the format above.")}]},
            ]
    return result


def horizontal_word(y: float, ys: Sequence[float]) -> str:
    """Image left/right from base y: the robot's right (-y) is the image left. Bins over the scene's spread."""
    lo, hi = min(ys), max(ys)
    if hi - lo < 0.05:
        return "center"
    f = (y - lo) / (hi - lo)
    return ("left side", "left of center", "center", "right of center", "right side")[min(int(f * 5), 4)]


def depth_word(x: float, xs: Sequence[float]) -> str:
    """Distance from the camera from base x (larger x = nearer the camera, lower in the image)."""
    lo, hi = min(xs), max(xs)
    if hi - lo < 0.05:
        return "middle"
    if hi - x < 0.02:
        return "nearest to the camera"
    f = (x - lo) / (hi - lo)
    return ("near the robot", "middle", "near the camera")[min(int(f * 3), 2)]


def scene_lines_v2(objects: Sequence[tuple[str, str, Sequence[float], str]], regions: Sequence[str] = ()) -> list[str]:
    """M3 scene: ``(alias, category prompt, xyz, relation text)`` -> ``- bowl_A (black bowl): on plate_B;
    center of the image, nearest to the camera``. Relation texts (on X / empty / holds X) come from the
    initial Observe."""
    if not objects:
        return []
    xs = [float(o[2][0]) for o in objects]
    ys = [float(o[2][1]) for o in objects]
    out = []
    for alias, prompt, pos, rel in objects:
        h = horizontal_word(float(pos[1]), ys)
        h = "center of the image" if h == "center" else h
        out.append(f"- {alias} ({prompt}): {rel}; {h}, {depth_word(float(pos[0]), xs)}")
    for r in regions:
        out.append(f'- {r}: a region of the table (a place named in the instruction: use "{r}" as the target)')
    return out
