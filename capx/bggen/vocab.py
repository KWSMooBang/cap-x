"""Relation vocabulary, statement text and detection categories of the BG generator.

Edge keys are ``(pred, subject, target)``:

* ``("on", X, Y)`` / ``("in", X, Y)``: X rests flat on Y / is inside container Y (Y may be ``table``)
* ``("inhand", X, arm)`` / ``("handempty", arm, None)``: robot state (arm ``None`` for a one-arm robot)
* ``("lifted", X, None)``, ``("open", X, None)``, ``("toggled_on", X, None)``, ``("cooked", X, None)``
* ``("at", X, region)``: X is in a named table region (LIBERO-Mem ``center``)

Statement text (what models read and write) is ``(on bowl_A plate_B)``, ``(inhand bowl_A)`` /
``(inhand_left hotdog_A)``, ``(handempty)`` / ``(handempty_left)``, ``(open fridge_A)``.

Ver1 serialization (``serialize.py``) renames with a vocabulary map: LIBERO keeps these names; B1K
uses ``ontop`` / ``inside`` (Ver1 B1K BG-CoT names). ``parse_statement`` accepts both.

Detection categories reuse bg_est2's ``VOCABULARY_V2`` (prompt candidates measured on LIBERO-Mem)
and add B1K task-45 categories (prompt candidates NOT measured yet: first real run decides).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from capx.belief.perception import VOCABULARY_V2, Category, categories_from_instruction

EdgeKey = tuple[str, str, "str | None"]
TABLE = "table"
ARMS_ONE = (None,)

BINARY = ("on", "in", "at")
UNARY = ("lifted", "open", "toggled_on", "cooked")
ROBOT = ("inhand", "handempty")
ALL_PREDS = BINARY + UNARY + ROBOT
LOCATION = ("on", "in", "inhand")  # one per object at a time
STATE_PREDS = ("open", "toggled_on", "cooked")  # object states (not locations)

VER1_LIBERO: dict[str, str] = {}
VER1_B1K: dict[str, str] = {"on": "ontop", "in": "inside"}
ALIASES_IN = {"ontop": "on", "inside": "in", "on_top": "on", "toggledon": "toggled_on", "isopen": "open"}

# B1K task 45 (cook hot dogs) categories. ``prompts``: SAM3 text candidates, best first -- a guess until
# the first real run measures them (as bg_est2 did for LIBERO-Mem). ``container``: In(x, this).
B1K_CATEGORIES: tuple[Category, ...] = (
    Category("hotdog", "hot dog", ("hot dog", "hotdog"), (0.15, 0.04, 0.04),
             prompts=("hot dog", "sausage", "hot dog bun"), query="hot dog"),
    Category("fridge", "refrigerator", ("refrigerator", "fridge"), (0.8, 0.7, 1.8), container=True,
             prompts=("refrigerator", "fridge", "refrigerator door"), query="refrigerator"),
    Category("microwave", "microwave", ("microwave",), (0.5, 0.35, 0.3), container=True,
             prompts=("microwave", "microwave oven"), query="microwave"),
    Category("countertop", "countertop", ("countertop", "counter"), (1.0, 0.6, 0.05),
             prompts=("countertop", "kitchen counter"), query="countertop"),
)
VOCABULARY: tuple[Category, ...] = tuple(VOCABULARY_V2) + B1K_CATEGORIES
SUPPORT_KEYS = ("plate", "basket", "tray", "countertop", "fridge", "microwave")
CONTAINER_KEYS = ("basket", "tray", "fridge", "microwave")
OPENABLE_KEYS = ("fridge", "microwave")
TOGGLEABLE_KEYS = ("microwave",)
COOKABLE_KEYS = ("hotdog",)


def category(key: str, vocabulary: Sequence[Category] = VOCABULARY) -> Category | None:
    return next((c for c in vocabulary if c.key == key), None)


def categories_for(instruction: str, extra: Sequence[str] = (), max_categories: int = 6,
                   vocabulary: Sequence[Category] = VOCABULARY) -> list[Category]:
    """Categories named in the instruction (first mention first) + ``extra`` keys (config, e.g. the B1K
    countertop the instruction does not name). Each starts on its first prompt candidate."""
    cats = categories_from_instruction(instruction, max_categories, vocabulary)
    for k in extra:
        c = category(k, vocabulary)
        if c is not None and all(x.key != k for x in cats):
            cats.append(c)
    return [replace(c, query=c.query or c.candidates[0]) for c in cats]


def is_support(key: str) -> bool:
    return key in SUPPORT_KEYS


def is_container(key: str) -> bool:
    return key in CONTAINER_KEYS


# --- statements --------------------------------------------------------------------------------

def statement_text(key: EdgeKey, rename: dict[str, str] | None = None) -> str:
    pred, s, t = key
    name = (rename or {}).get(pred, pred)
    if pred == "handempty":
        return "(handempty)" if s in (None, "", "gripper") else f"(handempty_{s})"
    if pred == "inhand":
        return f"(inhand {s})" if t in (None, "", "gripper") else f"(inhand_{t} {s})"
    if t is None:
        return f"({name} {s})"
    return f"({name} {s} {t})"


_STMT = re.compile(r"\(\s*([A-Za-z_]+)((?:[\s,]+[\w\-]+)*)\s*\)")


def parse_statement(text: str, names: Sequence[str] = (), arms: Sequence[str | None] = ARMS_ONE) -> EdgeKey | None:
    """``"(on bowl_a plate_B)"`` -> ``("on", "bowl_A", "plate_B")``; None if malformed / unknown name.
    Names are matched without case; ``table`` is always known."""
    m = _STMT.search(text or "")
    if m is None:
        return None
    return statement_key(m.group(1), [a for a in re.split(r"[\s,]+", m.group(2).strip()) if a], names, arms)


def statement_key(pred: str, args: Sequence[str], names: Sequence[str] = (),
                  arms: Sequence[str | None] = ARMS_ONE) -> EdgeKey | None:
    canon = {n.lower(): n for n in names}
    canon[TABLE] = TABLE
    pred = pred.lower()
    arm: str | None = None
    for base in ("inhand", "handempty"):
        if pred.startswith(base + "_"):
            arm = pred[len(base) + 1:]
            pred = base
    pred = ALIASES_IN.get(pred, pred)
    if pred not in ALL_PREDS:
        return None
    two_arm = any(a is not None for a in arms)
    if pred in ROBOT:
        if two_arm and arm not in arms:
            return None
        if not two_arm:
            arm = None
    a = [canon.get(x.lower()) for x in args]
    if names and any(x is None for x in a):
        return None
    a = [x if x is not None else y for x, y in zip(a, args, strict=True)]
    if pred == "handempty":
        return ("handempty", arm, None) if not a else None
    if pred == "inhand":
        return ("inhand", a[0], arm) if len(a) == 1 else None
    if pred in UNARY:
        return (pred, a[0], None) if len(a) == 1 else None
    if len(a) != 2 or a[0] == a[1] or a[0] == TABLE:
        return None
    return (pred, a[0], a[1])


def parse_values(text: str, asked: Sequence[EdgeKey], names: Sequence[str],
                 arms: Sequence[str | None] = ARMS_ONE) -> tuple[dict[EdgeKey, str], str | None, list[str]]:
    """Read an ``Observe:`` reply (bg_est2 M5 format): ``(stmt) 1|0|?`` items and a ``Notes:`` line.
    Only asked keys are kept (first answer wins); returns (values, notes, unreadable items)."""
    words = {"1": "1", "0": "0", "?": "?", "true": "1", "false": "0", "yes": "1", "no": "0", "unknown": "?",
             "unclear": "?"}
    text = text or ""
    lines = [ln.strip().strip("*`> ").strip() for ln in text.splitlines()]
    notes = None
    for ln in lines:
        m = re.match(r"(?i)\**notes?\**\s*:\s*(.*)", ln)
        if m and m.group(1).strip():
            notes = m.group(1).strip()
            break
    obs = [ln for ln in lines if re.match(r"(?i)\**observe\**\s*:", ln)]
    body = " | ".join(re.sub(r"(?i)^\**observe\**\s*:", "", ln) for ln in obs) if obs else text
    if notes:
        body = body.replace(notes, " ")
    asked_set = set(asked)
    out: dict[EdgeKey, str] = {}
    bad: list[str] = []
    for m in re.finditer(r"(\([^()]*\))\s*[:=]?\s*(true|false|yes|no|unknown|unclear|[01?])", body, re.I):
        key = parse_statement(m.group(1), names, arms)
        val = words.get(m.group(2).lower())
        if key is None or key not in asked_set or val is None:
            bad.append(m.group(1))
            continue
        out.setdefault(key, val)
    return out, notes, bad
