"""Subtask / Effect declarations of the code agent (``bg_est2``, prompt rule M4).

A code block starts with two comment lines::

    # Subtask: place(bowl_B, plate_B)
    # Effect: (on bowl_B plate_B) 0>1 | (inhand bowl_B) 1>0

``parse_declaration`` reads them (the first two non-empty lines of the block; a leading code fence
is skipped). A missing or malformed Subtask makes the declaration void (``declared: false``) and
the estimator falls back to the v1 hook inference. Effect segments that cannot be read are dropped
and listed in ``errors``; the Subtask still counts as declared.

``check_records`` compares a declaration with the turn's API-call records: when they clearly
disagree (a place declared but the gripper opened above another tracked support, or a pick
declared but another object grasped) the declaration is dropped and the reason logged. Declared
Effects are applied as expectations only when the operator's API call actually ran (pick:
close_gripper; place / put_in / place_region: open_gripper; look: never).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .observe import EdgeKey, statement_key, statement_text

OPS = ("pick", "place", "put_in", "place_region", "look")
OP_ARITY = {"pick": (1, 2), "place": (2, 2), "put_in": (2, 2), "place_region": (2, 2), "look": (0, 2)}
TRIGGER = {"pick": "close_gripper", "place": "open_gripper", "put_in": "open_gripper",
           "place_region": "open_gripper", "look": None}

_SUBTASK = re.compile(r"^#+\s*subtask\s*:\s*([A-Za-z_]+)\s*\(([^)]*)\)", re.I)
_EFFECT = re.compile(r"^#+\s*effects?\s*:\s*(.*)$", re.I)
_EFFECT_SEG = re.compile(r"\(\s*([A-Za-z_]+)((?:[\s,]+[\w\-]+)*)\s*\)\s*([01?])\s*(?:>|->|=>|→)\s*([01])")


@dataclass
class Effect:
    key: EdgeKey
    before: str  # 0 | 1 | ?
    after: bool

    @property
    def text(self) -> str:
        return f"{statement_text(self.key)} {self.before}>{int(self.after)}"


@dataclass
class Declaration:
    op: str | None = None
    args: list[str] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    declared: bool = False
    errors: list[str] = field(default_factory=list)
    raw_subtask: str | None = None
    raw_effect: str | None = None
    dropped: str | None = None  # reason the declaration was discarded after the API check
    applied: bool = False  # the operator's API call ran, so the Effects were applied

    @property
    def subtask_text(self) -> str | None:
        return f"{self.op}({', '.join(self.args)})" if self.op else None

    @property
    def effect_text(self) -> str:
        return " | ".join(e.text for e in self.effects)

    @property
    def valid(self) -> bool:
        return self.declared and self.dropped is None

    def to_dict(self) -> dict[str, Any]:
        return {"declared": self.declared, "valid": self.valid, "subtask": self.subtask_text,
                "effects": [e.text for e in self.effects], "errors": self.errors, "dropped": self.dropped,
                "applied": self.applied, "raw_subtask": self.raw_subtask, "raw_effect": self.raw_effect}


def _canon(name: str, names: dict[str, str]) -> str | None:
    return names.get(name.lower())


def parse_declaration(code: str | None, names: Sequence[str], regions: Sequence[str] = ()) -> Declaration:
    """Declaration of a code block. ``names``: object aliases; ``regions``: region words (e.g. center)."""
    d = Declaration()
    if not code:
        d.errors.append("no code")
        return d
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    head = lines[:2]
    canon = {n.lower(): n for n in names}
    canon.update({"table": "table", "gripper": "gripper"})
    region_set = {r.lower(): r for r in regions}
    for ln in head:
        m = _SUBTASK.match(ln)
        if m:
            d.raw_subtask = ln
            op = m.group(1).lower()
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            if op not in OPS:
                d.errors.append(f"unknown op {op!r}")
                continue
            lo, hi = OP_ARITY[op]
            if not lo <= len(args) <= hi:
                d.errors.append(f"{op} takes {lo}..{hi} arguments, got {len(args)}")
                continue
            out = []
            for i, a in enumerate(args):
                c = _canon(a, canon)
                if c is None and op == "place_region" and i == 1:
                    c = region_set.get(a.lower(), a.lower())
                if c is None:
                    d.errors.append(f"unknown object {a!r} in Subtask")
                    break
                out.append(c)
            else:
                d.op, d.args, d.declared = op, out, True
            continue
        m = _EFFECT.match(ln)
        if m:
            d.raw_effect = ln
            for seg in m.group(1).split("|"):
                seg = seg.strip()
                if not seg:
                    continue
                sm = _EFFECT_SEG.search(seg)
                if sm is None:
                    d.errors.append(f"effect {seg!r} not in the form (pred args) a>b")
                    continue
                args = [a for a in re.split(r"[\s,]+", sm.group(2).strip()) if a]
                cargs = []
                for a in args:
                    c = _canon(a, canon) or region_set.get(a.lower())
                    if c is None:
                        break
                    cargs.append(c)
                key = statement_key(sm.group(1), cargs) if len(cargs) == len(args) else None
                if key is None:
                    d.errors.append(f"effect {seg!r}: unknown predicate or object")
                    continue
                d.effects.append(Effect(key, sm.group(3), sm.group(4) == "1"))
    if d.raw_subtask is None:
        d.errors.append("no '# Subtask:' line in the first two lines")
    return d


@dataclass
class RecordFacts:
    """What the hooks saw this turn (filled by the estimator from its v1 hook inference)."""

    closed: bool = False  # close_gripper ran
    opened: bool = False  # open_gripper ran
    grasped: list[str] = field(default_factory=list)  # objects judged grasped at a close
    released: list[tuple[str, str | None]] = field(default_factory=list)  # (object, support under it or None=table)


def check_records(d: Declaration, facts: RecordFacts) -> Declaration:
    """Drop a declaration that the API calls clearly contradict; mark whether its Effects apply."""
    if not d.declared:
        return d
    trig = TRIGGER.get(d.op or "")
    obj = d.args[0] if d.args else None
    if d.op == "pick" and facts.grasped and obj not in facts.grasped:
        d.dropped = f"pick({obj}) declared but the gripper closed on {facts.grasped[-1]}"
    elif d.op in ("place", "put_in") and facts.released:
        target = d.args[1]
        rel = [r for r in facts.released if r[0] == obj]
        if not rel:
            d.dropped = f"{d.subtask_text} declared but the gripper released {facts.released[-1][0]}"
        elif target != "table" and rel[-1][1] is not None and rel[-1][1] != target:
            d.dropped = f"{d.subtask_text} declared but open_gripper ran above {rel[-1][1]}"
    elif d.op == "place_region" and facts.released and all(r[0] != obj for r in facts.released):
        d.dropped = f"{d.subtask_text} declared but the gripper released {facts.released[-1][0]}"
    if d.dropped is None:
        d.applied = trig is not None and (facts.closed if trig == "close_gripper" else facts.opened)
    return d
