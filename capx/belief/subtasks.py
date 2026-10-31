"""Subtask layer of the estimated Belief Graph v2 (``bg_est2``).

Goal steps are expanded into operators (design: 연산자 표):

=================  ========================================  ==========================================
operator           pre                                       eff
=================  ========================================  ==========================================
pick(o, s)         handempty, on(o, s) or in(o, s)           inhand(o), lifted(o), not on(o, s), not handempty
place(o, t)        inhand(o), above(gripper, t)              on(o, t), handempty, not inhand(o)
put_in(o, c)       inhand(o), above(gripper, c)              in(o, c), handempty
place_region(o,r)  inhand(o)                                 at(o, r), handempty
=================  ========================================  ==========================================

Expansion: on(o, t) -> pick(o, current support) + place(o, t); in(o, c) -> pick + put_in; at(o, r)
-> pick + place_region; a repeated step -> pick(o, t) + place(o, t) every repetition (the object is
already on t, so its current support is t).

Status: todo / active / done / failed / pending. Completion (judged by the estimator): pick = the
robot holds o and lifted it; place / put_in / place_region = open_gripper ran and Observe says the
relation is 1 (0: failed, ?: pending).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .observe import EdgeKey

STATUSES = ("todo", "active", "done", "failed", "pending")
PLACE_OPS = ("place", "put_in", "place_region")
TABLE = "table"


@dataclass
class Subtask:
    op: str
    args: tuple[str, ...]
    status: str = "todo"
    turn: int | None = None
    by: str | None = None  # robot | vdm | geo | robot+vdm ...
    declared: bool = False

    @property
    def text(self) -> str:
        return f"{self.op}({', '.join(self.args)})"

    @property
    def obj(self) -> str | None:
        return self.args[0] if self.args else None

    @property
    def target(self) -> str | None:
        return self.args[1] if len(self.args) > 1 else None

    def same(self, other: Subtask) -> bool:
        """Same operator on the same object and destination (pick: any source)."""
        if self.op != other.op or self.obj != other.obj:
            return False
        return self.op == "pick" or self.target == other.target

    def effect_key(self) -> EdgeKey | None:
        """The relation that decides a place-type subtask."""
        if self.op == "place":
            return ("on", self.args[0], self.args[1])
        if self.op == "put_in":
            return ("in", self.args[0], self.args[1])
        if self.op == "place_region":
            return ("at", self.args[0], self.args[1])
        if self.op == "pick":
            return ("inhand", self.args[0], None)
        return None

    def preconditions(self) -> list[tuple[EdgeKey, bool]]:
        o = self.obj
        if self.op == "pick":
            pre: list[tuple[EdgeKey, bool]] = [(("handempty", "gripper", None), True)]
            if self.target is not None and self.target != "?":
                pre.append(((("in" if self.target_is_container else "on"), o, self.target), True))  # type: ignore[arg-type]
            return pre
        if self.op in PLACE_OPS:
            return [(("inhand", o, None), True)]
        return []

    target_is_container: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": list(self.args), "status": self.status, "turn": self.turn, "by": self.by,
                "declared": self.declared}

    def render(self) -> str:
        word = {"done": " done", "failed": " FAILED", "pending": " pending (not confirmed by the camera)",
                "active": " (in progress)"}.get(self.status, "")
        return self.text + word


Location = dict[str, frozenset]  # alias -> {(pred, target)} (estimator snapshot)


def current_support(loc: Location, obj: str) -> str:
    rels = loc.get(obj, frozenset())
    order = {"in": 0, "on": 1}  # a region (at) is not a support
    for pred, target in sorted(rels, key=lambda r: (order.get(r[0], 9), r[1] or "")):
        if pred in ("on", "in") and target:
            return target
    return "?"


def expand_step(step: Sequence[tuple[str, ...]], loc: Location, containers: set[str] = frozenset(),
                held: str | None = None) -> list[Subtask]:
    """Operators that achieve ``step`` from location ``loc``. ``held``: the object in the gripper (its
    pick is already done)."""
    out: list[Subtask] = []
    loc = dict(loc)
    for p in step:
        if p[0] not in ("on", "in", "at") or len(p) < 3:
            continue
        o, t = p[1], p[2]
        if held != o:
            src = current_support(loc, o)
            st = Subtask("pick", (o, src))
            st.target_is_container = src in containers
            out.append(st)
        op = {"on": "place", "in": "put_in", "at": "place_region"}[p[0]]
        out.append(Subtask(op, (o, t)))
        loc[o] = frozenset({(p[0], t)})
        held = None
    return out


def expand_branch(steps: Sequence[Sequence[tuple[str, ...]]], loc: Location, containers: set[str] = frozenset(),
                  held: str | None = None) -> list[list[Subtask]]:
    """Per step, its operators, tracking where each object goes."""
    out = []
    loc = dict(loc)
    for s in steps:
        ops = expand_step(s, loc, containers, held)
        held = None
        for p in s:
            if p[0] in ("on", "in", "at") and len(p) >= 3:
                loc[p[1]] = frozenset({(p[0], p[2])})
        out.append(ops)
    return out


@dataclass
class SubtaskLog:
    """Judged subtasks of the trial and those of the current goal step (reset when a step is counted)."""

    all: list[Subtask] = field(default_factory=list)
    current: list[Subtask] = field(default_factory=list)  # judged since the last counted step

    def add(self, st: Subtask) -> Subtask:
        """Record a judgement; a pending entry of the same subtask is updated in place."""
        for old in reversed(self.current):
            if old.same(st) and old.status == "pending":
                old.status, old.turn, old.by = st.status, st.turn, st.by
                old.declared = old.declared or st.declared
                return old
        self.all.append(st)
        self.current.append(st)
        return st

    def pending(self) -> list[Subtask]:
        return [s for s in self.current if s.status == "pending"]

    def step_counted(self) -> None:
        self.current = []
