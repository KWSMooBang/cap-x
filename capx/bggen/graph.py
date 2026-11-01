"""Belief graph state of the BG generator: relation edges with P(true) and a source, goal progress.

Sources (design 그래프 스키마): ``obs`` (VDM Observe), ``robot`` (proprio), ``exp`` (declared
expectation), ``mem`` (not re-observed since), ``conflict`` (geometry vetoed an Observe 1).
Only the verifier (``verifier.py``) writes edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .vocab import EdgeKey, statement_text

DECIDED_HI = 0.7
DECIDED_LO = 0.3
CONFIRMING = ("obs", "robot", "mem")  # sources that count as observed for goal progress


@dataclass
class Edge:
    p: float
    source: str
    t: int  # frame of the last change
    obs_t: int | None = None  # frame of the last accepted observation


def decided(p: float | None) -> bool | None:
    """True / False when decided (P >= 0.7 / <= 0.3), None otherwise."""
    if p is None:
        return None
    if p >= DECIDED_HI:
        return True
    if p <= DECIDED_LO:
        return False
    return None


@dataclass
class BeliefGraph:
    edges: dict[EdgeKey, Edge] = field(default_factory=dict)
    remember: list[str] = field(default_factory=list)  # free "things to remember" from the init LLM
    notes: dict[str, str] = field(default_factory=dict)  # remembered facts, e.g. orig(bowl_A) -> plate_B
    events: list[str] = field(default_factory=list)  # robot -grasp-> bowl_A @118 (from plate_B)

    def get(self, key: EdgeKey) -> Edge | None:
        return self.edges.get(key)

    def p(self, key: EdgeKey) -> float | None:
        e = self.edges.get(key)
        return None if e is None else e.p

    def set(self, key: EdgeKey, p: float, source: str, t: int, observed: bool = False) -> None:
        e = self.edges.get(key)
        if e is None:
            self.edges[key] = Edge(round(p, 3), source, t, t if observed else None)
            return
        if round(p, 3) != e.p or source != e.source:
            e.t = t
        e.p, e.source = round(p, 3), source
        if observed:
            e.obs_t = t

    def age(self, observed: set[EdgeKey], recomputed: set[EdgeKey]) -> None:
        """After an event: observed-sourced edges not re-observed become ``mem`` (P unchanged)."""
        for k, e in self.edges.items():
            if k in observed or k in recomputed:
                continue
            if e.source == "obs":
                e.source = "mem"

    def location(self, alias: str) -> list[tuple[EdgeKey, Edge]]:
        return [(k, e) for k, e in self.edges.items() if k[1] == alias and k[0] in ("on", "in", "inhand") and e.p >= 0.5]

    def snapshot(self, rename: dict[str, str] | None = None) -> dict[str, list[Any]]:
        return {statement_text(k, rename): [e.p, e.source] for k, e in sorted(self.edges.items(), key=_order)}

    def summary(self, keys: Sequence[EdgeKey] | None = None, max_items: int = 30) -> str:
        items = [(k, e) for k, e in sorted(self.edges.items(), key=_order) if keys is None or k in keys]
        items = [(k, e) for k, e in items if e.p >= 0.5 or k[0] in ("handempty",)] + \
                [(k, e) for k, e in items if e.p < 0.5 and k[0] not in ("handempty",) and e.source in ("conflict",)]
        return " | ".join(f"{statement_text(k)} {e.p:.2f} {e.source}" for k, e in items[:max_items]) or "(empty)"


def _order(item: tuple[EdgeKey, Edge]) -> tuple:
    k = item[0]
    return (k[0] not in ("on", "in", "at"), k[0], str(k[1]), str(k[2]))


# --- goal -----------------------------------------------------------------------------------------

Step = tuple[EdgeKey, ...]


@dataclass
class GoalTracker:
    """Goal = branches of ordered steps; each step lists the relations that BECOME true at it. A step
    counts when all its relations are decided true with an observing source (obs / robot / mem after
    an observation), in order, causally (never from later frames)."""

    branches: list[list[Step]] = field(default_factory=list)
    done: list[int] = field(default_factory=list)  # per branch
    done_t: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.done:
            self.done = [0] * len(self.branches)
            self.done_t = [[] for _ in self.branches]

    @property
    def known(self) -> bool:
        return bool(self.branches)

    def update(self, g: BeliefGraph, t: int) -> list[tuple[int, int]]:
        """Advance every branch; returns the (branch, step) pairs confirmed now."""
        out = []
        for b, steps in enumerate(self.branches):
            while self.done[b] < len(steps) and all(
                    (e := g.get(k)) is not None and e.p >= DECIDED_HI and e.source in CONFIRMING for k in steps[self.done[b]]):
                out.append((b, self.done[b]))
                self.done[b] += 1
                self.done_t[b].append(t)
        return out

    def best(self) -> int | None:
        if not self.branches:
            return None
        return max(range(len(self.branches)), key=lambda b: (self.done[b], -b))

    def pending(self, g: BeliefGraph) -> list[EdgeKey]:
        """Relations of the next step(s) that are neither confirmed nor ruled out (e.g. vetoed 0.5)."""
        out: list[EdgeKey] = []
        for b, steps in enumerate(self.branches):
            if self.done[b] < len(steps):
                for k in steps[self.done[b]]:
                    e = g.get(k)
                    if e is not None and 0.3 < e.p < 0.7 and k not in out:
                        out.append(k)
        return out

    def next_keys(self) -> list[EdgeKey]:
        out: list[EdgeKey] = []
        top = max(self.done) if self.done else 0
        for b, steps in enumerate(self.branches):
            if self.done[b] == top and self.done[b] < len(steps):
                out += [k for k in steps[self.done[b]] if k not in out]
        return out

    def all_keys(self) -> list[EdgeKey]:
        return list(dict.fromkeys(k for b in self.branches for s in b for k in s))

    def text(self, g: BeliefGraph, rename: dict[str, str] | None = None) -> str:
        if not self.branches:
            return "progress unknown"
        b = self.best()
        assert b is not None
        n, k = len(self.branches[b]), self.done[b]
        tied = [i for i in range(len(self.branches)) if self.done[i] == k and k < len(self.branches[i])]
        parts = [f"{k}/{n}"]
        if k >= n:
            parts.append("done")
            return " · ".join(parts)
        nxt = [" and ".join(statement_text(x, rename) for x in self.branches[i][k]) for i in tied]
        nxt = list(dict.fromkeys(nxt))
        parts.append("next " + (nxt[0] if len(nxt) == 1 else " or ".join(nxt)))
        if any(0.3 < (g.p(x) or 0.0) < 0.7 for x in self.branches[b][k]):
            parts.append(f"step {k + 1} pending")
        return " · ".join(parts)

    def satisfied_final(self, g: BeliefGraph) -> bool:
        """Every relation of the last step of the best branch holds now (goal consistency)."""
        b = self.best()
        if b is None or not self.branches[b]:
            return False
        return all((g.p(k) or 0.0) >= 0.5 for k in self.branches[b][-1])
