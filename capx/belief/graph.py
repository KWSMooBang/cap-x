"""Belief Graph: object nodes, predicate edges with a belief probability, and an event log.

Confidence constants follow the BEHAVIOR Belief Graph design: prior 0.5, direct observation
0.97 / 0.03, expected (predicted from an executed action) 0.7, no time decay. An edge holds when
P >= 0.5. An edge that was not re-observed in an update keeps its P and is marked source="mem".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

P_PRIOR = 0.5
P_OBS_TRUE = 0.97
P_OBS_FALSE = 0.03
P_EXPECTED = 0.7
HOLD_THRESHOLD = 0.5

SOURCES = ("obs", "mem", "expected", "prior")

# Predicates that place the subject somewhere; their true targets form the subject's location.
LOCATION_PREDICATES = ("on", "in", "inhand")
GRIPPER = "gripper"


def edge_key(predicate: str, subject: str, target: str | None = None) -> tuple[str, str, str | None]:
    return (predicate.lower(), subject, target)


EdgeKey = tuple[str, str, "str | None"]


@dataclass
class Node:
    name: str  # sim instance name from the BDDL, e.g. akita_black_bowl_1
    alias: str | None = None  # optional human-readable name, e.g. bowl_A
    pos: tuple[float, float, float] | None = None
    last_seen_turn: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.alias})" if self.alias else self.name


@dataclass
class Edge:
    predicate: str  # lower-case BDDL predicate: on, in, inhand, ...
    subject: str
    target: str | None
    p: float
    source: str
    last_seen_turn: int | None = None

    @property
    def key(self) -> EdgeKey:
        return (self.predicate, self.subject, self.target)

    @property
    def holds(self) -> bool:
        return self.p >= HOLD_THRESHOLD


@dataclass
class Event:
    """A change in an object's location between two updates."""

    turn: int
    subject: str
    before: str
    after: str

    @property
    def text(self) -> str:
        return f"t{self.turn} {self.subject} {self.before} -> {self.after}"


def format_location(relations: Iterable[tuple[str, str | None]]) -> str:
    """Location label from the subject's true (predicate, target) pairs.

    ``on plate_3``, ``in basket_1_contain_region``, ``in gripper``; several are joined with
    `` + `` and no true relation at all is ``on nothing tracked``.
    """
    parts = []
    for pred, target in sorted(relations, key=lambda r: (r[0] != "inhand", r[0], r[1] or "")):
        if pred == "inhand":
            parts.append(f"in {GRIPPER}")
        else:
            parts.append(f"{pred} {target}" if target else pred)
    return " + ".join(parts) if parts else "on nothing tracked"


@dataclass
class BeliefGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[EdgeKey, Edge] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    turn: int = 0

    # --- nodes -------------------------------------------------------------
    def add_node(self, name: str, alias: str | None = None) -> Node:
        node = self.nodes.get(name)
        if node is None:
            node = self.nodes[name] = Node(name=name, alias=alias)
        elif alias is not None:
            node.alias = alias
        return node

    # --- edges -------------------------------------------------------------
    def set_prior(self, key: EdgeKey) -> Edge:
        """Edge believed only from task knowledge (e.g. the BDDL :init), P=0.5."""
        return self._set(key, P_PRIOR, "prior", None)

    def set_expected(self, key: EdgeKey, turn: int, holds: bool = True) -> Edge:
        """Edge predicted from an executed action but not yet observed, P=0.7 (or 0.3)."""
        return self._set(key, P_EXPECTED if holds else 1.0 - P_EXPECTED, "expected", turn)

    def _set(self, key: EdgeKey, p: float, source: str, turn: int | None) -> Edge:
        edge = self.edges.get(key)
        if edge is None:
            edge = self.edges[key] = Edge(key[0], key[1], key[2], p, source, turn)
        else:
            edge.p, edge.source = p, source
            if turn is not None:
                edge.last_seen_turn = turn
        return edge

    def holding(self, subject: str) -> list[tuple[str, str | None]]:
        """True location relations of ``subject``."""
        return [
            (e.predicate, e.target)
            for e in self.edges.values()
            if e.subject == subject and e.predicate in LOCATION_PREDICATES and e.holds
        ]

    def location(self, subject: str) -> str:
        return format_location(self.holding(subject))

    def true_edges(self) -> list[Edge]:
        return [e for e in self.edges.values() if e.holds]

    # --- update ------------------------------------------------------------
    def update(
        self,
        turn: int,
        observations: dict[EdgeKey, bool],
        positions: dict[str, tuple[float, float, float]] | None = None,
    ) -> list[Event]:
        """Apply one round of observations and return the location changes it caused.

        Observed edges get source=obs and P=0.97/0.03. Edges not in ``observations`` keep their P
        and become source=mem (no time decay). Subjects whose location label changed produce an
        Event; a subject's first location observation produces none, even if other predicates of
        it were observed before.
        """
        subjects = {k[1] for k in observations} | {k[1] for k in self.edges}
        before = {s: self.location(s) for s in subjects}
        # Subjects whose location was observed before; a first location observation is no move.
        seen_before = {
            e.subject for e in self.edges.values()
            if e.source != "prior" and e.predicate in LOCATION_PREDICATES
        }

        for edge in self.edges.values():
            if edge.key not in observations and edge.source == "obs":
                edge.source = "mem"
        for key, value in observations.items():
            self._set(key, P_OBS_TRUE if value else P_OBS_FALSE, "obs", turn)
            self.add_node(key[1])
            if key[2] is not None and key[2] != GRIPPER:
                self.add_node(key[2])
        for name, pos in (positions or {}).items():
            node = self.add_node(name)
            node.pos = tuple(float(v) for v in pos)  # type: ignore[assignment]
            node.last_seen_turn = turn

        new_events = []
        for s in sorted(subjects):
            after = self.location(s)
            if s in seen_before and after != before[s]:
                new_events.append(Event(turn, s, before[s], after))
        self.events.extend(new_events)
        self.turn = turn
        return new_events
