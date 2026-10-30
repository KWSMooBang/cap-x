"""Belief Graph -> prompt text, within a character budget.

Format:
    Belief graph (turn N):
    Progress: step 1/3 done (akita_black_bowl_1 on plate_3) | next: akita_black_bowl_2 on plate_1 | remaining: 2
    Objects: akita_black_bowl_1 at (x, y, z) on plate_3 | ...
    Events: t1 akita_black_bowl_1 on plate_1 -> on plate_3 | ...
    Signals: S2 no progress this turn ...

When the text is longer than ``max_chars``, the oldest events are dropped first (replaced by a
"(k earlier omitted)" marker); if it is still too long the objects line is shortened, and as a
last resort the whole text is cut at ``max_chars``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .graph import BeliefGraph, Event
from .progress import Pred, Progress, progress_line
from .signals import Signal

TRUNCATION_MARK = " ...[truncated]"


def objects_line(graph: BeliefGraph, subjects: Sequence[str] | None = None) -> str:
    names = list(subjects) if subjects is not None else sorted(
        {e.subject for e in graph.edges.values()} | {n for n, node in graph.nodes.items() if node.pos is not None}
    )
    parts = []
    for name in names:
        node = graph.nodes.get(name)
        label = node.label if node else name
        if node is not None and node.pos is not None:
            label += " at ({:.2f}, {:.2f}, {:.2f})".format(*((round(v, 2) or 0.0) for v in node.pos))
        parts.append(f"{label} {graph.location(name)}")
    return "Objects: " + " | ".join(parts)


def events_line(events: Sequence[Event], omitted: int = 0) -> str:
    texts = [e.text for e in events]
    if omitted:
        texts.insert(0, f"({omitted} earlier omitted)")
    return "Events: " + (" | ".join(texts) if texts else "none")


def signals_line(signals: Sequence[Signal]) -> str:
    return "Signals: " + (" | ".join(f"{s.id} {s.text}" for s in signals) if signals else "none")


def serialize(
    graph: BeliefGraph,
    progress: Progress | None,
    signals: Sequence[Signal] = (),
    *,
    turn: int,
    max_chars: int = 3000,
    subjects: Sequence[str] | None = None,
    current_truths: dict[Pred, bool] | None = None,
) -> str:
    head = [f"Belief graph (turn {turn}):"]
    if progress is not None:
        head.append(progress_line(progress, current_truths))
    head.append(objects_line(graph, subjects))
    tail = signals_line(signals)

    events = list(graph.events)
    omitted = 0
    while True:
        text = "\n".join(head + [events_line(events, omitted), tail])
        if len(text) <= max_chars or not events:
            break
        events.pop(0)
        omitted += 1
    if len(text) > max_chars:
        # Shorten the objects line first so progress and signals survive.
        lines = text.split("\n")
        obj_idx = len(head) - 1
        budget = len(lines[obj_idx]) - (len(text) - max_chars) - len(TRUNCATION_MARK)
        if budget > len("Objects: "):
            lines[obj_idx] = lines[obj_idx][:budget] + TRUNCATION_MARK
            text = "\n".join(lines)
    if len(text) > max_chars:
        mark = TRUNCATION_MARK if max_chars > len(TRUNCATION_MARK) else ""
        text = text[: max_chars - len(mark)] + mark
    return text
