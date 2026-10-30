"""Belief Graph -> prompt text, within a character budget.

Format:
    Belief graph (turn N):
    Progress: step 1/3 done (akita_black_bowl_1 on plate_3) | next: akita_black_bowl_2 on plate_1 | remaining: 2
    Objects (xyz in robot base frame, m): akita_black_bowl_1 at (x, y, z) on plate_3 | ...
    Events: t1 akita_black_bowl_1 on plate_1 -> on plate_3 | ...
    Signals: S2 no progress this turn ...

When the text is longer than ``max_chars``, the oldest events are dropped first (replaced by a
"(k earlier omitted)" marker), then the objects line is shortened, then the objects and events
lines are left out. The title, Progress and Signals lines are never cut; if they alone exceed
``max_chars`` the text is returned over budget.
"""

from __future__ import annotations

from collections.abc import Sequence

from .graph import BeliefGraph, Event
from .progress import Pred, Progress, progress_line
from .signals import Signal

TRUNCATION_MARK = " ...[truncated]"
OBJECTS_PREFIX = "Objects (xyz in robot base frame, m): "


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
    return OBJECTS_PREFIX + " | ".join(parts)


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
    title = f"Belief graph (turn {turn}):"
    prog = progress_line(progress, current_truths) if progress is not None else None
    objects = objects_line(graph, subjects)
    sig = signals_line(signals)

    def compose(objects_text: str | None, events: list[Event], omitted: int) -> str:
        lines = [title] + ([prog] if prog else [])
        if objects_text is not None:
            lines += [objects_text, events_line(events, omitted)]
        return "\n".join(lines + [sig])

    # 1. Drop the oldest events.
    events = list(graph.events)
    omitted = 0
    text = compose(objects, events, omitted)
    while len(text) > max_chars and events:
        events.pop(0)
        omitted += 1
        text = compose(objects, events, omitted)
    # 2. Shorten the objects line.
    if len(text) > max_chars:
        budget = len(objects) - (len(text) - max_chars) - len(TRUNCATION_MARK)
        objects = objects[: max(budget, len(OBJECTS_PREFIX))] + TRUNCATION_MARK
        text = compose(objects, events, omitted)
    # 3. Drop objects and events entirely. Title, progress and signals are never cut, even if
    #    they alone exceed max_chars.
    if len(text) > max_chars:
        text = compose(None, events, omitted)
    return text
