"""Refinement signals computed from the Belief Graph and task progress.

Implemented now (computable from the oracle graph):
  S2 progress stall: a turn ended and Delta (remaining steps) did not go down.
  S5 completion check: (a) the model says FINISH while steps remain -> reject once and re-ask;
     (b) Delta is 0 -> tell the model all steps are verified complete (the harness does not end
     the trial itself; the model still decides FINISH).
Reserved for the API-hook / estimated-graph stage: S1 expectation violation, S3 precondition,
S4 uncertainty. They plug in as more ``SignalDetector`` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .graph import BeliefGraph, Event
from .progress import Progress, format_step

ALL_SIGNALS = ("S2", "S5")


@dataclass
class Signal:
    id: str  # S2, S5a, S5b, ...
    reason: str  # short machine-readable cause, logged
    text: str  # what goes into the prompt

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "reason": self.reason, "text": self.text}


@dataclass
class SignalContext:
    turn: int
    graph: BeliefGraph
    progress: Progress
    prev_progress: Progress | None
    events: list[Event] = field(default_factory=list)
    executed: bool = True  # False: the turn ran no code (reply cut off at max_tokens twice)


class SignalDetector(Protocol):
    id: str

    def detect(self, ctx: SignalContext) -> Signal | None: ...


class ProgressStall:
    """S2: an executed turn ended with Delta unchanged (and not zero, not overshot, progress known).
    ``prev_progress`` is the progress after the last executed turn."""

    id = "S2"

    def detect(self, ctx: SignalContext) -> Signal | None:
        if not ctx.executed or ctx.prev_progress is None or ctx.progress.overshot:
            return None
        if not (ctx.progress.known and ctx.prev_progress.known) or ctx.progress.remaining == 0:
            return None
        before, after = ctx.prev_progress.remaining, ctx.progress.remaining
        if after < before:  # type: ignore[operator]
            return None
        return Signal(
            "S2",
            f"remaining {before} -> {after}",
            f"no progress this turn (remaining steps still {after}). Reconsider the strategy: work out why "
            "the last code did not complete the next step before writing new code; do not just repeat it.",
        )


class CompletionVerified:
    """S5(b): every step is verified complete."""

    id = "S5"

    def detect(self, ctx: SignalContext) -> Signal | None:
        if not ctx.progress.complete:
            return None
        return Signal("S5b", "remaining 0", "all steps verified complete. If you agree, respond FINISH.")


def default_detectors(enabled: tuple[str, ...] | list[str] = ALL_SIGNALS) -> list[SignalDetector]:
    out: list[SignalDetector] = []
    if "S2" in enabled:
        out.append(ProgressStall())
    if "S5" in enabled:
        out.append(CompletionVerified())
    return out


class FinishGate:
    """S5(a): reject FINISH while steps remain, at most ``max_rejections`` times per trial.

    Not applied after an overshoot (the task can no longer succeed) or when progress is unknown."""

    def __init__(self, max_rejections: int = 1) -> None:
        self.max_rejections = max_rejections
        self.rejections = 0

    def check(self, progress: Progress) -> Signal | None:
        """A rejection signal if this FINISH should be refused, else None (accept)."""
        # Overshot: the task can no longer succeed, so re-asking cannot help. Unknown progress:
        # remaining steps cannot be named, so do not override the model.
        if (progress.complete or progress.overshot or not progress.known
                or self.rejections >= self.max_rejections):
            return None
        self.rejections += 1
        steps = progress.remaining_steps()
        shortest = min(steps, key=len) if steps else []
        left = f"{progress.remaining} step(s) remain: " + "; then ".join(format_step(s) for s in shortest)
        if len(steps) > 1:
            left += " (or another branch of the goal)"
        return Signal(
            "S5a",
            f"finish with remaining {progress.remaining}",
            f"FINISH rejected: the task is not complete — {left}. Respond REGENERATE with new code to "
            "continue, or FINISH again to stop anyway.",
        )
