"""Refinement signals computed from the Belief Graph and task progress.

Implemented now (computable from the oracle graph):
  S2 progress stall: a turn ended and Delta (remaining steps) did not go down.
  S5 completion check: (a) the model says FINISH while steps remain -> reject once and re-ask;
     (b) Delta is 0 -> tell the model all steps are verified complete (the harness does not end
     the trial itself; the model still decides FINISH).
Estimated graph (``bg_est``) only:
  S1 expectation violation: an effect expected from this turn's API calls (e.g. an object in hand
     after close_gripper) is contradicted by an observation.
  S4 uncertainty: a goal-related predicate has 0.3 <= P <= 0.7 or rests on an old memory only ->
     suggest re-observing.
  With an estimated graph S5 (a) and (b) are suppressed when progress is unknown or uncertain
  (``SignalContext.suppress_completion``, ``FinishGate.check(..., suppress=True)``).
Out of scope: S3 precondition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .graph import BeliefGraph, Event
from .progress import Progress, format_step

ALL_SIGNALS = ("S2", "S5")
EST_SIGNALS = ("S1", "S2", "S4", "S5")


@dataclass
class Signal:
    id: str  # S2, S5a, S5b, ...
    reason: str  # short machine-readable cause, logged
    text: str  # what goes into the prompt

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "reason": self.reason, "text": self.text}


@dataclass
class Violation:
    """S1: an expected effect of an API call contradicted by an observation."""

    turn: int
    call: str  # e.g. "close_gripper (sim step 1234)"
    expected: str  # e.g. "bowl_A in gripper"
    observed: str  # e.g. "gripper empty (gripper 0.00)"

    @property
    def text(self) -> str:
        return f"{self.call}: expected {self.expected}, observed {self.observed}"

    def to_dict(self) -> dict[str, object]:
        return {"turn": self.turn, "call": self.call, "expected": self.expected, "observed": self.observed}


@dataclass
class Uncertain:
    """S4: a goal-related predicate the estimate is unsure about."""

    pred: str  # e.g. "bowl_A on plate_C"
    p: float
    source: str
    last_seen_turn: int | None
    reason: str  # "p" (0.3-0.7) or "stale" (old memory only)

    @property
    def text(self) -> str:
        seen = f", last seen t{self.last_seen_turn}" if self.last_seen_turn is not None else ", never seen"
        return f"{self.pred} (P {self.p:.2f} {self.source}{seen})"

    def to_dict(self) -> dict[str, object]:
        return {"pred": self.pred, "p": round(self.p, 3), "source": self.source,
                "last_seen_turn": self.last_seen_turn, "reason": self.reason}


@dataclass
class SignalContext:
    turn: int
    graph: BeliefGraph
    progress: Progress
    prev_progress: Progress | None
    events: list[Event] = field(default_factory=list)
    executed: bool = True  # False: the turn ran no code (reply cut off at max_tokens twice)
    # Estimated graph only (defaults keep the oracle behaviour).
    violations: list[Violation] = field(default_factory=list)
    uncertain: list[Uncertain] = field(default_factory=list)
    suppress_completion: bool = False  # progress unknown/uncertain: no S5b


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

    def __init__(self, text: str = "all steps verified complete. If you agree, respond FINISH.") -> None:
        self.text = text

    def detect(self, ctx: SignalContext) -> Signal | None:
        if not ctx.progress.complete or ctx.suppress_completion:
            return None
        return Signal("S5b", "remaining 0", self.text)


class ExpectationViolation:
    """S1: this turn's expected effects contradicted by observation (estimated graph)."""

    id = "S1"

    def detect(self, ctx: SignalContext) -> Signal | None:
        mine = [v for v in ctx.violations if v.turn == ctx.turn]
        if not ctx.executed or not mine:
            return None
        return Signal(
            "S1",
            f"{len(mine)} violation(s)",
            "expectation violated: " + "; ".join(v.text for v in mine)
            + ". The call did not have the expected effect; check why before building on it.",
        )


class UncertainBelief:
    """S4: goal-related predicates with 0.3 <= P <= 0.7 or only old memory (estimated graph)."""

    id = "S4"

    def detect(self, ctx: SignalContext) -> Signal | None:
        if not ctx.uncertain:
            return None
        return Signal(
            "S4",
            "; ".join(f"{u.pred}:{u.reason}" for u in ctx.uncertain),
            "uncertain: " + "; ".join(u.text for u in ctx.uncertain)
            + ". Re-observe before relying on these: move the arm out of the camera view (e.g. "
            "goto_home_joint_position()) or look from another viewpoint, then check again.",
        )


def default_detectors(enabled: tuple[str, ...] | list[str] = ALL_SIGNALS) -> list[SignalDetector]:
    out: list[SignalDetector] = []
    if "S2" in enabled:
        out.append(ProgressStall())
    if "S5" in enabled:
        out.append(CompletionVerified())
    return out


def est_detectors(enabled: tuple[str, ...] | list[str] = EST_SIGNALS) -> list[SignalDetector]:
    """Detectors for the estimated graph, in prompt order S1, S2, S4, S5b."""
    out: list[SignalDetector] = []
    if "S1" in enabled:
        out.append(ExpectationViolation())
    if "S2" in enabled:
        out.append(ProgressStall())
    if "S4" in enabled:
        out.append(UncertainBelief())
    if "S5" in enabled:
        out.append(CompletionVerified("all steps estimated complete. If you agree, respond FINISH."))
    return out


class FinishGate:
    """S5(a): reject FINISH while steps remain, at most ``max_rejections`` times per trial.

    Not applied after an overshoot (the task can no longer succeed) or when progress is unknown."""

    def __init__(self, max_rejections: int = 1) -> None:
        self.max_rejections = max_rejections
        self.rejections = 0

    def check(self, progress: Progress, suppress: bool = False) -> Signal | None:
        """A rejection signal if this FINISH should be refused, else None (accept).

        ``suppress``: the estimated progress is uncertain, so a wrong rejection could mislead."""
        # Overshot: the task can no longer succeed, so re-asking cannot help. Unknown progress:
        # remaining steps cannot be named, so do not override the model.
        if (progress.complete or progress.overshot or not progress.known or suppress
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
