"""Per-trial memory conditions for the multi-turn decision prompt.

memory_mode:
  none        -- the cap-x baseline: no memory block (prompts unchanged).
  vdm_history -- the VDM descriptions of all previous turns, with turn numbers (unstructured text memory).
  bg_oracle   -- Belief Graph built from simulator ground truth, plus S2/S5 signals.

All three expose the same calls, so the harness does not branch on the mode:
  reset(low_level_env)            after env.reset()
  initial_text()                  memory text for the first code-generation prompt (or None)
  after_turn(turn, vdm_text, executed)
                                  after each turn -> MemoryTurn; executed=False for a turn that ran no
                                  code (a reply cut off at max_tokens twice)
  check_finish()                  on a FINISH decision -> rejection Signal or None (S5a)
  state()                         small dict for logs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .oracle import GroundTruthSource, LiberoMemSource, OracleEstimator
from .progress import Progress
from .serialize import serialize
from .signals import ALL_SIGNALS, FinishGate, Signal, SignalContext, default_detectors

MEMORY_MODES = ("none", "vdm_history", "bg_oracle")
DEFAULT_MAX_CHARS = 3000


@dataclass
class MemoryTurn:
    text: str | None
    signals: list[Signal] = field(default_factory=list)

    def signal_dicts(self) -> list[dict[str, str]]:
        return [s.to_dict() for s in self.signals]


class NoMemory:
    mode = "none"

    def reset(self, low_level_env: Any = None) -> None:
        pass

    def initial_text(self) -> str | None:
        return None

    def after_turn(self, turn: int, vdm_text: str | None = None, executed: bool = True) -> MemoryTurn:
        return MemoryTurn(None)

    def check_finish(self) -> Signal | None:
        return None

    def state(self) -> dict[str, Any]:
        return {}


class VdmHistoryMemory:
    """Accumulates VDM descriptions as ``turn k: ...``; the oldest are dropped past ``max_chars``.

    The text for turn k holds turns before k only: the decision prompt already shows turn k's VDM
    description in its own block."""

    mode = "vdm_history"
    header = "VDM history (previous turns, oldest first):"

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.max_chars = max_chars
        self.entries: list[tuple[int, str]] = []

    def reset(self, low_level_env: Any = None) -> None:
        self.entries = []

    def initial_text(self) -> str | None:
        return None

    def add(self, turn: int, vdm_text: str | None) -> None:
        if vdm_text:
            self.entries.append((turn, " ".join(vdm_text.split())))

    def render(self) -> str | None:
        if not self.entries:
            return None
        lines = [f"turn {t}: {s}" for t, s in self.entries]
        omitted = 0
        while True:
            body = ([f"({omitted} earlier turns omitted)"] if omitted else []) + lines
            text = "\n".join([self.header] + body)
            if len(text) <= self.max_chars or len(lines) == 1:
                break
            lines.pop(0)
            omitted += 1
        if len(text) > self.max_chars:  # a single entry longer than the budget: keep its end
            keep = self.max_chars - len(self.header) - len("\n...")
            if keep > 0:
                text = self.header + "\n..." + text[len(text) - keep:]
            else:  # budget smaller than the header: just the newest characters
                text = text[len(text) - self.max_chars:] if self.max_chars > 0 else ""
        return text or None

    def after_turn(self, turn: int, vdm_text: str | None = None, executed: bool = True) -> MemoryTurn:
        """History of the turns before ``turn`` (this turn's VDM text is already in the prompt's VDM
        block); ``vdm_text`` is then stored for later turns."""
        text = self.render()
        self.add(turn, vdm_text)
        return MemoryTurn(text)

    def check_finish(self) -> Signal | None:
        return None

    def state(self) -> dict[str, Any]:
        return {"entries": len(self.entries)}


class BgOracleMemory:
    """Ground-truth Belief Graph serialized into the prompt, with S2 / S5 signals."""

    mode = "bg_oracle"

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        signals: tuple[str, ...] | list[str] = ALL_SIGNALS,
        max_finish_rejections: int = 1,
        visibility: str = "full",
    ) -> None:
        self.max_chars = max_chars
        self.enabled_signals = tuple(signals)
        self.max_finish_rejections = max_finish_rejections
        self.visibility = visibility
        self.estimator: OracleEstimator | None = None
        self.progress: Progress | None = None
        self.turn = 0
        self.over_budget = False

    def reset(self, low_level_env: Any = None, source: GroundTruthSource | None = None) -> None:
        if source is None:
            source = LiberoMemSource(low_level_env)
        self.estimator = OracleEstimator(source, visibility=self.visibility)
        self.detectors = default_detectors(self.enabled_signals)
        self.gate = FinishGate(self.max_finish_rejections if "S5" in self.enabled_signals else 0)
        _, self.progress = self.estimator.observe(0)
        self.s2_baseline = self.progress  # progress after the last executed turn
        self.turn = 0

    def render(self, signals: list[Signal] = ()) -> str:  # type: ignore[assignment]
        assert self.estimator is not None, "reset() first"
        text = serialize(
            self.estimator.graph, self.progress, signals,
            turn=self.turn, max_chars=self.max_chars,
            subjects=self.estimator.source.movable_objects(),
            current_truths=self.estimator.current_truths(),
        )
        # Title, Progress and Signals alone exceeded max_chars (serialize never cuts them).
        self.over_budget = len(text) > self.max_chars
        return text

    def initial_text(self) -> str | None:
        return self.render()

    def after_turn(self, turn: int, vdm_text: str | None = None, executed: bool = True) -> MemoryTurn:
        """Observe the sim after ``turn``. A turn that ran no code is still observed (the text keeps
        the turn number current) but cannot fire S2; S2 then compares the next executed turn with the
        last executed one."""
        assert self.estimator is not None, "reset() first"
        events, progress = self.estimator.observe(turn)
        ctx = SignalContext(turn, self.estimator.graph, progress, self.s2_baseline, events, executed=executed)
        fired = [s for d in self.detectors if (s := d.detect(ctx)) is not None]
        self.progress, self.turn = progress, turn
        if executed:
            self.s2_baseline = progress
        return MemoryTurn(self.render(fired), fired)

    def check_finish(self) -> Signal | None:
        assert self.progress is not None, "reset() first"
        return self.gate.check(self.progress)

    def state(self) -> dict[str, Any]:
        """Progress numbers for logs."""
        p = self.progress
        if p is None:
            return {}
        return {"done": len(p.done), "remaining": p.remaining, "overshot": p.overshot, "known": p.known,
                "over_budget": self.over_budget}


def make_memory(mode: str = "none", **opts: Any) -> NoMemory | VdmHistoryMemory | BgOracleMemory:
    """Memory for one trial. ``opts``: max_chars (vdm_history, bg_oracle); signals,
    max_finish_rejections, visibility (bg_oracle). Unknown options for a mode are ignored."""
    if mode == "none":
        return NoMemory()
    if mode == "vdm_history":
        return VdmHistoryMemory(max_chars=opts.get("max_chars", DEFAULT_MAX_CHARS))
    if mode == "bg_oracle":
        keys = ("max_chars", "signals", "max_finish_rejections", "visibility")
        return BgOracleMemory(**{k: opts[k] for k in keys if k in opts})
    raise ValueError(f"Unknown memory_mode {mode!r}; expected one of {MEMORY_MODES}")
