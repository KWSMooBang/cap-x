"""Per-trial memory conditions for the multi-turn decision prompt.

memory_mode:
  none        -- the cap-x baseline: no memory block (prompts unchanged).
  vdm_history -- the VDM descriptions of all previous turns, with turn numbers (unstructured text memory).
  bg_oracle   -- Belief Graph built from simulator ground truth, plus S2/S5 signals.
  bg_est      -- Belief Graph estimated from cameras, robot state, API-call hooks, VDM and the task
                 language only, plus S1/S2/S4/S5 signals; simulator truth is an evaluation-only shadow.
  bg_est2     -- bg_est v2 (memory2.py): VDM Observe on marked images, declared Subtask / Effect,
                 subtask layer, S1-S5, belief_trace.jsonl.

All three expose the same calls, so the harness does not branch on the mode:
  reset(low_level_env, **context) after env.reset() (bg_est takes code_env, instruction and the
                                  model query functions as context; the others ignore it)
  initial_text()                  memory text for the first code-generation prompt (or None)
  after_turn(turn, vdm_text, executed)
                                  after each turn -> MemoryTurn; executed=False for a turn that ran no
                                  code (a reply cut off at max_tokens twice)
  check_finish()                  on a FINISH decision -> rejection Signal or None (S5a)
  state()                         small dict for logs
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .estimator import BeliefEstimator, EstConfig, EstTurn
from .hooks import ApiCallRecorder
from .oracle import GroundTruthSource, LiberoMemSource, OracleEstimator
from .perception import (
    PerceptionSource,
    Sam3PerceptionSource,
    SimPerceptionSource,
    proprio_from_obs,
)
from .progress import Progress
from .serialize import serialize
from .shadow import OracleShadow
from .signals import (
    ALL_SIGNALS,
    EST_SIGNALS,
    FinishGate,
    Signal,
    SignalContext,
    default_detectors,
    est_detectors,
)
from .verify import VdmVerifier

MEMORY_MODES = ("none", "vdm_history", "bg_oracle", "bg_est", "bg_est2")
EST_PERCEPTIONS = ("sam3", "sim_fake_for_tests")
DEFAULT_MAX_CHARS = 3000


@dataclass
class MemoryTurn:
    text: str | None
    signals: list[Signal] = field(default_factory=list)

    def signal_dicts(self) -> list[dict[str, str]]:
        return [s.to_dict() for s in self.signals]


class NoMemory:
    mode = "none"

    def reset(self, low_level_env: Any = None, **context: Any) -> None:
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

    def reset(self, low_level_env: Any = None, **context: Any) -> None:
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

    def reset(self, low_level_env: Any = None, source: GroundTruthSource | None = None, **context: Any) -> None:
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


def _usage_sum(usages: list[Any]) -> dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0}
    for u in usages:
        if isinstance(u, dict):
            for k in out:
                out[k] += int(u.get(k) or 0)
    return out


class _CountingQuery:
    """Wraps a model query function; counts calls and keeps each reply's usage."""

    def __init__(self, fn: Callable[[list[dict[str, Any]]], Any]) -> None:
        self.fn = fn
        self.calls = 0
        self.usage: list[Any] = []

    def __call__(self, prompt: list[dict[str, Any]]) -> Any:
        self.calls += 1
        out = self.fn(prompt)
        self.usage.append(out.get("usage") if isinstance(out, dict) else None)
        return out


class BgEstMemory:
    """Estimated Belief Graph (capx.belief.estimator) serialized into the prompt, with S1/S2/S4/S5.

    The estimator gets only the robot observation (``low_level_env.get_observation()``: cameras and
    robot state), the sim step counter, hook records of the sandbox API calls, the task language and
    model replies. The oracle shadow (``shadow=True``) reads simulator truth for the evaluation logs
    only (``turn_log()['shadow']``); it never reaches the prompt.

    ``perception``: ``sam3`` (cap-x's SAM3 service) or ``sim_fake_for_tests`` (simulator poses plus
    noise; tests and CPU checks only, flagged in every log)."""

    mode = "bg_est"

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        signals: tuple[str, ...] | list[str] = EST_SIGNALS,
        max_finish_rejections: int = 1,
        goal_source: str = "llm",
        max_categories: int = 5,
        gate_dist: float = 0.10,
        verify: bool = True,
        perception: str = "sam3",
        shadow: bool = True,
    ) -> None:
        if perception not in EST_PERCEPTIONS:
            raise ValueError(f"bg_est perception must be one of {EST_PERCEPTIONS}, got {perception!r}")
        if goal_source not in ("llm", "bddl"):
            raise ValueError(f"bg_est goal_source must be llm or bddl, got {goal_source!r}")
        self.max_chars = max_chars
        self.enabled_signals = tuple(signals)
        self.max_finish_rejections = max_finish_rejections
        self.goal_source = goal_source
        self.config = EstConfig(gate_dist=gate_dist, max_categories=max_categories)
        self.verify = verify
        self.perception = perception
        self.shadow_on = shadow
        self.estimator: BeliefEstimator | None = None
        self.shadow: OracleShadow | None = None
        self.last: EstTurn | None = None
        self.turn = 0
        self.over_budget = False
        self._log: dict[str, Any] = {}

    def reset(
        self,
        low_level_env: Any = None,
        *,
        code_env: Any = None,
        instruction: str | None = None,
        goal_query_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
        vdm_query_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
        perception_source: PerceptionSource | None = None,
        shadow_source: GroundTruthSource | None = None,
        **context: Any,
    ) -> None:
        low = low_level_env
        self.low = low
        if perception_source is None:
            if self.perception == "sim_fake_for_tests":
                perception_source = SimPerceptionSource(LiberoMemSource(low))
            else:
                perception_source = Sam3PerceptionSource()
        self.recorder = ApiCallRecorder(
            proprio_fn=lambda: proprio_from_obs(low.get_observation()),
            clock_fn=lambda: getattr(low, "_sim_step_count", None),
        )
        if code_env is not None:
            code_env.api_call_wrapper = self.recorder.wrap
        self.goal_query = _CountingQuery(goal_query_fn) if goal_query_fn is not None else None
        self.vdm_query = _CountingQuery(vdm_query_fn) if (vdm_query_fn is not None and self.verify) else None
        verifier = VdmVerifier(self.vdm_query, self.config.max_hypotheses) if self.vdm_query is not None else None
        self.estimator = BeliefEstimator(perception_source, self.config, goal_query_fn=self.goal_query,
                                         verifier=verifier)
        self.shadow = None
        if self.shadow_on and (shadow_source is not None or hasattr(low, "handle")):
            self.shadow = OracleShadow(shadow_source or LiberoMemSource(low))
        goal_fn = None
        if self.goal_source == "bddl":
            if self.shadow is None:
                raise ValueError("bg_est goal_source=bddl needs the simulator (oracle shadow)")
            goal_fn = self.shadow.bddl_goal
        self.detectors = est_detectors(self.enabled_signals)
        self.gate = FinishGate(self.max_finish_rejections if "S5" in self.enabled_signals else 0)
        first = self.estimator.start(low.get_observation(), instruction or "", goal_fn)
        self.last = first
        self.s2_baseline = first.progress
        self.turn = 0
        self._log = {"est": first.to_log(), "goal": self.estimator.goal.to_dict(),
                     "perception": self.perception_label}
        if self.shadow is not None:
            self._log["shadow"] = self.shadow.start(self.estimator)
            self._shadow_baseline = self.shadow.progress

    @property
    def perception_label(self) -> str:
        """Configured perception; sim_fake is spelled out so no log can hide it."""
        if self.estimator is not None and self.estimator.perception_name == "sim_fake":
            return "sim_fake_for_tests (FAKE: simulator poses + noise, not a valid evaluation)"
        return self.estimator.perception_name if self.estimator is not None else self.perception

    def render(self, signals: list[Signal] = ()) -> str:  # type: ignore[assignment]
        assert self.estimator is not None, "reset() first"
        uncertain = bool(self.last and self.last.uncertain)
        text = self.estimator.render(signals, self.max_chars, uncertain=uncertain)
        self.over_budget = len(text) > self.max_chars
        return text

    def initial_text(self) -> str | None:
        return self.render()

    def after_turn(self, turn: int, vdm_text: str | None = None, executed: bool = True) -> MemoryTurn:
        """Update from this turn's hook records and the robot observation. A turn that ran no code is
        still observed but cannot fire S1 / S2 (S2 then compares with the last executed turn)."""
        assert self.estimator is not None, "reset() first"
        records = self.recorder.drain()
        r = self.estimator.end_turn(turn, self.low.get_observation(), records, executed=executed)
        ctx = SignalContext(turn, self.estimator.graph, r.progress, self.s2_baseline, r.events, executed=executed,
                            violations=r.violations, uncertain=r.uncertain,
                            suppress_completion=r.suppress_completion)
        fired = [s for d in self.detectors if (s := d.detect(ctx)) is not None]
        self.last, self.turn = r, turn
        if executed:
            self.s2_baseline = r.progress
        self._log = {"est": r.to_log(), "perception": self.perception_label}
        if self.shadow is not None:
            self._log["shadow"] = self.shadow.compare(turn, self.estimator, r, fired, self._shadow_baseline)
            if executed:
                self._shadow_baseline = self.shadow.progress
        return MemoryTurn(self.render(fired), fired)

    def check_finish(self) -> Signal | None:
        assert self.last is not None, "reset() first"
        rejection = self.gate.check(self.last.progress, suppress=self.last.suppress_completion)
        if self.shadow is not None:
            self._log.setdefault("shadow", {})["finish_check"] = self.shadow.finish_check(rejection is not None)
        return rejection

    def state(self) -> dict[str, Any]:
        """Progress numbers, perception source and extra model / detector calls for logs."""
        if self.estimator is None or self.last is None:
            return {}
        out = self.estimator.state()
        out.update(perception=self.perception_label, over_budget=self.over_budget,
                   uncertain=len(self.last.uncertain), suppress_completion=self.last.suppress_completion,
                   calls=self.calls())
        if self.shadow is not None:
            out["shadow_totals"] = self.shadow.totals
        return out

    def calls(self) -> dict[str, Any]:
        q, v = self.goal_query, self.vdm_query
        return {
            "goal_parse": q.calls if q else 0, "goal_parse_tokens": _usage_sum(q.usage) if q else {},
            "verify_vdm": v.calls if v else 0, "verify_vdm_tokens": _usage_sum(v.usage) if v else {},
            "perception": getattr(self.estimator.perception, "calls", 0) if self.estimator else 0,
        }

    def turn_log(self) -> dict[str, Any]:
        """Estimator details and the oracle shadow for this turn (logs only, never the prompt)."""
        return self._log


def make_memory(mode: str = "none", **opts: Any) -> NoMemory | VdmHistoryMemory | BgOracleMemory | BgEstMemory:
    """Memory for one trial. ``opts``: max_chars (all graph / history modes); signals,
    max_finish_rejections (bg_oracle, bg_est); visibility (bg_oracle); goal_source, max_categories,
    gate_dist, verify, perception, shadow (bg_est). Unknown options for a mode are ignored."""
    if mode == "none":
        return NoMemory()
    if mode == "vdm_history":
        return VdmHistoryMemory(max_chars=opts.get("max_chars", DEFAULT_MAX_CHARS))
    if mode == "bg_oracle":
        keys = ("max_chars", "signals", "max_finish_rejections", "visibility")
        return BgOracleMemory(**{k: opts[k] for k in keys if k in opts})
    if mode == "bg_est":
        keys = ("max_chars", "signals", "max_finish_rejections", "goal_source", "max_categories", "gate_dist",
                "verify", "perception", "shadow")
        return BgEstMemory(**{k: opts[k] for k in keys if k in opts})
    if mode == "bg_est2":
        from .memory2 import BgEst2Memory

        keys = ("max_chars", "signals", "max_finish_rejections", "goal_source", "max_categories", "gate_dist",
                "observe", "perception", "shadow", "observe_max_statements", "event_frames")
        return BgEst2Memory(**{k: opts[k] for k in keys if k in opts})
    raise ValueError(f"Unknown memory_mode {mode!r}; expected one of {MEMORY_MODES}")
