"""``memory_mode: bg_est2`` -- the estimated Belief Graph v2 as a per-trial memory condition.

Same calls as the other memories (``memory.py``) plus:

* ``prompt_rules()`` -> the M4 declaration rules, appended by the harness to the first prompt and to
  every decision prompt (``bg_est2`` only; other modes return None and their prompts are unchanged);
* ``after_turn(turn, vdm_text, executed, code=..., event_frames=...)``: ``code`` is the code block
  that ran this turn (its ``# Subtask`` / ``# Effect`` lines are read), ``event_frames`` optional
  recorded frames after gripper close / open (``event_frames=True`` only);
* ``trace_path`` (reset context): ``<trial_dir>/belief_trace.jsonl``, one JSON line at the start, per
  turn end and per event-frame Observe.

The estimator sees only the robot observation, hook records, model replies and the task language;
``perception=sim_fake_for_tests`` and the oracle shadow read the simulator (tests / logs only).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any

from .estimator2 import BeliefEstimatorV2, Est2Config, EstTurn2
from .hooks import ApiCallRecorder
from .memory import DEFAULT_MAX_CHARS, EST_PERCEPTIONS, MemoryTurn, _CountingQuery, _usage_sum
from .observe import VdmObserver, statement_text
from .oracle import GroundTruthSource, LiberoMemSource
from .perception import PerceptionSource, Sam3PerceptionSource, SimPerceptionSource, proprio_from_obs
from .prompts_v2 import M4_RULES
from .shadow import OracleShadowV2
from .signals import EST2_SIGNALS, FinishGate, Signal, SignalContextV2, est2_detectors


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


class BgEst2Memory:
    mode = "bg_est2"

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CHARS,
        signals: tuple[str, ...] | list[str] = EST2_SIGNALS,
        max_finish_rejections: int = 1,
        goal_source: str = "llm",
        max_categories: int = 5,
        gate_dist: float = 0.10,
        observe: bool = True,
        perception: str = "sam3",
        shadow: bool = True,
        observe_max_statements: int = 10,
        event_frames: bool = False,
    ) -> None:
        if perception not in EST_PERCEPTIONS:
            raise ValueError(f"bg_est2 perception must be one of {EST_PERCEPTIONS}, got {perception!r}")
        if goal_source not in ("llm", "bddl"):
            raise ValueError(f"bg_est2 goal_source must be llm or bddl, got {goal_source!r}")
        self.max_chars = max_chars
        self.enabled_signals = tuple(signals)
        self.max_finish_rejections = max_finish_rejections
        self.goal_source = goal_source
        self.config = Est2Config(gate_dist=gate_dist, max_categories=max_categories,
                                 observe_max_statements=observe_max_statements, event_frames=event_frames)
        self.observe_on = observe
        self.perception = perception
        self.shadow_on = shadow
        self.estimator: BeliefEstimatorV2 | None = None
        self.shadow: OracleShadowV2 | None = None
        self.last: EstTurn2 | None = None
        self.turn = 0
        self.over_budget = False
        self.trace_path: str | None = None
        self._log: dict[str, Any] = {}
        self._last_signals: list[Signal] = []

    def prompt_rules(self) -> str:
        """M4: appended to the first prompt and every decision prompt (bg_est2 only)."""
        return M4_RULES

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
        trace_path: str | None = None,
        **context: Any,
    ) -> None:
        low = self.low = low_level_env
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
        self.vdm_query = _CountingQuery(vdm_query_fn) if (vdm_query_fn is not None and self.observe_on) else None
        observer = VdmObserver(self.vdm_query) if self.vdm_query is not None else None
        self.estimator = BeliefEstimatorV2(perception_source, self.config, goal_query_fn=self.goal_query,
                                           observer=observer)
        self.shadow = None
        if self.shadow_on and (shadow_source is not None or hasattr(low, "handle")):
            self.shadow = OracleShadowV2(shadow_source or LiberoMemSource(low))
        goal_fn = None
        if self.goal_source == "bddl":
            if self.shadow is None:
                raise ValueError("bg_est2 goal_source=bddl needs the simulator (oracle shadow)")
            goal_fn = self.shadow.bddl_goal
        self.detectors = est2_detectors(self.enabled_signals)
        self.gate = FinishGate(self.max_finish_rejections if "S5" in self.enabled_signals else 0)
        self.trace_path = trace_path
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            open(trace_path, "w").close()
        first = self.estimator.start(low.get_observation(), instruction or "", goal_fn)
        self.last = first
        self.s2_baseline = first.progress
        self.turn = 0
        self._last_signals = []
        self._log = {"est": first.to_log(), "goal": self.estimator.goal.to_dict(), "perception": self.perception_label,
                     "cot_record": self.estimator.cot}
        if self.shadow is not None:
            self._log["shadow"] = self.shadow.start(self.estimator)
            self._shadow_baseline = self.shadow.progress
        self._write_trace("start")

    @property
    def perception_label(self) -> str:
        if self.estimator is not None and self.estimator.perception_name == "sim_fake":
            return "sim_fake_for_tests (FAKE: simulator poses + noise, not a valid evaluation)"
        return self.estimator.perception_name if self.estimator is not None else self.perception

    def render(self, signals: Sequence[Signal] = ()) -> str:
        assert self.estimator is not None, "reset() first"
        uncertain = bool(self.last and (self.last.uncertain or self.last.suppress_completion))
        text = self.estimator.render(signals, self.max_chars, uncertain=uncertain)
        self.over_budget = len(text) > self.max_chars
        return text

    def initial_text(self) -> str | None:
        return self.render()

    def after_turn(self, turn: int, vdm_text: str | None = None, executed: bool = True, code: str | None = None,
                   event_frames: Sequence[dict[str, Any]] = ()) -> MemoryTurn:
        assert self.estimator is not None, "reset() first"
        records = self.recorder.drain()
        frames = event_frames if self.config.event_frames else ()
        r = self.estimator.end_turn(turn, self.low.get_observation(), records, executed=executed,
                                    code=code if executed else None, event_frames=frames)
        ctx = SignalContextV2(turn, self.estimator.graph, r.progress, self.s2_baseline, r.events, executed=executed,
                              violations=r.violations, uncertain=r.uncertain,
                              suppress_completion=r.suppress_completion, s1_items=r.s1_items,
                              subtask_progress=r.subtask_progress, s3_items=r.s3_items, s4_items=r.s4_items)
        fired = [s for d in self.detectors if (s := d.detect(ctx)) is not None]
        self.last, self.turn = r, turn
        if executed:
            self.s2_baseline = r.progress
        self.estimator.set_signals(fired)
        self._last_signals = fired
        self._log = {"est": r.to_log(), "perception": self.perception_label, "cot_record": self.estimator.cot}
        if self.shadow is not None:
            self._log["shadow"] = self.shadow.compare(turn, self.estimator, r, fired, self._shadow_baseline,
                                                      had_code=bool(executed and code))
            if executed:
                self._shadow_baseline = self.shadow.progress
        for ev in r.event_observes:
            self._write_trace("event", extra={"event": ev})
        self._write_trace("turn")
        return MemoryTurn(self.render(fired), fired)

    def check_finish(self) -> Signal | None:
        assert self.last is not None, "reset() first"
        rejection = self.gate.check(self.last.progress, suppress=self.last.suppress_completion)
        if self.shadow is not None:
            self._log.setdefault("shadow", {})["finish_check"] = self.shadow.finish_check(rejection is not None)
        return rejection

    def state(self) -> dict[str, Any]:
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
            "observe_vdm": v.calls if v else 0, "observe_vdm_tokens": _usage_sum(v.usage) if v else {},
            "perception": getattr(self.estimator.perception, "calls", 0) if self.estimator else 0,
        }

    def turn_log(self) -> dict[str, Any]:
        """Estimator details, the CoT record and the oracle shadow (logs only, never the prompt)."""
        return self._log

    def _write_trace(self, kind: str, extra: dict[str, Any] | None = None) -> None:
        if not self.trace_path or self.estimator is None:
            return
        rec = self.estimator.trace(kind)
        rec["signals"] = [s.to_dict() for s in self._last_signals] if kind != "start" else []
        rec["perception"] = self.perception_label
        if self.shadow is not None:
            # Evaluation only: the simulator truth of every key, next to the estimate.
            for text, k in rec["keys"].items():
                key = self._key_of(text)
                k["shadow"] = self.shadow.truth(self.shadow.to_sim(key, self.estimator)) if key else None
            rec["shadow"] = self._log.get("shadow")
        if extra:
            rec.update(extra)
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")

    def _key_of(self, text: str) -> tuple | None:
        est = self.estimator
        for key in est.graph.edges if est is not None else ():
            if statement_text(key) == text:
                return key
        return None
