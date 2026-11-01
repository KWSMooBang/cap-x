"""Rule-based verifier: the only code that changes P (design: diff 제안과 검증). The agent never
writes a confidence.

Rules (each decision is logged with its rule id):

* R1 Observe 1 / 0 -> P 0.9 / 0.1, source obs.
* R2 Observe ? -> previous value kept (source becomes mem).
* R3 robot state -> 0.97 / 0.03, source robot (inhand, handempty); robot wins over Observe (logged
  as a conflict). A held object is on / in nothing: its location edges -> 0.03 robot.
* R4 declared expectation (agent ``expect``) -> 0.7 / 0.3, source exp, only for a key not observed at
  this event and not robot-owned.
* R5 geometric veto: Observe 1 for on / in whose geometry (depth path, both objects detected now)
  shows the object is not resting flat (bg_est2 ``flat_veto``: d_xy > support radius + 0.5 cm or
  dz > the pair's flat dz + 1 cm) -> P 0.5, source conflict, S4 "may be resting on the rim".
* R6 hidden state: a state predicate (cooked, toggled_on) of an object believed inside a closed
  container cannot be observed: the answer is treated as ? (kept, source mem).
* R7 relations change only when the robot touches them (``touch_rule``, can be turned off): an Observe
  that contradicts a decided belief while no event touched the subject or the target since the last
  observation is NOT applied; the key is re-asked at the next event / keyframe and applied only when
  that answer agrees (2 agreeing answers).
* Marks: only tracks detected in the current image are marked (the generator draws them).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from capx.belief.estimator2 import Est2Config, flat_veto

from .graph import BeliefGraph, decided
from .vocab import ROBOT, STATE_PREDS, TABLE, EdgeKey, statement_text

P_OBS = (0.9, 0.1)
P_ROBOT = (0.97, 0.03)
P_EXP = (0.7, 0.3)


@dataclass
class VerifierConfig:
    touch_rule: bool = True
    recheck_agree: int = 2  # answers that must agree before a no-touch change is applied
    veto: bool = True
    veto_xy_margin: float = 0.005
    veto_dz_margin: float = 0.010
    # "<object category>/<support category>" -> center dz resting flat (m). Default: bg_est2's measured
    # LIBERO-Mem table (bowl/plate 0.028, bottle/plate 0.063). Other pairs use the dz of their first
    # accepted Observe 1. Measure again per benchmark (design: 기준값은 실측 분포에서).
    flat_dz: dict[str, float] = field(default_factory=lambda: dict(Est2Config().flat_dz))


@dataclass
class Decision:
    key: str
    rule: str
    value: str | None  # the evidence ("1" / "0" / "?" / "robot" / "exp")
    before: float | None
    after: float | None
    source: str | None
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Verifier:
    def __init__(self, graph: BeliefGraph, cfg: VerifierConfig | None = None) -> None:
        self.g = graph
        self.cfg = cfg or VerifierConfig()
        self.pending: dict[EdgeKey, dict[str, Any]] = {}  # R7: key -> {"value", "t", "count"}
        self.flat_measured: dict[str, float] = {}
        self.s4: list[str] = []  # S4 messages of the current event
        self.decisions: list[Decision] = []

    def begin(self) -> None:
        self.s4 = []
        self.decisions = []

    def _log(self, key: EdgeKey, rule: str, value: str | None, before: float | None, why: str = "") -> None:
        e = self.g.get(key)
        self.decisions.append(Decision(statement_text(key), rule, value, before,
                                       None if e is None else e.p, None if e is None else e.source, why))

    # --- R3 robot -----------------------------------------------------------------------------
    def robot(self, key: EdgeKey, value: bool, t: int, why: str = "") -> None:
        before = self.g.p(key)
        self.g.set(key, P_ROBOT[0] if value else P_ROBOT[1], "robot", t, observed=True)
        if before is None or round(before, 3) != self.g.p(key) or why:
            self._log(key, "R3", "robot", before, why)

    def held_off_supports(self, alias: str, t: int) -> None:
        for k in list(self.g.edges):
            if k[1] == alias and k[0] in ("on", "in", "at") and self.g.edges[k].p > P_ROBOT[1]:
                self.robot(k, False, t, "held and lifted objects rest on nothing")
                self.pending.pop(k, None)

    # --- R4 expectation -----------------------------------------------------------------------
    def expect(self, key: EdgeKey, value: bool, t: int, observed_now: set[EdgeKey]) -> None:
        e = self.g.get(key)
        if key in observed_now or key[0] in ROBOT or (e is not None and e.source == "robot"):
            self._log(key, "R4-skip", "exp", None if e is None else e.p, "observed now or robot-owned")
            return
        before = None if e is None else e.p
        self.g.set(key, P_EXP[0] if value else P_EXP[1], "exp", t)
        self._log(key, "R4", "exp", before)

    # --- R1 / R2 / R5 / R6 / R7 observe ---------------------------------------------------------
    def observe(self, values: dict[EdgeKey, str], t: int, touched: Iterable[str] = (), geometry: dict[EdgeKey, dict] | None = None,
                hidden: Iterable[str] = (), categories: dict[str, str] | None = None) -> set[EdgeKey]:
        """Apply Observe answers; returns the keys whose answer was accepted as an observation."""
        touched, hidden = set(touched), set(hidden)
        geometry = geometry or {}
        accepted: set[EdgeKey] = set()
        for key, v in values.items():
            e = self.g.get(key)
            before = None if e is None else e.p
            if key[0] in ROBOT or (e is not None and e.source == "robot" and key[0] in ("on", "in")):
                want = None if v == "?" else (v == "1")
                if want is not None and e is not None and decided(e.p) is not None and decided(e.p) != want:
                    self._log(key, "R3-conflict", v, before, "robot state wins over Observe")
                continue
            if key[0] in STATE_PREDS and key[1] in hidden:
                v = "?"
                self._log(key, "R6", "?", before, "inside a closed container: not observable")
            if v == "?":
                if e is not None and e.source == "obs":
                    e.source = "mem"
                self._log(key, "R2", "?", before)
                continue
            val = v == "1"
            # R7: no-touch change needs agreeing re-observations
            if self.cfg.touch_rule and e is not None and decided(e.p) is not None and decided(e.p) != val \
                    and key[1] not in touched and (key[2] is None or key[2] not in touched) and e.source != "exp":
                pend = self.pending.get(key)
                count = pend["count"] + 1 if pend is not None and pend["value"] == v else 1
                if count < self.cfg.recheck_agree:
                    self.pending[key] = {"value": v, "t": t, "count": count}
                    self._log(key, "R7-hold", v, before, "changed with no robot contact: re-check at the next keyframe")
                    accepted.add(key)  # it was looked at (not aged to mem)
                    continue
                self.pending.pop(key, None)
                self._log(key, "R7-confirm", v, before, f"{count} agreeing answers")
            else:
                if key in self.pending:
                    self._log(key, "R7-clear", v, before, "re-check agrees with the belief" if
                              (decided(before) == val) else "touched since")
                self.pending.pop(key, None)
            # R5 geometric veto
            g = geometry.get(key)
            if val and self.cfg.veto and key[0] in ("on", "in") and key[2] not in (None, TABLE) and g and g.get("mode") == "depth":
                why = self._veto(key, g, categories or {})
                if why:
                    self.g.set(key, 0.5, "conflict", t, observed=True)
                    if why == ["outside"]:
                        self.s4.append(f"{key[1]} may not be inside {key[2]} (its center is outside {key[2]}'s box)")
                    else:
                        self.s4.append(f"{key[1]} may be resting on the rim of {key[2]} (not flat: {', '.join(why)})")
                    self._log(key, "R5", v, before, f"geometry veto {why} {g}")
                    accepted.add(key)
                    continue
            self.g.set(key, P_OBS[0] if val else P_OBS[1], "obs", t, observed=True)
            self._log(key, "R1", v, before)
            accepted.add(key)
        return accepted

    def _veto(self, key: EdgeKey, g: dict[str, Any], categories: dict[str, str]) -> list[str]:
        if "d_xy" not in g or g.get("inside") is True:
            return []
        if g.get("inside") is False:  # container path: the center is outside the container box
            return ["outside"]
        pair = f"{categories.get(key[1], key[1])}/{categories.get(key[2], key[2])}"
        flat = self.cfg.flat_dz.get(pair, self.flat_measured.get(pair))
        why = flat_veto(g["d_xy"], g["r"], g.get("dz"), flat, self.cfg.veto_xy_margin, self.cfg.veto_dz_margin)
        if not why and flat is None and g.get("dz") is not None:
            self.flat_measured[pair] = float(g["dz"])
        return why

    def recheck_keys(self) -> list[EdgeKey]:
        return list(self.pending)
