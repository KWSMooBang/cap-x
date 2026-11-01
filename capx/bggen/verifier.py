"""Rule-based verifier: the only code that changes P (design: diff 제안과 검증). The agent never
writes a confidence.

Rules (each decision is logged with its rule id):

* R1 Observe 1 / 0 -> P 0.9 / 0.1, source obs.
* R2 Observe ? -> previous value kept (source becomes mem).
* R3 robot state -> 0.97 / 0.03, source robot (inhand, handempty, lifted); robot wins over Observe (logged
  as a conflict). A held object is on / in nothing: its location edges -> 0.03 robot for as long as it is held.
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
* R8 process model: cooked(X) after in(X, C) and toggled_on(C) held for ``cook_s`` -> 0.7 mem (generator).
* R9 release geometry fills ``?`` (bggen-fix3 A1): at a release the generator always runs the geometry tool for
  the released object and the supports / containers marked near it (the agent need not ask for it). If the VDM
  answered ``?`` (or was not asked) and the geometry is positive for the best candidate -> 0.7 / source exp (an
  expectation, not an observation: goal steps still need an observation). Why: t08 d1 kept in(cheese, basket) at
  0.03 because every answer was ``?`` and nobody asked the geometry.
* R10 'in' conflict (bggen-fix3 A3): an Observe 0 for in(X, C) while the geometry says X is inside C is a
  conflict, not an observation: P 0.5 / conflict and an immediate re-check (another camera with ``look``; mode A
  also a hindsight frame of the same quiet interval); a re-check answer 0 lowers it (0.1 obs), 1 raises it (0.9
  obs), ``?`` keeps the key pending and it is asked again at the next event (whose 0 then lowers it). Why: the
  front camera sees a basket's wall, not its inside (t08 d2 / d3: one VDM 0 overruled geometry that put the
  cheese in the basket).
  bggen-fix4 A: the geometry need not be computed at the same event: an 'in' answered 0 without positive geometry now
  (and not measured apart in 3D now) gets the pair's last positive geometry (R9 release geometry, an earlier
  observe), valid while neither object was grasped since and neither track moved (``geometry.GeoMemory``; the
  validity record -- grasps, movement per object -- is in the decision and in the trace's ``geometry_memory``). Why:
  t08 d1 / d2 put the cheese in the basket by R9 (0.7) at the release and lowered it with the next keyframe's VDM 0
  (R1), whose event computed no geometry. 'on' is not included: a front camera sees a plate's top (no wall hides
  it), so an 'on' answered 0 is ordinary evidence (R7 / R5 already guard it).
  bggen-fix4 B: the geometry of a release (R9, R10) is that of the first settled frame after it (mode A) -- the
  released object detected with the hand gone or the object still -- else of the release frame with its height
  vetoing nothing (``frame`` / ``t_geo`` in the geometry say which; t08 d3: 13.3 cm above the basket, still falling).
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
ROBOT_OWNED = (*ROBOT, "lifted")  # predicates only the robot state decides
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
        self.geo_conflict: dict[EdgeKey, int] = {}  # R10: key -> frame of the conflicting Observe 0
        self.new_conflicts: list[EdgeKey] = []  # R10 holds of the current event (the generator re-checks them now)

    def begin(self) -> None:
        self.s4 = []
        self.decisions = []
        self.new_conflicts = []

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
        if key in observed_now or key[0] in ROBOT_OWNED or (e is not None and e.source == "robot"):
            self._log(key, "R4-skip", "exp", None if e is None else e.p, "observed now or robot-owned")
            return
        before = None if e is None else e.p
        self.g.set(key, P_EXP[0] if value else P_EXP[1], "exp", t)
        self._log(key, "R4", "exp", before)

    # --- R1 / R2 / R5 / R6 / R7 observe ---------------------------------------------------------
    def observe(self, values: dict[EdgeKey, str], t: int, touched: Iterable[str] = (), geometry: dict[EdgeKey, dict] | None = None,
                hidden: Iterable[str] = (), categories: dict[str, str] | None = None,
                held: Iterable[str] = ()) -> set[EdgeKey]:
        """Apply Observe answers; returns the keys whose answer was accepted as an observation. ``held``:
        objects in a gripper now -- their on / in / at relations stay robot-owned (0.03) while held."""
        touched, hidden, held = set(touched), set(hidden), set(held)
        geometry = geometry or {}
        accepted: set[EdgeKey] = set()
        for key, v in values.items():
            e = self.g.get(key)
            before = None if e is None else e.p
            if key[0] in ("on", "in", "at") and key[1] in held:
                if v == "1":
                    self._log(key, "R3-conflict", v, before, "held object: robot state wins over Observe")
                self.robot(key, False, t, "held objects rest on nothing")
                continue
            if key[0] in ROBOT_OWNED or (e is not None and e.source == "robot" and key[0] in ("on", "in")):
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
            # R10: 'in' answered 0 while the geometry puts the object inside -> conflict, re-check before lowering
            if key[0] == "in" and key in self.geo_conflict:
                self.geo_conflict.pop(key)
                if not val:
                    self.pending.pop(key, None)
                    self.g.set(key, P_OBS[1], "obs", t, observed=True)
                    self._log(key, "R10-confirm", v, before, "re-asked after a geometry conflict: 0 again")
                    accepted.add(key)
                    continue
                self._log(key, "R10-clear", v, before, "re-asked after a geometry conflict: 1")
            elif key[0] == "in" and not val and self.geo_positive(key, geometry.get(key), categories or {}):
                self.geo_conflict[key] = t
                self.new_conflicts.append(key)
                self.pending.pop(key, None)
                self.g.set(key, 0.5, "conflict", t, observed=True)
                self.s4.append(f"{key[1]} may be inside {key[2]} (geometry) though the image answer was no: re-checking")
                self._log(key, "R10-hold", v, before, f"Observe 0 vs geometry inside {geometry.get(key)}")
                accepted.add(key)
                continue
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
        return list(dict.fromkeys([*self.pending, *self.geo_conflict]))

    # --- R9 / R10 geometry ------------------------------------------------------------------------
    def geo_positive(self, key: EdgeKey, g: dict[str, Any] | None, categories: dict[str, str]) -> bool:
        """The geometry supports the relation: the container check (depth, ``inside``), the 2D evidence
        (``positive``), or for a support on the depth path the center within the support radius and resting flat."""
        if not g or key[0] not in ("on", "in"):
            return False
        if g.get("positive") is not None:
            return bool(g["positive"])
        if g.get("mode") == "depth" and "d_xy" in g and g.get("inside") is None:
            if g["d_xy"] > g["r"] + self.cfg.veto_xy_margin:
                return False
            pair = f"{categories.get(key[1], key[1])}/{categories.get(key[2], key[2])}"
            flat = self.cfg.flat_dz.get(pair, self.flat_measured.get(pair))
            return not flat_veto(g["d_xy"], g["r"], g.get("dz"), flat, self.cfg.veto_xy_margin, self.cfg.veto_dz_margin)
        return False

    def geo_fill(self, key: EdgeKey, g: dict[str, Any] | None, t: int, categories: dict[str, str],
                 held: Iterable[str] = ()) -> bool:
        """R9: a release-geometry candidate the VDM did not decide (``?`` / not asked): positive geometry -> 0.7 exp."""
        e = self.g.get(key)
        if key[1] in set(held) or (e is not None and e.source == "robot" and e.p >= P_ROBOT[0]):
            return False
        if not self.geo_positive(key, g, categories):
            return False
        before = None if e is None else e.p
        self.g.set(key, P_EXP[0], "exp", t)
        self._log(key, "R9", "geometry", before, f"release geometry, Observe undecided {g}")
        return True

    def recheck(self, key: EdgeKey, v: str, t: int, how: str) -> bool:
        """R10 re-check of a conflict at the same event (``look`` / ``hindsight``). Returns True when decided."""
        if key not in self.geo_conflict:
            return False
        e = self.g.get(key)
        before = None if e is None else e.p
        if v not in ("0", "1"):
            self._log(key, "R10-pending", v, before, f"{how} re-check undecided: asked again at the next event")
            return False
        self.geo_conflict.pop(key)
        self.g.set(key, P_OBS[0] if v == "1" else P_OBS[1], "obs", t, observed=True)
        self._log(key, "R10-confirm" if v == "0" else "R10-resolve", v, before, f"{how} re-check")
        return True
