"""Estimated Belief Graph (``bg_est``): built only from what the robot can use.

Inputs: camera detections (``perception.py``), the robot's end-effector pose and gripper opening,
the API calls of the generated code (``hooks.py``), the verification VDM (``verify.py``) and the task
language (``goal.py``). No simulator object pose, BDDL predicate or task-progress tracker is read.

Per turn (``end_turn``):
 1. Hook records -> expected effects (source=expected, P 0.7/0.3) and intra-turn location snapshots:
    close_gripper that did not close fully -> the tracked object nearest the EEF is held; it counts
    as picked up (``in gripper``) once the EEF rises ``lift_min`` while holding; open_gripper -> the
    held object is put on the highest tracked support under the EEF (or the table); the gripper
    closing fully while holding -> the object was dropped (S1 violation).
 2. Perception -> data association (Hungarian on 3D distance within ``gate_dist``, held objects
    predicted at the EEF; leftover detections are re-matched to leftover tracks of the same
    category within ``rematch_dist`` before new tracks are created). Unmatched tracks keep their
    last position (source=mem, no decay).
 3. Geometry -> On / In with P scaled by the margin to the thresholds; the gripper state -> inhand.
 4. Verification VDM on this turn's expectations and unsure goal predicates.
 5. Fusion per edge: geometry/gripper (obs) > VDM (obs) > expected > memory > prior. Geometry and VDM
    disagreeing is logged; the one further from P=0.5 wins (geometry on a tie).
 6. Events (location changes, intra-turn ones from the hooks first), progress, S1 / S4 inputs.

Progress counts a goal step when all its predicates hold in a snapshot after they did not hold
in an earlier snapshot since the previous count (like LIBERO-Mem's tracker, which needs the
subgoal false before it can count again). Snapshots are taken after every hooked call and at the
end of the turn, so a pick-and-place-back inside one turn counts once even though the turn-end
state is unchanged.
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .goal import GoalResult, goal_preds, parse_goal_llm
from .graph import BeliefGraph, EdgeKey, Event, format_location
from .hooks import CallRecord
from .perception import (
    VOCABULARY,
    Category,
    Detection,
    PerceptionSource,
    categories_from_instruction,
    project_to_pixel,
    proprio_from_obs,
)
from .progress import Pred, Progress, Step, format_pred, format_step, progress_line
from .serialize import serialize
from .signals import Signal, Uncertain, Violation
from .verify import Hypothesis, VdmVerifier, encode_png, pred_text

TABLE = "table"
UNKNOWN_LOCATION = "location unknown"
EST_TITLE = (
    "Belief graph (turn {turn}), ESTIMATED from the cameras, the robot state and the API calls: positions and "
    "relations may be wrong. Relations show belief P and source (obs: camera geometry or gripper, vdm: visual "
    "check, exp: expected from an API call, mem: not re-observed):"
)
EST_OBJECTS_PREFIX = "Objects (estimated xyz in robot base frame, m): "
Snapshot = dict[str, frozenset]  # alias -> {(predicate, target)} of true location relations


@dataclass
class EstConfig:
    gate_dist: float = 0.10  # m: a detection farther than this from a track's prediction is not matched in pass 1
    rematch_dist: float = 0.50  # m: pass 2 re-matches leftovers of the same category within this
    dedup_dist: float = 0.01  # m: detections of different categories this close are one object
    # gripper (0 closed .. 1 open) above this after close_gripper -> holding. LIBERO-Mem: closing on
    # nothing reads ~0.015; holding a bowl by the rim 0.11 after the close and ~0.044 once lifted.
    grasp_threshold: float = 0.025
    grasp_radius: float = 0.10  # m: fingertip centre to the grasped object's center
    lift_min: float = 0.03  # m: EEF rise while holding that counts as picked up
    held_offset: tuple[float, float, float] = (0.0, 0.0, -0.02)  # held object center - fingertip, if unknown
    support_radius_min: float = 0.03  # m
    # m: On(x, y) needs z_x - z_y (centres) in (on_dz_min, on_dz_max). A bowl resting on a LIBERO plate
    # is ~0.018 above it; a bowl tilted onto the plate rim (not On for LIBERO) sits level with it.
    on_dz_min: float = 0.0
    on_dz_max: float = 0.10
    table_tol: float = 0.03  # m: an object's bottom this close to the table height can be on the table
    margin_scale: float = 0.02  # m: margin to a threshold that gives full confidence (P 0.97 / 0.03)
    z_margin_scale: float = 0.01  # m: the same for the vertical On / In thresholds
    p_vdm: float = 0.9  # P of a VDM true (1 - p_vdm for false)
    stale_turns: int = 2  # a memory not re-observed for this many turns counts as uncertain (S4)
    uncertain_low: float = 0.3
    uncertain_high: float = 0.7
    max_hypotheses: int = 8
    max_categories: int = 5
    init_passes: int = 2  # perception passes merged at turn 0 (static scene)
    max_uncertain_shown: int = 4
    max_violations_shown: int = 3


@dataclass
class Track:
    alias: str
    category: Category
    pos: np.ndarray
    size: np.ndarray
    last_seen_turn: int | None


@dataclass
class EstEvent(Event):
    source: str = "obs"  # obs | exp | vdm

    @property
    def text(self) -> str:
        return f"t{self.turn} {self.subject} {self.before} -> {self.after} ({self.source})"


@dataclass
class EstTurn:
    """What one update did (logged; only ``render`` output goes to the prompt)."""

    turn: int
    progress: Progress
    events: list[EstEvent] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    uncertain: list[Uncertain] = field(default_factory=list)
    suppress_completion: bool = False
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    expectations: list[dict[str, Any]] = field(default_factory=list)
    detections: int = 0
    new_tracks: list[str] = field(default_factory=list)
    unmatched_tracks: list[str] = field(default_factory=list)
    rematched: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    verify: dict[str, Any] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    snapshots: int = 0
    pending: list[int] = field(default_factory=list)  # counted steps (indices) not yet confirmed
    confirmed_by: list[str | None] = field(default_factory=list)  # per counted step: obs | vdm | later_step
    rolled_back: list[str] = field(default_factory=list)
    perception: str = ""

    def to_log(self) -> dict[str, Any]:
        p = self.progress
        return {
            "turn": self.turn,
            "progress": {"done": len(p.done), "remaining": p.remaining, "known": p.known, "overshot": p.overshot,
                         "done_steps": [format_step(s) for s in p.done]},
            "events": [e.text for e in self.events],
            "violations": [v.to_dict() for v in self.violations],
            "uncertain": [u.to_dict() for u in self.uncertain],
            "suppress_completion": self.suppress_completion,
            "conflicts": self.conflicts,
            "expectations": self.expectations,
            "detections": self.detections,
            "new_tracks": self.new_tracks,
            "unmatched_tracks": self.unmatched_tracks,
            "rematched": self.rematched,
            "hypotheses": self.hypotheses,
            "verify": self.verify,
            "records": self.records,
            "snapshots": self.snapshots,
            "pending": self.pending,
            "confirmed_by": self.confirmed_by,
            "rolled_back": self.rolled_back,
            "perception": self.perception,
        }


def _holds_in(snapshot: Snapshot, pred: Pred) -> bool:
    if pred[0] == "inhand":
        return ("inhand", None) in snapshot.get(pred[1], frozenset())
    return (pred[0], pred[2]) in snapshot.get(pred[1], frozenset())


class StepCounter:
    """Counts goal steps over location snapshots (see module docstring).

    ``was_false`` is kept per step for every goal step from the first snapshot on, so a step that
    turns true in the same snapshot as (or right after) the previous count is still counted; a
    counted step must turn false again before it can count again (repeats)."""

    def __init__(self, branches: list[list[Step]]) -> None:
        self.branches = branches
        self.steps: list[Step] = []
        for b in branches:
            for s in b:
                if s not in self.steps:
                    self.steps.append(s)
        self.done: list[Step] = []
        self.was_false: dict[Step, bool] = {}
        self.overshot = False

    def progress(self) -> Progress:
        return Progress.from_completed(self.branches, self.done, self.overshot)

    def checkpoint(self) -> tuple[list[Step], dict[Step, bool], bool]:
        return list(self.done), dict(self.was_false), self.overshot

    def restore(self, ckpt: tuple[list[Step], dict[Step, bool], bool]) -> None:
        self.done, self.was_false, self.overshot = list(ckpt[0]), dict(ckpt[1]), ckpt[2]

    def feed(self, snapshot: Snapshot, final: bool = False, count: bool = True) -> list[Step]:
        """Count the steps this snapshot completes (several in a row if they all hold). ``count=False``
        only records which steps are false (a snapshot whose counts were withdrawn)."""
        holds = {s: all(_holds_in(snapshot, p) for p in s) for s in self.steps}
        counted: list[Step] = []
        while count and not self.progress().complete:
            hit = next((s for s in self.progress().next_steps() if holds[s] and self.was_false.get(s)), None)
            if hit is None:
                break
            self.done.append(hit)
            self.was_false[hit] = False
            counted.append(hit)
        for s in self.steps:
            if not holds[s]:
                self.was_false[s] = True
        if final and self.done and self.progress().complete and not holds[self.done[-1]]:
            self.overshot = True  # like LIBERO-Mem: the finished goal was undone
        return counted


def _pred_key(p: Pred) -> EdgeKey:
    return (p[0], p[1], p[2] if len(p) > 2 else None)


def _key_text(key: EdgeKey, holds: bool = True) -> str:
    pred, subj, target = key
    if pred == "inhand":
        return f"{subj} {'in gripper' if holds else 'not in gripper'}"
    return f"{subj} {pred if holds else 'not ' + pred} {target}"


class BeliefEstimator:
    def __init__(
        self,
        perception: PerceptionSource,
        config: EstConfig | None = None,
        goal_query_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
        verifier: VdmVerifier | None = None,
        vocabulary: Sequence[Category] = VOCABULARY,
        categories: Sequence[Category] | None = None,
    ) -> None:
        self.perception = perception
        self.cfg = config or EstConfig()
        self.goal_query_fn = goal_query_fn
        self.verifier = verifier
        self.vocabulary = vocabulary
        self.fixed_categories = list(categories) if categories is not None else None
        self.graph = BeliefGraph()
        self.tracks: dict[str, Track] = {}
        self.origin: dict[EdgeKey, str] = {}  # how each edge's current P was set: geo|grip|vdm|exp|mem|prior
        self.categories: list[Category] = []
        self.instruction = ""
        self.goal: GoalResult = GoalResult(None, "none")
        self.counter = StepCounter([])
        self.loc: Snapshot = {}
        self.gripper_cmd = "open"
        self.held: str | None = None
        self.held_lifted = False
        self.grasp_z = 0.0
        self.held_off = np.asarray(self.cfg.held_offset, dtype=np.float64)  # measured at the grasp
        self.violations: list[Violation] = []
        self.last_image: str | None = None
        self.turn = 0
        self.table_z: float | None = None  # table height: lowest object bottom at turn 0
        self.pending: set[int] = set()  # indices into counter.done not yet confirmed by observation
        self.confirmed_by: dict[int, str] = {}  # counted step index -> obs | vdm | later_step

    # --- tracks ---------------------------------------------------------------------------------
    def _new_alias(self, cat: Category) -> str:
        used = {t.alias for t in self.tracks.values() if t.category.key == cat.key}
        for letter in string.ascii_uppercase:
            if f"{cat.key}_{letter}" not in used:
                return f"{cat.key}_{letter}"
        return f"{cat.key}_{len(used)}"

    def _add_track(self, det: Detection, turn: int) -> Track:
        cat = next(c for c in self.categories if c.key == det.category)
        t = Track(self._new_alias(cat), cat, np.asarray(det.center, float), np.asarray(det.size, float), turn)
        self.tracks[t.alias] = t
        node = self.graph.add_node(t.alias)
        node.pos, node.last_seen_turn = tuple(float(v) for v in t.pos), turn
        return t

    def _dedup(self, dets: list[Detection]) -> list[Detection]:
        out: list[Detection] = []
        for d in sorted(dets, key=lambda d: -d.score):
            if any(o.category != d.category and np.linalg.norm(o.center - d.center) < self.cfg.dedup_dist for o in out):
                continue
            out.append(d)
        return out

    def _predicted(self, t: Track, eef: np.ndarray | None) -> np.ndarray:
        if t.alias == self.held and eef is not None and self.gripper_cmd == "closed":
            return eef + self.held_off
        return t.pos

    def _associate(self, dets: list[Detection], eef: np.ndarray | None, turn: int) -> tuple[set[str], list[str], list[str], list[str]]:
        """Match detections to tracks; returns (seen aliases, new aliases, unmatched aliases, re-matched aliases)."""
        from scipy.optimize import linear_sum_assignment

        seen: set[str] = set()
        new: list[str] = []
        rematched: list[str] = []
        for cat in self.categories:
            tracks = [t for t in self.tracks.values() if t.category.key == cat.key]
            cat_dets = [d for d in dets if d.category == cat.key]
            free_t, free_d = list(range(len(tracks))), list(range(len(cat_dets)))
            # Pass 1: one global assignment over every track and detection of the category (distances
            # to the predictions, the held object at the EEF) in which pairs beyond gate_dist cannot be
            # assigned, so a long crossing pair can never push out a match inside the gate.
            # Pass 2: leftovers only, within rematch_dist.
            for gate, label in ((self.cfg.gate_dist, "gate"), (self.cfg.rematch_dist, "rematch")):
                if not free_t or not free_d:
                    break
                cost = np.array([[np.linalg.norm(self._predicted(tracks[i], eef) - cat_dets[j].center)
                                  for j in free_d] for i in free_t])
                rows, cols = linear_sum_assignment(np.where(cost <= gate, cost, 1e6))
                matched_t, matched_d = set(), set()
                for r, c in zip(rows, cols, strict=True):
                    if cost[r, c] > gate:
                        continue
                    t, d = tracks[free_t[r]], cat_dets[free_d[c]]
                    t.pos, t.size, t.last_seen_turn = np.asarray(d.center, float), np.asarray(d.size, float), turn
                    seen.add(t.alias)
                    if label == "rematch":
                        rematched.append(t.alias)
                    matched_t.add(free_t[r])
                    matched_d.add(free_d[c])
                free_t = [i for i in free_t if i not in matched_t]
                free_d = [j for j in free_d if j not in matched_d]
            for j in free_d:
                t = self._add_track(cat_dets[j], turn)
                seen.add(t.alias)
                new.append(t.alias)
        for t in self.tracks.values():
            node = self.graph.add_node(t.alias)
            node.pos = tuple(float(v) for v in t.pos)
            if t.alias in seen:
                node.last_seen_turn = turn
        unmatched = sorted(a for a in self.tracks if a not in seen)
        return seen, new, unmatched, rematched

    def _initial_tracks(self, dets: list[Detection]) -> None:
        """Turn 0: aliases per category in order of the robot base frame y (robot's right first),
        ties by x: bowl_A, bowl_B, ..."""
        for d in sorted(dets, key=lambda d: (d.category, round(float(d.center[1]), 3), float(d.center[0]))):
            self._add_track(d, 0)

    # --- geometry -------------------------------------------------------------------------------
    def _relation(self, x: Track, y: Track, y_seen: bool) -> tuple[EdgeKey, float]:
        cfg = self.cfg
        d_xy = float(np.linalg.norm(x.pos[:2] - y.pos[:2]))
        dz = float(x.pos[2] - y.pos[2])
        r = max(0.5 * float(max(y.size[0], y.size[1])), cfg.support_radius_min)
        if y.category.container:
            key: EdgeKey = ("in", x.alias, y.alias)
            lo, hi = -0.5 * float(y.size[2]) - 0.02, 0.5 * float(y.size[2]) + 0.03
        else:
            key = ("on", x.alias, y.alias)
            lo, hi = cfg.on_dz_min, cfg.on_dz_max
        zs = cfg.z_margin_scale
        m = min((r - d_xy) / max(0.5 * r, cfg.margin_scale), (dz - lo) / zs, (hi - dz) / zs)
        c = min(abs(m), 1.0) * (1.0 if y_seen else 0.5)  # support position from memory: less sure
        return key, 0.5 + 0.47 * c if m > 0 else 0.5 - 0.47 * c

    def _support_below(self, held: str, eef: np.ndarray) -> Track | None:
        best = None
        for t in self.tracks.values():
            if t.alias == held:
                continue
            r = max(0.5 * float(max(t.size[0], t.size[1])), self.cfg.support_radius_min)
            if np.linalg.norm(t.pos[:2] - eef[:2]) < r and t.pos[2] < eef[2] and (best is None or t.pos[2] > best.pos[2]):
                best = t
        return best

    def _nearest(self, eef: np.ndarray, radius: float, only: set[str] | None = None) -> Track | None:
        """The tracked object the fingers are on: within ``radius`` of the fingertip centre, the highest
        one (a rim grasp is about as close to the plate under the bowl as to the bowl)."""
        cands = [t for t in self.tracks.values()
                 if (only is None or t.alias in only) and float(np.linalg.norm(t.pos - eef)) <= radius]
        return max(cands, key=lambda t: float(t.pos[2])) if cands else None

    # --- graph helpers --------------------------------------------------------------------------
    def _location_edges(self, alias: str) -> list[EdgeKey]:
        return [k for k in self.graph.edges if k[1] == alias and k[0] in ("on", "in", "inhand")]

    def _snapshot(self) -> Snapshot:
        return {a: frozenset((p, t) for p, t in self.graph.holding(a)) for a in self.tracks}

    def location_text(self, alias: str, snap: Snapshot | None = None) -> str:
        rels = (snap or self.loc).get(alias, frozenset())
        return format_location(rels) if rels else UNKNOWN_LOCATION

    # --- turn 0 ---------------------------------------------------------------------------------
    def start(
        self,
        obs: dict[str, Any],
        instruction: str,
        goal_fn: Callable[[BeliefEstimator], GoalResult] | None = None,
    ) -> EstTurn:
        """First perception and goal parsing. ``goal_fn`` replaces the LLM goal (``goal_source: bddl``)."""
        self.instruction = instruction
        self.categories = self.fixed_categories or categories_from_instruction(
            instruction, self.cfg.max_categories, self.vocabulary)
        proprio = proprio_from_obs(obs)
        eef = proprio[0] if proprio else None
        if proprio is not None and proprio[1] <= self.cfg.grasp_threshold:
            self.gripper_cmd = "closed"
        dets = self._dedup(self.perception.detect(obs, self.categories))
        self._initial_tracks(dets)
        if dets:
            self.table_z = min(float(d.center[2] - 0.5 * d.size[2]) for d in dets)
        for _ in range(self.cfg.init_passes - 1):  # merge further passes (fills in missed objects)
            self._associate(self._dedup(self.perception.detect(obs, self.categories)), eef, 0)
        self._fuse(0, self._observe_relations(0, set(self.tracks), proprio), {}, {})
        self.loc = self._snapshot()
        if goal_fn is not None:
            self.goal = goal_fn(self)
        elif self.goal_query_fn is not None:
            self.goal = parse_goal_llm(self.goal_query_fn, instruction,
                                       self.objects_text(with_p=False, prefix="- ", sep="\n- "), sorted(self.tracks))
        self.counter = StepCounter(self.goal.branches or [])
        if self.goal.branches:
            self.counter.feed(self.loc)
        self.last_image = self._image(obs)
        self.turn = 0
        return EstTurn(0, self.counter.progress(), detections=len(dets), new_tracks=sorted(self.tracks),
                       perception=self.perception_name)

    def _image(self, obs: dict[str, Any]) -> str | None:
        if self.verifier is None:
            return None
        try:
            return encode_png(obs["agentview"]["images"]["rgb"])
        except Exception:
            return None

    # --- per turn -------------------------------------------------------------------------------
    def _apply_records(self, turn: int, records: Sequence[CallRecord]) -> tuple[dict[EdgeKey, tuple[bool, str]], list[Snapshot], list[EstEvent], list[Violation]]:
        expectations: dict[EdgeKey, tuple[bool, str]] = {}
        snapshots: list[Snapshot] = []
        events: list[EstEvent] = []
        violations: list[Violation] = []
        loc = dict(self.loc)
        thr = self.cfg.grasp_threshold

        def move(alias: str, new: frozenset) -> None:
            before = format_location(loc[alias]) if loc.get(alias) else UNKNOWN_LOCATION
            after = format_location(new) if new else UNKNOWN_LOCATION
            loc[alias] = new
            if before != after:
                events.append(EstEvent(turn, alias, before, after, "exp"))

        for rec in records:
            eef, g = rec.eef_after, rec.gripper_after
            if eef is None:
                continue
            if rec.fn == "close_gripper":
                self.gripper_cmd = "closed"
                near = self._nearest(eef, self.cfg.grasp_radius)
                cand = near if g is not None and g > thr else None
                self.held, self.held_lifted, self.grasp_z = (cand.alias if cand else None), False, float(eef[2])
                if cand is not None:
                    self.held_off = cand.pos - eef
                    expectations[("inhand", cand.alias, None)] = (True, rec.desc)
                elif g is not None and near is not None:  # closed fully next to an object: grasp missed
                    violations.append(Violation(
                        turn, rec.desc, f"{near.alias} in gripper (nearest object to the gripper)",
                        f"gripper closed fully ({g:.2f}): nothing grasped", ("inhand", near.alias, None), True))
            elif rec.fn == "open_gripper":
                self.gripper_cmd = "open"
                if self.held is not None:
                    t = self.tracks[self.held]
                    at = eef + self.held_off  # where the released object is
                    support = self._support_below(t.alias, at)
                    expectations[("inhand", t.alias, None)] = (False, rec.desc)
                    if support is not None:
                        pred = "in" if support.category.container else "on"
                        expectations[(pred, t.alias, support.alias)] = (True, rec.desc)
                        t.pos = np.array([at[0], at[1], support.pos[2] + 0.02])
                    else:
                        expectations[("on", t.alias, TABLE)] = (True, rec.desc)
                        t.pos = np.array([at[0], at[1], self.grasp_z + self.held_off[2]])
                    if self.held_lifted:
                        if support is None:
                            move(t.alias, frozenset({("on", TABLE)}))
                        else:
                            move(t.alias, frozenset({("in" if support.category.container else "on", support.alias)}))
                self.held, self.held_lifted = None, False
            elif self.held is not None:  # a motion while holding
                t = self.tracks[self.held]
                if g is not None and g <= thr:
                    violations.append(Violation(turn, rec.desc, f"{t.alias} in gripper",
                                                f"gripper closed fully ({g:.2f}): object dropped",
                                                ("inhand", t.alias, None), True))
                    t.pos = eef + self.held_off
                    if self.held_lifted:
                        move(t.alias, frozenset())
                    expectations.pop(("inhand", t.alias, None), None)
                    self.held, self.held_lifted = None, False
                else:
                    t.pos = eef + self.held_off
                    if not self.held_lifted and float(eef[2]) - self.grasp_z >= self.cfg.lift_min:
                        self.held_lifted = True
                        for p, tgt in loc.get(t.alias, frozenset()):
                            if p != "inhand":
                                expectations[(p, t.alias, tgt)] = (False, rec.desc)
                        move(t.alias, frozenset({("inhand", None)}))
            snapshots.append(dict(loc))
        return expectations, snapshots, events, violations

    def _observe_relations(self, turn: int, seen: set[str], proprio: tuple[np.ndarray, float] | None) -> dict[EdgeKey, tuple[float, str]]:
        """Geometry (On / In) for seen subjects and the gripper state (inhand); returns key -> (P, origin)."""
        obs: dict[EdgeKey, tuple[float, str]] = {}
        holder: str | None = None
        if proprio is not None:
            eef, g = proprio
            if self.gripper_cmd == "closed" and g > self.cfg.grasp_threshold:
                cand = self.tracks.get(self.held) if self.held else None
                if cand is None or cand.alias not in seen or np.linalg.norm(cand.pos - eef) > self.cfg.grasp_radius:
                    near = self._nearest(eef, self.cfg.grasp_radius, only=seen)
                    cand = near or cand
                if cand is not None and cand.alias in seen and np.linalg.norm(cand.pos - eef) <= self.cfg.grasp_radius:
                    holder = cand.alias
                    obs[("inhand", holder, None)] = (0.97, "grip")
                self.held = cand.alias if cand is not None else None
            else:
                self.held, self.held_lifted = None, False
            for a in self.tracks:  # the gripper holds at most one object: the held one (if any)
                if a != holder and a != self.held:
                    obs[("inhand", a, None)] = (0.03, "grip")
        for a in seen:
            x = self.tracks[a]
            support = 0.0
            for y in self.tracks.values():
                if y.alias == a or y.category.container and x.category.container:
                    continue
                key, p = self._relation(x, y, y.alias in seen)
                obs[key] = (0.03, "geo") if a == holder else (p, "geo")
                support = max(support, p)
            if self.table_z is not None:  # On(x, table) only from the table height, never by default
                bottom = float(x.pos[2] - 0.5 * x.size[2])
                m = (self.cfg.table_tol - abs(bottom - self.table_z)) / self.cfg.margin_scale
                p = 0.5 + 0.47 * min(abs(m), 1.0) if m > 0 else 0.5 - 0.47 * min(abs(m), 1.0)
                p = 0.03 if a == holder else min(p, 1.0 - support)
                obs[("on", a, TABLE)] = (p, "geo")
        return obs

    def _fuse(self, turn: int, obs: dict[EdgeKey, tuple[float, str]], vdm: dict[EdgeKey, bool],
              expectations: dict[EdgeKey, tuple[bool, str]]) -> list[dict[str, Any]]:
        conflicts = []
        keys = set(obs) | set(vdm) | set(expectations) | set(self.graph.edges)
        for key in keys:
            o, v, e = obs.get(key), vdm.get(key), expectations.get(key)
            if o is not None and v is not None:
                pv = self.cfg.p_vdm if v else 1.0 - self.cfg.p_vdm
                if (o[0] >= 0.5) != v:
                    use_vdm = abs(pv - 0.5) > abs(o[0] - 0.5)
                    conflicts.append({"edge": _key_text(key), "geometry_p": round(o[0], 3), "vdm": v,
                                      "chosen": "vdm" if use_vdm else o[1]})
                    p, origin = (pv, "vdm") if use_vdm else o
                else:
                    p, origin = o
                self.graph.set_edge(key, p, "obs", turn)
            elif o is not None:
                p, origin = o
                self.graph.set_edge(key, p, "obs", turn)
            elif v is not None:
                p, origin = (self.cfg.p_vdm if v else 1.0 - self.cfg.p_vdm), "vdm"
                self.graph.set_edge(key, p, "obs", turn)
            elif e is not None:
                self.graph.set_expected(key, turn, holds=e[0])
                origin = "exp"
            else:
                edge = self.graph.edges[key]
                if edge.source != "prior":
                    edge.source = "mem"
                origin = "mem" if edge.source == "mem" else "prior"
            self.origin[key] = origin
        return conflicts

    def _hypotheses(self, turn: int, expectations: dict[EdgeKey, tuple[bool, str]], seen: set[str],
                    obs: dict[EdgeKey, tuple[float, str]]) -> list[Hypothesis]:
        keys: list[tuple[EdgeKey, str]] = [(k, "expected") for k in expectations if expectations[k][0]]
        for p in self.goal_relevant():
            key = _pred_key(p)
            if any(k == key for k, _ in keys):
                continue
            o = obs.get(key)
            if o is None or self.cfg.uncertain_low <= o[0] <= self.cfg.uncertain_high:
                keys.append((key, "uncertain"))
        return [Hypothesis(i + 1, k, pred_text(k), why) for i, (k, why) in enumerate(keys[: self.cfg.max_hypotheses])]

    def _object_refs(self, obs: dict[str, Any]) -> list[str]:
        cam = obs.get("agentview", {}) if isinstance(obs, dict) else {}
        K, T = cam.get("intrinsics"), cam.get("pose_mat")
        refs = []
        for a in sorted(self.tracks):
            t = self.tracks[a]
            px = project_to_pixel(t.pos, np.asarray(K), np.asarray(T)) if K is not None and T is not None else None
            where = f"near pixel ({px[0]:.0f}, {px[1]:.0f})" if px else "position unknown in the image"
            refs.append(f"- {a}: a {t.category.prompt}, {where}")
        return refs

    def end_turn(self, turn: int, obs: dict[str, Any], records: Sequence[CallRecord] = (), executed: bool = True) -> EstTurn:
        """Update after a turn from its hook records and the current observation."""
        result = EstTurn(turn, self.counter.progress(), records=[r.to_dict() for r in records],
                         perception=self.perception_name)
        expectations, snapshots, events, violations = self._apply_records(turn, records)
        result.expectations = [{"edge": _key_text(k, v[0]), "call": v[1]} for k, v in expectations.items()]

        proprio = proprio_from_obs(obs)
        eef = proprio[0] if proprio else None
        dets = self._dedup(self.perception.detect(obs, self.categories))
        seen, result.new_tracks, result.unmatched_tracks, result.rematched = self._associate(dets, eef, turn)
        result.detections = len(dets)
        geo = self._observe_relations(turn, seen, proprio)

        vdm: dict[EdgeKey, bool] = {}
        image = self._image(obs)
        if self.verifier is not None and image is not None:
            hyps = self._hypotheses(turn, expectations, seen, geo)
            result.hypotheses = [{"id": h.id, "edge": _key_text(h.key), "why": h.why} for h in hyps]
            if hyps:
                res = self.verifier.verify(self.instruction, self.last_image, image, hyps, self._object_refs(obs))
                result.verify = res.to_dict()
                for h in hyps:
                    ans = res.answers.get(h.id)
                    if ans in ("true", "false"):
                        vdm[h.key] = ans == "true"
        self.last_image = image

        result.conflicts = self._fuse(turn, geo, vdm, expectations)
        final = self._snapshot()

        # S1: expected effects contradicted by an observation (geometry, gripper or VDM) this turn.
        reported = {v.expected for v in violations}
        for key, (holds, desc) in expectations.items():
            edge = self.graph.edges.get(key)
            if edge is None or self.origin.get(key) not in ("geo", "grip", "vdm") or edge.holds == holds:
                continue
            expected = _key_text(key, holds)
            if expected in reported:
                continue
            if key[0] == "inhand" and holds:
                g = proprio[1] if proprio else None
                observed = (f"gripper empty (gripper {g:.2f})" if g is not None and g <= self.cfg.grasp_threshold
                            else f"{key[1]} not at the gripper ({self.location_text(key[1], final)})")
            else:
                observed = f"{key[1]} {self.location_text(key[1], final)} (P {edge.p:.2f} {self.origin[key]})"
            violations.append(Violation(turn, desc, expected, observed, key, holds))
        self.violations += violations
        result.violations = violations

        # A hook-predicted move the end-of-turn observation confirms is labelled exp+obs.
        for i in range(len(events) - 1, -1, -1):
            e = events[i]
            if e.source == "exp" and all(x.subject != e.subject for x in events[i + 1:]) \
                    and self.location_text(e.subject, final) == e.after and e.subject in seen:
                e.source = "exp+obs"
        # Events: changes from the last hook snapshot to the observed end state.
        last = snapshots[-1] if snapshots else self.loc
        for a in sorted(final):
            if a in result.new_tracks or a not in last:
                continue
            before = self.location_text(a, last)
            after = self.location_text(a, final)
            if before != after:
                src = "vdm" if any(self.origin.get(k) == "vdm" for k in self._location_edges(a)) else (
                    "obs" if a in seen or ("inhand", a, None) in geo else "mem")
                events.append(EstEvent(turn, a, before, after, src))
        self.graph.events.extend(events)
        result.events = events

        if self.goal.branches:
            self._count(turn, snapshots, final, result)
        self.loc = final
        self.turn = turn
        result.progress = self.counter.progress()
        result.pending = sorted(self.pending)
        result.snapshots = len(snapshots) + 1
        result.uncertain = self.uncertain()
        result.suppress_completion = (not result.progress.known) or bool(result.uncertain)
        return result

    # --- counting -------------------------------------------------------------------------------
    def _observed(self, pred: Pred) -> bool | None:
        """``pred`` as observed this turn (geometry, gripper or VDM) with confidence: True for
        P >= uncertain_high, False for P <= uncertain_low, None if unsure or not observed."""
        key = _pred_key(pred)
        edge = self.graph.edges.get(key)
        if edge is None or self.origin.get(key) not in ("geo", "grip", "vdm"):
            return None
        if edge.p >= self.cfg.uncertain_high:
            return True
        if edge.p <= self.cfg.uncertain_low:
            return False
        return None

    def _verdict(self, step: Step, moved_later: bool) -> tuple[str, str]:
        """(confirmed | rejected | pending, obs | vdm): the end-of-turn evidence for a count."""
        if moved_later:  # picked up again in the same turn: no end-of-turn evidence for this count
            return "pending", ""
        seen = [self._observed(p) for p in step]
        how = "vdm" if any(self.origin.get(_pred_key(p)) == "vdm" for p in step) else "obs"
        if any(v is False for v in seen):
            return "rejected", how
        return ("confirmed", how) if all(v is True for v in seen) else ("pending", "")

    def _withdraw(self, turn: int, result: EstTurn, step: Step, expected: str, observed: str) -> None:
        result.rolled_back.append(format_step(step))
        subjects = {p[1] for p in step}
        if not any(v.expected.split(" ")[0] in subjects for v in result.violations):
            self.violations.append(Violation(turn, "progress check", expected, observed, _pred_key(step[0]), True))
            result.violations.append(self.violations[-1])

    def _confirm(self, idx: int, how: str) -> None:
        """Count ``idx`` confirmed by observation. Earlier pending counts of a *different* step are
        confirmed with it (a later step of the sequence was reached). Earlier pending repeats of the
        same step stay pending: seeing the third place-back says nothing about the first two."""
        step = self.counter.done[idx]
        keep = set()
        for q in self.pending:
            if q < idx and self.counter.done[q] != step:
                self.confirmed_by[q] = "later_step"
            elif q != idx:
                keep.add(q)
        self.pending = keep
        self.confirmed_by[idx] = how

    def _count(self, turn: int, snapshots: list[Snapshot], final: Snapshot, result: EstTurn) -> None:
        """Feed this turn's snapshots. A count is pending until an end-of-turn observation confirms
        it; a contradicted one (on a hook snapshot or the final one) is withdrawn with S1 and the
        turn is counted again without it; a confirmed count also confirms the pending counts before
        it (recorded as confirmed_by=later_step)."""
        def subjects(step: Step) -> set[str]:
            return {p[1] for p in step}

        moved = {a for snap in snapshots for a in snap if snap.get(a) != self.loc.get(a)}
        # The last count of an earlier turn, if still pending, is judged by this turn's observation
        # when the hooks did not move its object this turn.
        last = len(self.counter.done) - 1
        if last in self.pending and not (subjects(self.counter.done[last]) & moved):
            v, how = self._verdict(self.counter.done[last], False)
            if v == "confirmed":
                self._confirm(last, how)
            elif v == "rejected":
                step = self.counter.done.pop()
                self.pending.discard(last)
                self.counter.was_false[step] = True
                self._withdraw(turn, result, step, f"{format_step(step)} (counted in an earlier turn)",
                               "not what the camera shows now: count withdrawn")

        ckpt = self.counter.checkpoint()
        count_final = True
        for _attempt in range(len(snapshots) + 2):
            counts: list[tuple[Step, int | None]] = []
            for i, snap in enumerate(snapshots):
                counts += [(st, i) for st in self.counter.feed(snap)]
            counts += [(st, None) for st in self.counter.feed(final, final=True, count=count_final)]
            verdicts = []
            for st, i in counts:
                moved_later = i is not None and any(
                    snap.get(a) != snapshots[i].get(a) for snap in snapshots[i + 1:] for a in subjects(st))
                verdicts.append(self._verdict(st, moved_later))
            bad = [i for (_, i), (v, _) in zip(counts, verdicts, strict=True) if v == "rejected"]
            if not bad:
                break
            # Withdraw and count the turn again: without the first contradicted hook snapshot and the
            # ones after it, or (contradicted on the final snapshot) without counting on it.
            for (st, i), (v, _) in zip(counts, verdicts, strict=True):
                if v == "rejected":
                    where = "after the hooked place" if i is not None else "at the end of the turn"
                    self._withdraw(turn, result, st, f"{format_step(st)} {where}",
                                   "not confirmed by the camera: count withdrawn")
            self.counter.restore(ckpt)
            intra = [i for i in bad if i is not None]
            if intra:
                snapshots = snapshots[:min(intra)]
            else:
                count_final = False
        base = len(ckpt[0])
        for j, (v, how) in enumerate(verdicts):
            idx = base + j
            if idx >= len(self.counter.done):
                break
            if v == "confirmed":
                self._confirm(idx, how)
            else:
                self.pending.add(idx)
                self.confirmed_by.pop(idx, None)
        n = len(self.counter.done)
        self.pending = {q for q in self.pending if q < n}
        self.confirmed_by = {q: h for q, h in self.confirmed_by.items() if q < n}
        result.confirmed_by = [self.confirmed_by.get(q) for q in range(n)]

    # --- progress / uncertainty -----------------------------------------------------------------
    def goal_relevant(self) -> list[Pred]:
        """Predicates of the last counted step and of the next possible steps."""
        prog = self.counter.progress()
        steps = ([prog.done[-1]] if prog.done else []) + prog.next_steps()
        out: list[Pred] = []
        for s in steps:
            for p in s:
                if p not in out:
                    out.append(p)
        return out

    def pred_belief(self, p: Pred) -> tuple[float, str, int | None]:
        """(P, source label, last seen turn) of a goal predicate."""
        subj = self.graph.nodes.get(p[1])
        seen = subj.last_seen_turn if subj else None
        if p[0] != "inhand" and p[2] != TABLE:
            tgt = self.graph.nodes.get(p[2])
            if tgt is None:
                seen = None
            elif seen is not None and tgt.last_seen_turn is not None:
                seen = min(seen, tgt.last_seen_turn)
        edge = self.graph.edges.get(_pred_key(p))
        if edge is None:
            return 0.5, "prior", seen
        return edge.p, self._label(edge.key), seen

    def _label(self, key: EdgeKey) -> str:
        o = self.origin.get(key, self.graph.edges[key].source if key in self.graph.edges else "prior")
        return {"geo": "obs", "grip": "obs", "expected": "exp"}.get(o, o)

    @property
    def perception_name(self) -> str:
        return str(getattr(self.perception, "name", type(self.perception).__name__))

    def state(self) -> dict[str, Any]:
        """Small summary for logs (includes which perception source fed the graph)."""
        p = self.counter.progress()
        return {"perception": self.perception_name, "tracks": len(self.tracks), "goal_source": self.goal.source,
                "goal_parsed": self.goal.branches is not None, "done": len(p.done), "remaining": p.remaining,
                "known": p.known, "overshot": p.overshot, "pending": len(self.pending),
                "later_step_confirmations": self.later_step_confirmations(),
                "violations": len(self.violations)}

    def later_step_confirmations(self) -> int:
        """Counted steps confirmed only because a later step was observed (not seen themselves)."""
        return sum(1 for q, h in self.confirmed_by.items() if h == "later_step" and q < len(self.counter.done))

    def uncertain(self) -> list[Uncertain]:
        """S4 / S5 suppression: goal predicates of the last counted and the next steps that are unsure,
        plus every counted step not yet confirmed by observation. Earlier counted steps are not
        re-checked against the current state: later steps may undo them on purpose (e.g. a swap)."""
        out = []
        for j in sorted(self.pending):
            step = self.counter.done[j]
            prob, src, seen = self.pred_belief(step[0])
            out.append(Uncertain(f"{format_step(step)} (step {j + 1}, counted from the API calls, not yet seen)",
                                 prob, src, seen, "unconfirmed", _pred_key(step[0])))
        flagged = {u.key for u in out}
        for p in self.goal_relevant():
            if _pred_key(p) in flagged:  # already listed as an unconfirmed count
                continue
            prob, src, seen = self.pred_belief(p)
            if self.cfg.uncertain_low <= prob <= self.cfg.uncertain_high:
                out.append(Uncertain(format_pred(p), prob, src, seen, "p", _pred_key(p)))
            elif src in ("mem", "exp", "prior") and (seen is None or self.turn - seen >= self.cfg.stale_turns):
                out.append(Uncertain(format_pred(p), prob, src, seen, "stale", _pred_key(p)))
        return out

    def current_truths(self) -> dict[Pred, bool]:
        return {p: self.pred_belief(p)[0] > 0.5 for p in goal_preds(self.goal.branches or [])}

    # --- text -----------------------------------------------------------------------------------
    def objects_text(self, with_p: bool = True, prefix: str = "", sep: str = " | ") -> str:
        parts = []
        for a in sorted(self.tracks):
            t = self.tracks[a]
            text = f"{a} ({t.category.prompt}) at ({', '.join(f'{(round(float(v), 2) or 0.0):.2f}' for v in t.pos)})"
            rels = sorted((k for k in self._location_edges(a) if self.graph.edges[k].holds),
                          key=lambda k: (k[0] != "inhand", k[0], k[2] or ""))
            if rels:
                for k in rels:
                    e = self.graph.edges[k]
                    rel = "in gripper" if k[0] == "inhand" else f"{k[0]} {k[2]}"
                    text += f" {rel}" + (f" {e.p:.2f} {self._label(k)}" if with_p else "")
            else:
                text += f" {UNKNOWN_LOCATION}"
            node = self.graph.nodes.get(a)
            if with_p and node is not None and node.last_seen_turn is not None and node.last_seen_turn < self.turn:
                text += f", not seen since t{node.last_seen_turn}"
            parts.append(text)
        return prefix + sep.join(parts)

    def violations_line(self) -> str:
        shown = self.violations[-self.cfg.max_violations_shown:]
        if not shown:
            return "Violations: none"
        return "Violations: " + " | ".join(f"t{v.turn} {v.text}" for v in shown)

    def progress_text(self, uncertain: bool = False) -> str:
        if not self.goal.branches:
            return "Progress: progress unknown (the goal could not be parsed into steps)"
        done_text = "all steps counted, but NOT all confirmed" if uncertain else "all steps estimated complete"
        text = progress_line(self.counter.progress(), self.current_truths(), complete_text=done_text)
        if self.pending:
            text += f" | {len(self.pending)} counted step(s) not yet confirmed by observation"
        if self.later_step_confirmations():
            text += f" | {self.later_step_confirmations()} step(s) confirmed only by a later step, not seen"
        return text + (" | estimate uncertain (see S4)" if uncertain else "")

    def render(self, signals: Sequence[Signal] = (), max_chars: int = 3000, uncertain: bool = False) -> str:
        """Prompt text: the oracle layout plus P / source per relation and a Violations line."""
        return serialize(
            self.graph, None, signals, turn=self.turn, max_chars=max_chars,
            title=EST_TITLE.format(turn=self.turn),
            progress_text=self.progress_text(uncertain),
            objects_text=self.objects_text(prefix=EST_OBJECTS_PREFIX),
            objects_prefix=EST_OBJECTS_PREFIX,
            fixed_lines=[self.violations_line()],
        )
