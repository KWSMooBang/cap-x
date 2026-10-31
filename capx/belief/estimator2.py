"""Estimated Belief Graph v2 (``bg_est2``): B1K-style CoT record + VDM Observe + declared effects.

Built on the v1 estimator (``estimator.py``: tracks, data association, geometry, API-call hooks,
step counting) with these changes (design: /outputs/cap-bg/tasks/bg-est2-design.md, sections 3-4):

* Marks (``som.py``): every track has a mark (A, B, ... / 1, 2, ...) drawn at its image center.
* Observe (``observe.py``) replaces v1's verification VDM: M2 once at the start (all candidate
  on / in relations), M5 at the end of every turn (at most ``observe_max_statements``: declared
  Effects, the moved / declared objects' relations, goal-relevant predicates). 1 / 0 -> P 0.9 / 0.1
  (source vdm); ? keeps the memory. A reply with no readable statement -> geometry only (v1 way).
* Robot predicates from the robot state, P 0.97 (source robot): inhand(o), handempty,
  above(gripper, t), lifted(o). They win over the VDM (conflicts logged).
* Declarations (``declare.py``): ``# Subtask:`` / ``# Effect:`` of the executed block. Declared
  Effects become expectations (0.7 / 0.3) only if the operator's API call ran; no / bad declaration
  -> v1 hook inference; a declaration the API calls contradict is dropped.
* Goal (M3): viewpoint-explicit scene, steps normalized to newly-true predicates; regions (center)
  as at(o, r), judged from the distance to the region center.
* Subtasks (``subtasks.py``): goal steps expanded into pick / place / put_in / place_region.
* Fusion priority: robot > Observe (vdm) > [geometry only when Observe failed] > expected > memory.
  An Observe 1 for on / in whose objects are more than 2 support radii apart -> P 0.5 + S4; Notes
  about a tilted / fallen object or a wrong mark -> that object's observed relations P 0.5 + S4.
* Steps: counted from hook snapshots like v1; confirmed only by an observation (Observe / robot);
  declaration + API only -> pending; Observe 0 -> withdrawn + S1.
* Signals: S1 (declared Effect vs Observe, failed subtask, grasp lost), S2 (no step AND no subtask
  progress), S3 (next subtask's precondition false), S4 (? twice in a row, geometric anomaly, Notes,
  unconfirmed counts), S5 (suppressed while anything is pending / uncertain).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .declare import Declaration, RecordFacts, check_records, parse_declaration
from .estimator import (
    UNKNOWN_LOCATION,
    BeliefEstimator,
    EstConfig,
    EstEvent,
    EstTurn,
    Snapshot,
    StepCounter,
    _pred_key,
)
from .goal import GoalResult, goal_preds, parse_goal_llm_v2, regions_in, scene_lines_v2
from .graph import EdgeKey, format_location
from .hooks import CallRecord
from .observe import ObserveResult, VdmObserver, note_flags, statement_text
from .perception import (
    VOCABULARY,
    Category,
    PerceptionSource,
    categories_from_instruction,
    project_to_pixel,
    proprio_from_obs,
)
from .progress import Pred, Progress, Step
from .prompts_v2 import marks_now_text, observe_initial_prompt, observe_turn_prompt
from .signals import Uncertain, Violation
from .som import MarkTable, draw_marks, is_support
from .subtasks import PLACE_OPS, Subtask, SubtaskLog, current_support, expand_step
from .verify import encode_png

TABLE = "table"
GRIPPER = "gripper"
LOC_PREDS = ("on", "in", "at", "inhand")
EXCLUSIVE = ("on", "in", "inhand")  # one of these per object at a time
ROBOT_PREDS = ("inhand", "handempty", "above", "lifted")
FIXED_KEYS = ("plate",)  # supports that are never moved in LIBERO-Mem: not Observe subjects
OBSERVED_ORIGINS = ("vdm", "robot", "geo")
TITLE0 = ("Belief graph (turn 0, ESTIMATED; names match the marks in the camera image; xyz in the robot base "
          "frame, m: x away from the robot, y to the robot's left):")
TITLE = ("Belief graph (turn {turn}, ESTIMATED; names match the marks; xyz in the robot base frame, m: x away "
         "from the robot, y to the robot's left):")


@dataclass
class Est2Config(EstConfig):
    observe_max_statements: int = 10  # M5 statements per turn
    observe_max_initial: int = 20  # M2 statements at the start
    p_robot: float = 0.97
    above_min_dz: float = 0.05  # m: fingertips this far above a support (and within its radius) -> above
    anomaly_factor: float = 2.0  # Observe on=1 with the objects more than this many support radii apart -> P 0.5
    region_radius: float = 0.10  # m: at(o, region) when o's center is this close (xy) to the region center
    region_center: tuple[float, float] | None = None  # base-frame xy; None: camera axis meets the table
    unseen_turns: int = 2  # the same goal-relevant key answered ? this many turns in a row -> S4
    pending_turns: int = 2  # a counted step unconfirmed for this many turns -> S4
    max_belief_items: int = 12
    event_frames: bool = False  # extra Observe on recorded frames right after gripper close / open


@dataclass
class EstTurn2(EstTurn):
    declaration: dict[str, Any] | None = None
    observe: dict[str, Any] | None = None
    statements: list[str] = field(default_factory=list)
    judged: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    notes_flags: dict[str, str] = field(default_factory=dict)
    s1_items: list[str] = field(default_factory=list)
    s3_items: list[str] = field(default_factory=list)
    s4_items: list[str] = field(default_factory=list)
    subtask_progress: bool = False
    robot_conflicts: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    fallback: bool = False  # Observe failed: geometry only
    event_observes: list[dict[str, Any]] = field(default_factory=list)

    def to_log(self) -> dict[str, Any]:
        out = super().to_log()
        out.update(declaration=self.declaration, observe=self.observe, statements=self.statements,
                   judged=self.judged, anomalies=self.anomalies, notes_flags=self.notes_flags,
                   s1_items=self.s1_items, s3_items=self.s3_items, s4_items=self.s4_items,
                   subtask_progress=self.subtask_progress, robot_conflicts=self.robot_conflicts,
                   facts=self.facts, fallback=self.fallback, event_observes=self.event_observes)
        return out


def key_text(key: EdgeKey) -> str:
    return statement_text(key)


def pred_stmt(p: Pred) -> str:
    return statement_text(_pred_key(p))


def step_stmt(step: Step) -> str:
    return " and ".join(pred_stmt(p) for p in step)


class BeliefEstimatorV2(BeliefEstimator):
    def __init__(
        self,
        perception: PerceptionSource,
        config: Est2Config | None = None,
        goal_query_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
        observer: VdmObserver | None = None,
        vocabulary: Sequence[Category] = VOCABULARY,
        categories: Sequence[Category] | None = None,
    ) -> None:
        super().__init__(perception, config or Est2Config(), goal_query_fn, None, vocabulary, categories)
        self.cfg: Est2Config
        self.observer = observer
        self.marks = MarkTable()
        self.regions: list[str] = []
        self.region_xy: dict[str, np.ndarray] = {}
        self.q_streak: dict[EdgeKey, int] = {}
        self.key_seen: dict[EdgeKey, int] = {}
        self.obs_now: dict[EdgeKey, str] = {}  # this turn's Observe values
        self.flags: dict[str, tuple[str, str]] = {}  # alias -> (kind, note) from this turn's Notes
        self.anomaly_keys: set[EdgeKey] = set()
        self.subtasks = SubtaskLog()
        self.decl: Declaration | None = None
        self.last_observe: ObserveResult | None = None
        self.last_marked: str | None = None  # data URL of the last marked image (M5 image 1)
        self.pixels: dict[str, tuple[float, float]] = {}
        self.seen: set[str] = set()
        self.robot: dict[str, Any] = {}
        self.count_turn: dict[int, int] = {}
        self.last2: EstTurn2 | None = None
        self.cot: dict[str, Any] = {}
        self.event_values: dict[EdgeKey, tuple[str, int, int | None]] = {}  # event Observe: key -> (value, turn, sim step)
        self._grasp_hint: str | None = None  # the object the current declaration names

    # --- names / helpers ------------------------------------------------------------------------
    @property
    def aliases(self) -> list[str]:
        return sorted(self.tracks)

    def movables(self) -> list[str]:
        return [a for a in self.aliases if self.tracks[a].category.key not in FIXED_KEYS]

    def supports(self) -> list[str]:
        return [a for a in self.aliases if is_support(self.tracks[a].category.key)]

    def containers(self) -> set[str]:
        return {a for a in self.aliases if self.tracks[a].category.container}

    def _radius(self, alias: str) -> float:
        t = self.tracks[alias]
        return max(0.5 * float(max(t.size[0], t.size[1])), self.cfg.support_radius_min)

    def _nearest(self, eef: np.ndarray, radius: float, only: set[str] | None = None):  # type: ignore[override]
        """v1's rule (the highest tracked object within ``radius``), except that the object a valid
        declaration names wins when it is within ``radius`` too (e.g. a basket grasped by its rim with
        the cream cheese inside it)."""
        hint = self._grasp_hint
        t = self.tracks.get(hint) if hint else None
        if t is not None and (only is None or hint in only) and float(np.linalg.norm(t.pos - eef)) <= radius:
            return t
        return super()._nearest(eef, radius, only)

    def _location_edges(self, alias: str) -> list[EdgeKey]:
        return [k for k in self.graph.edges if k[1] == alias and k[0] in LOC_PREDS]

    def _snapshot(self) -> Snapshot:
        out = {}
        for a in self.tracks:
            # A relation set to P 0.5 by the anomaly / Notes rules is uncertain: it does not place the object.
            out[a] = frozenset((k[0], k[2]) for k in self._location_edges(a)
                               if self.graph.edges[k].holds and k not in self.anomaly_keys)
        return out

    def _cam(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        cam = obs.get("agentview", {}) if isinstance(obs, dict) else {}
        K, T = cam.get("intrinsics"), cam.get("pose_mat")
        if K is None or T is None:
            return None
        return np.asarray(K, float), np.asarray(T, float)

    def _rgb(self, obs: dict[str, Any]) -> np.ndarray | None:
        try:
            return np.asarray(obs["agentview"]["images"]["rgb"])
        except Exception:
            return None

    def _compute_pixels(self, obs: dict[str, Any]) -> dict[str, tuple[float, float]]:
        cam = self._cam(obs)
        if cam is None:
            return {}
        out = {}
        for a, t in self.tracks.items():
            px = project_to_pixel(t.pos, cam[0], cam[1])
            if px is not None:
                out[a] = px
        return out

    def _marked(self, rgb: np.ndarray, aliases: Sequence[str]) -> tuple[str | None, list[tuple[str, str, tuple[float, float]]]]:
        """Data URL of ``rgb`` with the marks of ``aliases`` (those with a pixel inside the image)."""
        h, w = rgb.shape[0], rgb.shape[1]
        shown = []
        for m, a in self.marks.ordered():
            px = self.pixels.get(a)
            if a in aliases and px is not None and 0 <= px[0] < w and 0 <= px[1] < h:
                shown.append((m, a, px))
        img = draw_marks(rgb, [(m, px) for m, _, px in shown])
        try:
            return encode_png(img), shown
        except Exception:
            return None, shown

    def _assign_marks(self) -> None:
        for a in self.aliases:
            self.marks.assign(a, self.tracks[a].category.key)

    def _region_setup(self, obs: dict[str, Any]) -> None:
        """Region centers in the base frame. ``center``: where the camera's optical axis meets the table
        (the front camera looks at the middle of the table), unless configured."""
        for r in self.regions:
            if self.cfg.region_center is not None:
                self.region_xy[r] = np.asarray(self.cfg.region_center, float)
                continue
            cam = self._cam(obs)
            z = self.table_z if self.table_z is not None else 0.0
            if cam is None:
                xs = [t.pos[:2] for t in self.tracks.values()]
                self.region_xy[r] = np.mean(xs, axis=0) if xs else np.zeros(2)
                continue
            T = cam[1]
            origin, axis = T[:3, 3], T[:3, 2]
            s = (z - origin[2]) / axis[2] if abs(axis[2]) > 1e-6 else 0.0
            self.region_xy[r] = (origin + s * axis)[:2]

    # --- robot predicates -----------------------------------------------------------------------
    def _robot_preds(self, raw: dict[EdgeKey, tuple[float, str]], proprio: tuple[np.ndarray, float] | None) -> dict[EdgeKey, float]:
        """inhand (v1 gripper rule), handempty, above(gripper, t), lifted(o); P 0.97 / 0.03."""
        hi, lo = self.cfg.p_robot, 1.0 - self.cfg.p_robot
        out: dict[EdgeKey, float] = {k: (hi if v[0] >= 0.5 else lo) for k, v in raw.items() if k[0] == "inhand"}
        if proprio is None:
            self.robot = {}
            return out
        eef, g = proprio
        holder = next((k[1] for k, v in out.items() if v >= 0.5), None)
        out[("handempty", GRIPPER, None)] = lo if holder else hi
        above = None
        for s in self.supports():
            t = self.tracks[s]
            ok = (float(np.linalg.norm(eef[:2] - t.pos[:2])) < self._radius(s)
                  and float(eef[2]) - float(t.pos[2]) >= self.cfg.above_min_dz and s != holder)
            out[("above", GRIPPER, s)] = hi if ok else lo
            if ok and (above is None or t.pos[2] > self.tracks[above].pos[2]):
                above = s
        lifted = bool(holder) and (self.held_lifted or float(eef[2]) - self.grasp_z >= self.cfg.lift_min)
        for a in self.movables():
            out[("lifted", a, None)] = hi if (a == holder and lifted) else lo
        self.robot = {"eef": [round(float(v), 4) for v in eef], "gripper": round(float(g), 4),
                      "handempty": holder is None, "above": above, "holding": holder, "lifted": bool(lifted)}
        return out

    # --- geometry -------------------------------------------------------------------------------
    def _geom(self, key: EdgeKey) -> dict[str, float] | None:
        pred, s, t = key
        if pred not in ("on", "in", "at") or s not in self.tracks:
            return None
        x = self.tracks[s]
        if pred == "at":
            c = self.region_xy.get(t or "")
            return None if c is None else {"d_xy": round(float(np.linalg.norm(x.pos[:2] - c)), 3)}
        if t == TABLE:
            if self.table_z is None:
                return None
            return {"dz": round(float(x.pos[2] - 0.5 * x.size[2] - self.table_z), 3)}
        if t not in self.tracks:
            return None
        y = self.tracks[t]
        return {"dz": round(float(x.pos[2] - y.pos[2]), 3), "d_xy": round(float(np.linalg.norm(x.pos[:2] - y.pos[:2])), 3),
                "r": round(self._radius(t), 3)}

    def _region_geo(self, seen: set[str]) -> dict[EdgeKey, tuple[float, str]]:
        out = {}
        for r, c in self.region_xy.items():
            for a in self.movables():
                if a not in seen:
                    continue
                d = float(np.linalg.norm(self.tracks[a].pos[:2] - c))
                m = (self.cfg.region_radius - d) / self.cfg.margin_scale
                cval = min(abs(m), 1.0)
                out[("at", a, r)] = (0.5 + 0.47 * cval if m > 0 else 0.5 - 0.47 * cval, "geo")
        return out

    def _geo_p(self, key: EdgeKey, geo: dict[EdgeKey, tuple[float, str]]) -> float | None:
        v = geo.get(key)
        return None if v is None else v[0]

    # --- statements -----------------------------------------------------------------------------
    def _rel_key(self, subj: str, target: str) -> EdgeKey:
        if target == TABLE:
            return ("on", subj, TABLE)
        if target in self.region_xy:
            return ("at", subj, target)
        return ("in" if target in self.containers() else "on", subj, target)

    def _initial_statements(self, geo: dict[EdgeKey, tuple[float, str]]) -> list[EdgeKey]:
        keys: list[EdgeKey] = []
        for a in self.movables():
            for s in self.supports():
                if s == a or (self.tracks[a].category.container and self.tracks[s].category.container):
                    continue
                keys.append(self._rel_key(a, s))
        geo_keys = sorted(keys, key=lambda k: -(self._geo_p(k, geo) or 0.0))
        table = [("on", a, TABLE) for a in self.movables()]
        n = self.cfg.observe_max_initial
        # Relations of objects to supports first (most likely first), then each object on the table.
        out = geo_keys[: max(n - len(table), 0)] + table
        return out[:n]

    def _turn_statements(self, decl: Declaration | None, moved: set[str], before: Snapshot) -> list[EdgeKey]:
        out: list[EdgeKey] = []

        def add(k: EdgeKey | None) -> None:
            if k is None or k in out or k[0] in ("above", "lifted"):
                return
            if k[1] != GRIPPER and k[1] not in self.tracks:
                return
            if k[1] in self.tracks and self.tracks[k[1]].category.key in FIXED_KEYS:
                return  # plates are not moved in LIBERO-Mem: no statements about where they are
            if k[2] not in (None, TABLE) and k[2] not in self.tracks and k[2] not in self.region_xy:
                return
            out.append(k)

        focus: list[str] = []
        if decl is not None and decl.valid:
            for e in decl.effects:
                add(e.key)
            if decl.args and decl.args[0] in self.tracks:
                focus.append(decl.args[0])
                if len(decl.args) > 1 and decl.op in PLACE_OPS:
                    add(self._rel_key(decl.args[0], decl.args[1]))
        focus += [a for a in sorted(moved) if a not in focus]
        goal_rel = self.goal_relevant()
        for a in focus:
            for _, t in sorted(before.get(a, frozenset()), key=str):
                if t is not None:
                    add(self._rel_key(a, t))
            for p in goal_rel:
                if p[1] == a and len(p) > 2:
                    add(_pred_key(p))
            add(("on", a, TABLE))
        for p in goal_rel:
            add(_pred_key(p))
        for j in sorted(self.pending):
            for p in self.counter.done[j]:
                add(_pred_key(p))
        # Keep the rest of the scene fresh: where the goal objects are believed to be.
        goal_objs = {p[1] for p in goal_preds(self.goal.branches or [])}
        for a in self.movables():
            if a in goal_objs:
                for pr, t in sorted(self.loc.get(a, frozenset()), key=str):
                    if pr != "inhand" and t is not None:
                        add(self._rel_key(a, t))
        return out[: self.cfg.observe_max_statements]

    # --- fusion ---------------------------------------------------------------------------------
    def _fuse2(self, turn: int, robot: dict[EdgeKey, float], geo: dict[EdgeKey, tuple[float, str]],
               values: dict[EdgeKey, str], expectations: dict[EdgeKey, tuple[bool, str]], fallback: bool,
               result: EstTurn2) -> None:
        cfg = self.cfg
        keys = set(robot) | set(values) | set(expectations) | set(self.graph.edges)
        keys |= {k for k in geo if fallback or k not in self.graph.edges}
        set_now: dict[EdgeKey, str] = {}
        for key in keys:
            r, v, e, g = robot.get(key), values.get(key), expectations.get(key), geo.get(key)
            if r is not None:
                if v in ("1", "0") and (v == "1") != (r >= 0.5):
                    result.robot_conflicts.append({"key": key_text(key), "robot": round(r, 2), "observe": v})
                self.graph.set_edge(key, r, "obs", turn)
                origin = "robot"
            elif v in ("1", "0"):
                self.graph.set_edge(key, cfg.p_vdm if v == "1" else 1.0 - cfg.p_vdm, "obs", turn)
                origin = "vdm"
            elif (fallback or key not in self.graph.edges) and g is not None:
                # Observe failed, or nothing is known about this key yet (? on the first look): geometry.
                self.graph.set_edge(key, g[0], "obs", turn)
                origin = "geo"
            elif e is not None:
                self.graph.set_expected(key, turn, holds=e[0])
                origin = "exp"
            elif key in self.graph.edges:
                edge = self.graph.edges[key]
                if edge.source != "prior":
                    edge.source = "mem"
                origin = "mem" if edge.source == "mem" else "prior"
            else:
                continue
            self.origin[key] = origin
            if origin in OBSERVED_ORIGINS:
                self.key_seen[key] = turn
                set_now[key] = origin
            elif origin == "exp":
                set_now[key] = origin
        # One location per object: a location set true now lowers the object's other location
        # relations that were not observed now (e.g. on plate_C from memory once on plate_B is seen).
        # Regions (at) overlap supports (a basket on the table at the center): not exclusive.
        for key, origin in set_now.items():
            if key[0] not in EXCLUSIVE or not self.graph.edges[key].holds:
                continue
            for other in self._location_edges(key[1]):
                if other[0] not in EXCLUSIVE or other == key or other in set_now or not self.graph.edges[other].holds:
                    continue
                self.graph.edges[other].p = min(self.graph.edges[other].p, 0.3 if origin == "exp" else 1.0 - cfg.p_vdm)
                self.graph.edges[other].source = "mem"
                self.origin[other] = "excl"

    def _anomalies(self, turn: int, values: dict[EdgeKey, str], seen: set[str], result: EstTurn2) -> None:
        """Observe on/in = 1 while the two objects are far apart (both seen now): the mark may be wrong."""
        self.anomaly_keys = set()
        for key, v in values.items():
            if v != "1" or key[0] not in ("on", "in") or key[2] in (None, TABLE) or key[2] not in self.tracks:
                continue
            if key[1] not in seen or key[2] not in seen:
                continue
            g = self._geom(key) or {}
            if g.get("d_xy", 0.0) > self.cfg.anomaly_factor * self._radius(key[2]):
                self.graph.set_edge(key, 0.5, "obs", turn)
                self.anomaly_keys.add(key)
                result.anomalies.append({"key": key_text(key), **g})

    def _apply_notes(self, turn: int, res: ObserveResult | None, values: dict[EdgeKey, str], result: EstTurn2) -> None:
        self.flags = {}
        if res is None or not res.parse.notes:
            return
        flags = note_flags(res.parse.notes, self.aliases, self.marks.by_alias)
        result.notes_flags = flags
        for a, kind in flags.items():
            self.flags[a] = (kind, res.parse.notes)
            for key, v in values.items():
                if kind == "wrong mark":
                    hit = (key[1] == a or key[2] == a) and v in ("1", "0")
                else:  # tilted / fallen: "on a support" is not trustworthy
                    hit = key[1] == a and v == "1" and key[2] not in (None, TABLE)
                if hit:
                    self.graph.set_edge(key, 0.5, "obs", turn)
                    self.anomaly_keys.add(key)

    # --- turn 0 ---------------------------------------------------------------------------------
    def start(self, obs: dict[str, Any], instruction: str,
              goal_fn: Callable[[BeliefEstimator], GoalResult] | None = None) -> EstTurn2:
        self.instruction = instruction
        self.categories = self.fixed_categories or categories_from_instruction(
            instruction, self.cfg.max_categories, self.vocabulary)
        self.regions = regions_in(instruction)
        proprio = proprio_from_obs(obs)
        eef = proprio[0] if proprio else None
        if proprio is not None and proprio[1] <= self.cfg.grasp_threshold:
            self.gripper_cmd = "closed"
        dets = self._dedup(self.perception.detect(obs, self.categories))
        self._initial_tracks(dets)
        if dets:
            self.table_z = min(float(d.center[2] - 0.5 * d.size[2]) for d in dets)
        for _ in range(self.cfg.init_passes - 1):
            self._associate(self._dedup(self.perception.detect(obs, self.categories)), eef, 0)
        self._assign_marks()
        self._region_setup(obs)
        self.seen = set(self.tracks)
        self.pixels = self._compute_pixels(obs)
        result = EstTurn2(0, self.counter.progress(), detections=len(dets), new_tracks=sorted(self.tracks),
                          perception=self.perception_name)

        raw = self._observe_relations(0, set(self.tracks), proprio)
        geo = {k: v for k, v in raw.items() if k[0] != "inhand"}
        geo.update(self._region_geo(self.seen))
        robot = self._robot_preds(raw, proprio)

        res = None
        rgb = self._rgb(obs)
        statements = self._initial_statements(geo)
        if rgb is not None:
            image, shown = self._marked(rgb, self.aliases)
            self.last_marked = image
            if self.observer is not None and image is not None:
                marks = [(m, a, self.tracks[a].category.prompt, px) for m, a, px in shown]
                prompt = observe_initial_prompt(instruction, image, marks, [statement_text(k) for k in statements])
                res = self.observer.observe("initial", prompt, statements, self.aliases)
        values = res.parse.values if res is not None and res.ok else {}
        result.fallback = not values
        result.statements = [statement_text(k) for k in statements]
        result.observe = res.to_dict() if res is not None else None
        self.last_observe = res
        self._fuse2(0, robot, geo, values, {}, result.fallback, result)
        self._anomalies(0, values, self.seen, result)
        self._apply_notes(0, res, values, result)
        self._update_streaks(statements, values)
        self.obs_now = dict(values)
        self.loc = self._snapshot()

        if goal_fn is not None:
            self.goal = goal_fn(self)
        elif self.goal_query_fn is not None:
            self.goal = parse_goal_llm_v2(self.goal_query_fn, instruction, self.scene_lines(), self.aliases,
                                          dict(self.loc), regions=tuple(self.regions))
        self.counter = StepCounter(self.goal.branches or [])
        if self.goal.branches:
            self.counter.feed(self.loc)
        self.turn = 0
        result.progress = self.counter.progress()
        result.uncertain, result.s4_items = self._uncertain()
        result.suppress_completion = (not result.progress.known) or bool(result.uncertain)
        self.last2 = result
        self.cot = self._cot(result, [])
        return result

    def scene_lines(self) -> list[str]:
        """M3 scene from the initial belief: movables 'on X' / supports 'empty' or 'holds X'."""
        objs = []
        holds: dict[str, list[str]] = {}
        for a in self.aliases:
            for p, t in self.loc.get(a, frozenset()):
                if p in ("on", "in") and t in self.tracks:
                    holds.setdefault(t, []).append(a)
        for a in self.aliases:
            t = self.tracks[a]
            if is_support(t.category.key) and t.category.key in FIXED_KEYS:
                rel = ("holds " + ", ".join(sorted(holds[a]))) if a in holds else "empty"
            else:
                rels = [(p, x) for p, x in self.loc.get(a, frozenset()) if p != "inhand"]
                rel = format_location(rels).replace("on table", "on the table") if rels else "location unclear"
                if is_support(t.category.key):
                    rel += "; " + (("holds " + ", ".join(sorted(holds[a]))) if a in holds else "empty")
            objs.append((a, t.category.prompt, t.pos, rel))
        return scene_lines_v2(objs, self.regions)

    # --- per turn -------------------------------------------------------------------------------
    def end_turn(self, turn: int, obs: dict[str, Any], records: Sequence[CallRecord] = (), executed: bool = True,
                 code: str | None = None, event_frames: Sequence[dict[str, Any]] = ()) -> EstTurn2:
        cfg = self.cfg
        result = EstTurn2(turn, self.counter.progress(), records=[r.to_dict() for r in records],
                          perception=self.perception_name)
        before = dict(self.loc)
        pos_before = {a: t.pos.copy() for a, t in self.tracks.items()}

        # 1. Declaration and hooks.
        decl = parse_declaration(code, self.aliases, self.regions) if code is not None else Declaration(
            errors=["no code"])
        self._grasp_hint = decl.args[0] if decl.declared and decl.op != "look" and decl.args else None
        hook_exp, snapshots, events, violations = self._apply_records(turn, records)
        f = getattr(self, "last_record_facts", {}) or {}
        result.facts = {k: list(v) for k, v in f.items()}
        facts = RecordFacts(closed=bool(f.get("closed")), opened=bool(f.get("opened")),
                            grasped=list(f.get("grasped", [])),
                            released=[(r[0], r[1]) for r in f.get("released", [])])
        check_records(decl, facts)
        self.decl = decl
        result.declaration = decl.to_dict()
        expectations: dict[EdgeKey, tuple[bool, str]] = dict(hook_exp)
        if decl.valid and decl.applied:
            for e in decl.effects:
                if e.key[0] in ("above", "lifted"):
                    continue
                expectations[e.key] = (e.after, f"declared {decl.subtask_text}")
        result.expectations = [{"edge": key_text(k), "holds": v[0], "call": v[1]} for k, v in expectations.items()]

        # 2. Perception, robot state, geometry.
        proprio = proprio_from_obs(obs)
        eef = proprio[0] if proprio else None
        dets = self._dedup(self.perception.detect(obs, self.categories))
        seen, result.new_tracks, result.unmatched_tracks, result.rematched = self._associate(dets, eef, turn)
        result.detections = len(dets)
        self._assign_marks()
        self.seen = seen
        self.pixels = self._compute_pixels(obs)
        raw = self._observe_relations(turn, seen, proprio)
        geo = {k: v for k, v in raw.items() if k[0] != "inhand"}
        geo.update(self._region_geo(seen))
        robot = self._robot_preds(raw, proprio)

        # 3. Observe (M5).
        moved = {a for a in self.tracks if a in pos_before and float(np.linalg.norm(self.tracks[a].pos - pos_before[a])) > 0.03}
        moved |= {e.subject for e in events} | set(result.new_tracks)
        moved |= {r[0] for r in f.get("released", [])} | set(f.get("grasped", []))
        statements = self._turn_statements(decl, moved, before)
        result.statements = [statement_text(k) for k in statements]
        by_index = {i: (fn, a) for i, fn, a in f.get("events", [])}
        for ev in event_frames:
            if ev.get("object") is None and ev.get("index") in by_index:  # the object the hooks saw at that call
                ev = {**ev, "after": by_index[ev["index"]][0], "object": by_index[ev["index"]][1]}
            result.event_observes.append(self.observe_event(turn, ev))
        res = None
        rgb = self._rgb(obs)
        image_before = self.last_marked
        if rgb is not None:
            image, shown = self._marked(rgb, sorted(seen))
            self.last_marked = image
            if self.observer is not None and image is not None and statements:
                missing = [m for m, a in self.marks.ordered() if a not in {s[1] for s in shown}]
                marks_now = marks_now_text([(m, a, px) for m, a, px in shown], missing)
                declared = (decl.subtask_text, decl.effect_text or "none") if decl.valid else None
                prompt = observe_turn_prompt(self.instruction, image_before, image, marks_now, declared,
                                             [statement_text(k) for k in statements])
                res = self.observer.observe("turn", prompt, statements, self.aliases)
        values = dict(res.parse.values) if res is not None and res.ok else {}
        # Event-frame answers fill in keys the turn-end Observe could not tell, if the gripper did not
        # close again after the event (the object cannot have been picked up since).
        closes = [r.step_after for r in records if r.fn == "close_gripper" and r.step_after is not None]
        for k, (v, t, step) in self.event_values.items():
            later_close = step is None or any(c > step for c in closes)
            if t == turn and values.get(k) in (None, "?") and v in ("1", "0") and not later_close:
                values[k] = v
        result.fallback = res is None or not res.ok
        result.observe = res.to_dict() if res is not None else None
        self.last_observe = res

        # 4. Fusion, anomalies, Notes.
        self._fuse2(turn, robot, geo, values, expectations, result.fallback, result)
        self._anomalies(turn, values, seen, result)
        self._apply_notes(turn, res, values, result)
        self._update_streaks(statements, values)
        self.obs_now = values
        final = self._snapshot()

        # 5. S1: hook violations (grasp missed / object dropped), declared Effects contradicted.
        s1: list[str] = [f"{v.call}: expected {v.expected}, observed {v.observed}" for v in violations]
        contradicted: list[tuple[EdgeKey, bool]] = []
        if decl.valid and decl.applied:
            for e in decl.effects:
                edge = self.graph.edges.get(e.key)
                if edge is None or self.origin.get(e.key) not in OBSERVED_ORIGINS or e.key in self.anomaly_keys:
                    continue
                if (e.after and edge.p <= cfg.uncertain_low) or (not e.after and edge.p >= cfg.uncertain_high):
                    contradicted.append((e.key, e.after))
                    violations.append(Violation(turn, f"declared {decl.subtask_text}", f"{key_text(e.key)} {int(e.after)}",
                                                f"{int(not e.after)} ({self._label(e.key)})", e.key, e.after))

        # 6. Events (as v1).
        for i in range(len(events) - 1, -1, -1):
            e = events[i]
            if e.source == "exp" and all(x.subject != e.subject for x in events[i + 1:]) \
                    and self.location_text(e.subject, final) == e.after and e.subject in seen:
                e.source = "exp+obs"
        last = snapshots[-1] if snapshots else self.loc
        for a in sorted(final):
            if a in result.new_tracks or a not in last:
                continue
            b, af = self.location_text(a, last), self.location_text(a, final)
            if b != af:
                src = "vdm" if any(self.origin.get(k) == "vdm" for k in self._location_edges(a)) else (
                    "obs" if a in seen else "mem")
                events.append(EstEvent(turn, a, b, af, src))
        self.graph.events.extend(events)
        result.events = events

        # 7. Steps (v1 counting; confirmation needs an observation).
        n_before = len(self.counter.done)
        result.violations = violations
        self.violations += violations  # _withdraw appends to both lists from here on
        if self.goal.branches:
            self._count(turn, snapshots, final, result)
        for idx in range(len(self.counter.done)):
            self.count_turn.setdefault(idx, turn)
        for idx in [i for i in self.count_turn if i >= len(self.counter.done)]:
            self.count_turn.pop(idx)
        if len(self.counter.done) > n_before:
            self.subtasks.step_counted()
        for r in result.rolled_back:
            s1.append(f"{r}: the count was withdrawn (the camera does not show it)")

        # 8. Subtasks.
        judged = self._judge_subtasks(turn, decl, f, before, final, moved)
        result.judged = [s.to_dict() for s in judged]
        result.subtask_progress = any(s.status in ("done", "pending") for s in judged)
        failed = [s for s in judged if s.status == "failed"]
        failed_objs = set()
        for s in failed:
            o = s.obj or "?"
            failed_objs.add(o)
            where = self.location_text(o, final)
            flag = f" ({self.flags[o][0]}, per Notes)" if o in self.flags else ""
            state = f"{o}'s location is unknown" if where == UNKNOWN_LOCATION else f"{o} is {where}"
            s1.append(f"{s.text} failed: {state}{flag}. Retry the {s.op}, not the whole task")
        for key, after in contradicted:
            if key[1] in failed_objs:
                continue
            s1.append(f"declared {key_text(key)} {int(after)} after {decl.subtask_text}, but the "
                      f"{'robot state' if self.origin.get(key) == 'robot' else 'camera'} shows {int(not after)}")
        result.s1_items = s1

        self.loc = final
        self.turn = turn
        result.progress = self.counter.progress()
        result.pending = sorted(self.pending)
        result.snapshots = len(snapshots) + 1
        result.s3_items = self._preconditions()
        result.uncertain, result.s4_items = self._uncertain()
        result.suppress_completion = (not result.progress.known) or bool(result.uncertain) or bool(self.pending)
        self.last2 = result
        self.cot = self._cot(result, [])
        return result

    def observe_event(self, turn: int, ev: dict[str, Any]) -> dict[str, Any]:
        """Optional (``event_frames``): Observe on a recorded frame right after a gripper close / open.

        ``ev``: ``{"rgb": frame, "after": "close_gripper" | "open_gripper", "object": alias, "step": n}``.
        The statements are the event's object in hand and its relations; answers are kept per key for
        this turn and fill in turn-end '?' when the object did not move after."""
        out: dict[str, Any] = {"after": ev.get("after"), "object": ev.get("object"), "step": ev.get("step")}
        o = ev.get("object")
        if self.observer is None or o not in self.tracks or ev.get("rgb") is None:
            out["skipped"] = True
            return out
        keys: list[EdgeKey] = [("inhand", o, None)]
        if ev.get("after") == "open_gripper":
            keys += [self._rel_key(o, s) for s in self.supports() if s != o][: self.cfg.observe_max_statements - 2]
            keys.append(("on", o, TABLE))
        image, shown = self._marked(np.asarray(ev["rgb"]), self.aliases)
        if image is None:
            out["skipped"] = True
            return out
        missing = [m for m, a in self.marks.ordered() if a not in {s[1] for s in shown}]
        prompt = observe_turn_prompt(self.instruction, None, image, marks_now_text([(m, a, px) for m, a, px in shown], missing),
                                     None, [statement_text(k) for k in keys])
        res = self.observer.observe("event", prompt, keys, self.aliases)
        for k, v in res.parse.values.items():
            self.event_values[k] = (v, turn, ev.get("step"))
        out.update(res.to_dict())
        return out

    def _update_streaks(self, statements: Sequence[EdgeKey], values: dict[EdgeKey, str]) -> None:
        for k in statements:
            v = values.get(k)
            if v == "?":
                self.q_streak[k] = self.q_streak.get(k, 0) + 1
            elif v in ("1", "0"):
                self.q_streak[k] = 0

    # --- counting (v1 machinery; v2 observation origins) ----------------------------------------
    def _observed(self, pred: Pred) -> bool | None:
        key = _pred_key(pred)
        edge = self.graph.edges.get(key)
        if edge is None or self.origin.get(key) not in OBSERVED_ORIGINS or key in self.anomaly_keys:
            return None
        if edge.p >= self.cfg.uncertain_high:
            return True
        if edge.p <= self.cfg.uncertain_low:
            return False
        return None

    def _label(self, key: EdgeKey) -> str:
        o = self.origin.get(key, self.graph.edges[key].source if key in self.graph.edges else "prior")
        return {"geo": "obs", "grip": "robot", "expected": "exp", "excl": "mem"}.get(o, o)

    # --- subtasks -------------------------------------------------------------------------------
    def _place_verdict(self, key: EdgeKey) -> tuple[str, str]:
        edge = self.graph.edges.get(key)
        origin = self.origin.get(key)
        if edge is None or origin not in OBSERVED_ORIGINS or key in self.anomaly_keys:
            return "pending", ""
        if edge.p >= self.cfg.uncertain_high:
            return "done", origin
        if edge.p <= self.cfg.uncertain_low:
            return "failed", origin
        return "pending", ""

    def _judge_subtasks(self, turn: int, decl: Declaration, f: dict[str, Any], before: Snapshot, final: Snapshot,
                        moved: set[str]) -> list[Subtask]:
        out: list[Subtask] = []
        lifted = list(f.get("lifted", []))
        releases = list(f.get("released", []))
        # Picks: grasped and lifted (robot).
        for o in lifted:
            src = current_support(before, o)
            out.append(Subtask("pick", (o, src), "done", turn, "robot"))
        for o in f.get("missed", []):
            out.append(Subtask("pick", (o, current_support(before, o)), "failed", turn, "robot"))
        for o in f.get("dropped", []):
            out.append(Subtask("pick", (o, current_support(before, o)), "failed", turn, "robot"))
        # Places: a lifted object released; judged by the turn-end observation of the target relation.
        for k, (o, support, was_lifted) in enumerate(releases):
            if not was_lifted:
                continue
            target = support if support is not None else TABLE
            op = "put_in" if target in self.containers() else "place"
            if decl.valid and decl.op in PLACE_OPS and decl.args and decl.args[0] == o:
                op, target = decl.op, decl.args[1]
            st = Subtask(op, (o, target), "pending", turn, None, declared=decl.valid)
            regrabbed = o in lifted[lifted.index(o) + 1:] if o in lifted else False
            later = any(r[0] == o for r in releases[k + 1:])
            if regrabbed or later:  # picked up again in this turn: the place cannot be checked any more
                st.status, st.by = "done", "api"
            else:
                key = st.effect_key()
                st.status, st.by = self._place_verdict(key) if key else ("pending", "")
            out.append(st)
        # A declared subtask with no matching API activity (e.g. look, or a place that never opened).
        # A declared pick of the object the robot still holds (grasped in an earlier turn).
        unmatched = decl.valid and not any(s.op == decl.op and s.obj == (decl.args[0] if decl.args else None) for s in out)
        if unmatched and decl.op == "pick" and decl.args and self.robot.get("holding") == decl.args[0]:
            args = tuple(decl.args) if len(decl.args) == 2 else (decl.args[0], current_support(before, decl.args[0]))
            out.append(Subtask("pick", args, "done" if self.robot.get("lifted") else "active", turn, "robot", True))
        for s in out:
            if decl.valid and s.op == decl.op and s.obj == (decl.args[0] if decl.args else None):
                s.declared = True
        # Earlier pending places: judged again if their object did not move this turn.
        for old in self.subtasks.pending():
            if old.obj in moved or any(s.same(old) for s in out):
                continue
            key = old.effect_key()
            v, by = self._place_verdict(key) if key else ("pending", "")
            if v != "pending":
                old.status, old.by, old.turn = v, by, turn
                out.append(Subtask(old.op, old.args, v, turn, by, old.declared))
                continue
        for s in out:
            self.subtasks.add(Subtask(s.op, s.args, s.status, s.turn, s.by, s.declared))
        return out

    def plan(self) -> list[list[Subtask]]:
        """Operators for each possible next step (first: the primary branch)."""
        prog = self.counter.progress()
        holder = self.robot.get("holding")
        out = []
        for step in prog.next_steps():
            out.append(expand_step(step, self.loc, self.containers(), holder))
        # The plan that continues with the held object first.
        out.sort(key=lambda pl: 0 if pl and pl[0].op != "pick" else 1)
        return out

    def _preconditions(self) -> list[str]:
        """S3: false preconditions of the next subtask (only confident beliefs)."""
        plans = self.plan()
        if not plans or not plans[0]:
            return []
        nxt = plans[0][0]
        items = []
        for key, want in nxt.preconditions():
            edge = self.graph.edges.get(key)
            if edge is None:
                continue
            p = edge.p
            false = (p <= self.cfg.uncertain_low) if want else (p >= self.cfg.uncertain_high)
            if not false or self.origin.get(key) not in OBSERVED_ORIGINS + ("mem", "exp"):
                continue
            if key[0] == "handempty":
                held = self.robot.get("holding")
                if held is None:
                    continue
                items.append(f"next subtask {nxt.text} needs (handempty), but the gripper holds {held}: "
                             f"put {held} where the goal needs it (or back down) first")
            else:
                items.append(f"next subtask {nxt.text} needs {key_text(key)}, which is false (P {p:.2f} "
                             f"{self._label(key)}): locate {key[1]} first")
        return items

    # --- uncertainty (S4) -----------------------------------------------------------------------
    def _uncertain(self) -> tuple[list[Uncertain], list[str]]:
        out: list[Uncertain] = []
        texts: list[str] = []
        goal_keys = [_pred_key(p) for p in self.goal_relevant()]
        for j in sorted(self.pending):
            step = self.counter.done[j]
            key = _pred_key(step[0])
            prob, src, seen = self.pred_belief(step[0])
            out.append(Uncertain(f"{step_stmt(step)} (step {j + 1}, counted from the API calls, not yet seen)",
                                 prob, src, seen, "unconfirmed", key))
            if self.turn - self.count_turn.get(j, self.turn) + 1 >= self.cfg.pending_turns:
                texts.append(f"{step_stmt(step)} counted from the API calls but not seen by the camera yet — "
                             "move the arm away and look again")
        flagged = {u.key for u in out}
        notes = self.last_observe.parse.notes if self.last_observe is not None else None
        for key in goal_keys:
            if key in flagged:
                continue
            if self.q_streak.get(key, 0) >= self.cfg.unseen_turns:
                e = self.graph.edges.get(key)
                out.append(Uncertain(key_text(key), e.p if e else 0.5, self._label(key) if e else "prior",
                                     self.key_seen.get(key), "unseen", key))
                texts.append(f"{key_text(key)} not visible" + (f" (Notes: {notes})" if notes else "")
                             + " — move the arm away and look again")
                flagged.add(key)
        for key in sorted(self.anomaly_keys, key=str):
            if key in flagged:
                continue
            e = self.graph.edges.get(key)
            a = key[1]
            if a in self.flags:
                kind, note = self.flags[a]
                out.append(Uncertain(key_text(key), e.p if e else 0.5, "vdm", self.key_seen.get(key), "notes", key))
                texts.append(f"{key_text(key)}: {a} may be {kind} (Notes: {note}) — look again before relying on it")
            else:
                g = self._geom(key) or {}
                out.append(Uncertain(key_text(key), e.p if e else 0.5, "vdm", self.key_seen.get(key), "anomaly", key))
                texts.append(f"{key_text(key)} was answered 1 but {key[1]} and {key[2]} are {g.get('d_xy', 0):.2f} m "
                             "apart in the estimate — a mark may be on the wrong object; look again")
            flagged.add(key)
        for a, (kind, note) in self.flags.items():
            if not any(u.key and u.key[1] == a for u in out):
                out.append(Uncertain(f"{a} {kind}", 0.5, "vdm", self.turn, "notes", ("on", a, None)))
                texts.append(f"{a} may be {kind} (Notes: {note})")
        return out, texts

    # --- text -----------------------------------------------------------------------------------
    def delta_text(self, uncertain: bool = False) -> str:
        if not self.goal.branches:
            return "Delta: progress unknown (the goal could not be parsed into steps)"
        p = self.counter.progress()
        if not p.known:
            return f"Delta: progress unknown (the {len(p.done)} counted step(s) match no branch of the goal)"
        done, total, rem = len(p.done), p.total, p.remaining
        rep = p.repeat_count()
        if rep is not None:
            head = f"Delta: {rep[0]}/{rep[1]} repetitions done ({step_stmt(p.branches[0][0])})"
        else:
            head = f"Delta: step {done}/{total} done" if done else f"Delta: step 0/{total}"
        parts = [head]
        if p.overshot:
            parts.append("OVERSHOT: the completed goal state was undone")
        nxt = p.next_steps()
        if nxt:
            texts = [step_stmt(s) for s in nxt]
            parts.append("next: " + (texts[0] if len(texts) == 1 else "one of " + " OR ".join(texts)))
        parts.append(f"remaining {rem}")
        if rem == 0 and not p.overshot:
            parts.append("all steps counted, but NOT all confirmed" if uncertain else "all steps estimated complete")
        if self.pending:
            parts.append(f"{len(self.pending)} counted step(s) not yet confirmed by the camera")
        return " | ".join(parts)

    def subtasks_text(self) -> str:
        plans = self.plan()
        nxt = " -> ".join(s.text for s in plans[0]) if plans and plans[0] else ""
        alt = ""
        if len(plans) > 1:
            alt = "   [or " + " | ".join(" -> ".join(s.text for s in pl) for pl in plans[1:] if pl) + "]"
        hist = self.subtasks.current[-3:]
        if not hist:
            return f"Subtasks (next): {nxt or 'none'}{alt}"
        line = "Subtasks: " + " -> ".join(s.render() for s in hist)
        if nxt:
            line += f" | next: {nxt}{alt}"
        return line

    def observe_line(self) -> str | None:
        res = self.last_observe
        if res is None:
            return f"Observe (t{self.turn}): none (no camera check this turn)"
        if not res.ok:
            return f"Observe (t{self.turn}): failed ({res.error or res.parse.error}); relations from geometry"
        line = f"Observe (t{self.turn}): {res.parse.line(res.statements)}"
        if res.parse.notes:
            line += f" | Notes: {res.parse.notes}"
        return line

    def belief_items(self) -> list[str]:
        items: list[tuple[int, str]] = []
        shown: set[EdgeKey] = set()
        holder = self.robot.get("holding")

        def add(rank: int, key: EdgeKey) -> None:
            e = self.graph.edges.get(key)
            if e is None or key in shown:
                return
            shown.add(key)
            items.append((rank, f"{key_text(key)} {e.p:.2f} {self._label(key)}"))

        for a in self.movables():
            for k in sorted(self._location_edges(a), key=str):
                if k[0] not in ("inhand", "at") and self.graph.edges[k].holds:
                    add(0, k)
        decl_keys = [e.key for e in self.decl.effects] if self.decl is not None and self.decl.valid else []
        for k in decl_keys + [_pred_key(p) for p in self.goal_relevant()]:
            if k in self.graph.edges and k[0] not in ("above", "lifted", "handempty", "inhand"):
                add(1, k)
        if holder:
            add(2, ("inhand", holder, None))
        elif ("handempty", GRIPPER, None) in self.graph.edges:
            add(2, ("handempty", GRIPPER, None))
        items.sort(key=lambda x: x[0])
        return [t for _, t in items]

    def objects_items(self) -> list[str]:
        holds: dict[str, list[str]] = {}
        for a in self.aliases:
            for p, t in self.loc.get(a, frozenset()):
                if p in ("on", "in") and t in self.tracks:
                    holds.setdefault(t, []).append(a)
        out = []
        for a in self.aliases:
            t = self.tracks[a]
            text = f"{a} at ({', '.join(f'{(round(float(v), 2) or 0.0):.2f}' for v in t.pos)})"
            if a in self.flags:
                text += f" {self.location_text(a)} ({self.flags[a][0]}, per Notes)"
            if is_support(t.category.key) and a not in holds:
                text += " empty"
            node = self.graph.nodes.get(a)
            if node is not None and node.last_seen_turn is not None and node.last_seen_turn < self.turn:
                text += f", not seen since t{node.last_seen_turn}"
            out.append(text)
        for r, c in self.region_xy.items():  # estimated: where the front camera looks at the table
            out.append(f"{r} (table region) at ({c[0]:.2f}, {c[1]:.2f}), radius {self.cfg.region_radius:.2f}")
        return out

    def render(self, signals: Sequence[Any] = (), max_chars: int = 3000, uncertain: bool = False) -> str:  # type: ignore[override]
        """Design block. Never cut: title, Subtask, Effect, Delta, Subtasks, Signals. Over budget: Belief
        items dropped from the end, then the Objects line, then the Observe line shortened / dropped."""
        sig = "Signals: " + (" | ".join(f"{s.id} {s.text}" for s in signals) if signals else "none")
        belief = self.belief_items()
        objects = self.objects_items()
        observe = self.observe_line()
        delta = self.delta_text(uncertain)
        subt = self.subtasks_text()

        def compose(belief: list[str], objects: list[str] | None, observe: str | None) -> str:
            lines = []
            if self.turn == 0:
                lines.append(TITLE0)
                lines += [delta, subt]
                if observe:
                    lines.append(observe)
                if belief:
                    lines.append("Belief: " + " | ".join(belief))
            else:
                lines.append(TITLE.format(turn=self.turn))
                d = self.decl
                if d is not None and d.valid:
                    status = ""
                    last = next((s for s in reversed(self.subtasks.current) if s.declared and s.op == d.op), None)
                    if last is not None and last.status == "failed":
                        status = " — FAILED"
                    lines.append(f"Subtask (declared): {d.subtask_text}{status}")
                    lines.append(f"Effect (declared): {d.effect_text or 'none'}")
                elif d is not None and d.declared and d.dropped:
                    lines.append(f"Subtask (declared): {d.subtask_text} — IGNORED ({d.dropped})")
                else:
                    lines.append("Subtask (declared): none")
                if observe:
                    lines.append(observe)
                if belief:
                    lines.append("Belief: " + " | ".join(belief))
                lines += [delta, subt]
            if objects is not None:
                lines.append("Objects: " + " | ".join(objects))
            lines.append(sig)
            return "\n".join(lines)

        text = compose(belief, objects, observe)
        while len(text) > max_chars and len(belief) > 3:
            belief = belief[:-1]
            text = compose(belief, objects, observe)
        if len(text) > max_chars:
            text = compose(belief, None, observe)
        if len(text) > max_chars and observe:
            over = len(text) - max_chars
            short = observe[: max(len(observe) - over - 15, 0)]
            text = compose(belief, None, (short + " ...[truncated]") if short else None)
        if len(text) > max_chars:
            text = compose([], None, None)
        return text

    # --- records --------------------------------------------------------------------------------
    def _cot(self, r: EstTurn2, signals: Sequence[Any]) -> dict[str, Any]:
        d = self.decl
        res = self.last_observe
        return {
            "turn": r.turn,
            "Subtask": d.subtask_text if d is not None and d.valid else None,
            "Effect": d.effect_text if d is not None and d.valid else None,
            "declared": bool(d is not None and d.valid),
            "Observe": res.parse.line(res.statements) if res is not None and res.ok else None,
            "Notes": res.parse.notes if res is not None else None,
            "Belief": " | ".join(self.belief_items()),
            "Delta": self.delta_text(bool(r.uncertain)),
            "Subtasks": self.subtasks_text(),
            "Signals": [f"{s.id} {s.text}" for s in signals],
        }

    def set_signals(self, signals: Sequence[Any]) -> None:
        if self.last2 is not None:
            self.cot = self._cot(self.last2, signals)

    def key_records(self) -> dict[str, dict[str, Any]]:
        out = {}
        for key, e in self.graph.edges.items():
            if key[0] in ("above", "lifted") and not e.holds:
                continue
            if key[0] == "inhand" and not e.holds and key not in self.obs_now:
                continue
            out[key_text(key)] = {
                "P": round(e.p, 3), "source": self._label(key),
                "obs": self.origin.get(key) in OBSERVED_ORIGINS and self.key_seen.get(key) == self.turn,
                "observe": self.obs_now.get(key), "last_seen_turn": self.key_seen.get(key),
                "geom": self._geom(key),
            }
        return out

    def trace(self, kind: str = "turn") -> dict[str, Any]:
        r = self.last2
        nodes = {}
        for a, t in self.tracks.items():
            node = self.graph.nodes.get(a)
            px = self.pixels.get(a)
            nodes[a] = {"cat": t.category.key, "mark": self.marks.mark(a),
                        "pos": [round(float(v), 4) for v in t.pos],
                        "size": [round(float(v), 4) for v in t.size],
                        "last_seen": node.last_seen_turn if node else None,
                        "pixel": [round(px[0], 1), round(px[1], 1)] if px else None,
                        "seen_now": a in self.seen, "flag": self.flags.get(a, (None,))[0]}
        steps = [{"i": i, "step": step_stmt(s), "status": "pending" if i in self.pending else "confirmed",
                  "by": self.confirmed_by.get(i), "turn": self.count_turn.get(i)}
                 for i, s in enumerate(self.counter.done)]
        plans = self.plan()
        return {
            "turn": self.turn, "kind": kind,
            "nodes": nodes, "robot": self.robot, "keys": self.key_records(),
            "goal": {**self.goal.to_dict(), "regions": {k: [round(float(v), 3) for v in c] for k, c in self.region_xy.items()}},
            "steps": steps,
            "progress": {"done": len(self.counter.done), "remaining": self.counter.progress().remaining,
                         "known": self.counter.progress().known},
            "subtasks": [s.to_dict() for s in self.subtasks.all],
            "plan": [[s.text for s in pl] for pl in plans],
            "declaration": self.decl.to_dict() if self.decl is not None else None,
            "observe": (self.last_observe.to_dict() | {"prompt_text": self.last_observe.prompt_text})
            if self.last_observe is not None else None,
            "cot": self.cot,
            "est": r.to_log() if r is not None else None,
        }

    def state(self) -> dict[str, Any]:
        out = super().state()
        out.update(declared=bool(self.decl is not None and self.decl.valid),
                   observe_ok=bool(self.last_observe is not None and self.last_observe.ok),
                   observe_calls=self.observer.calls if self.observer is not None else 0,
                   subtasks_failed=sum(1 for s in self.subtasks.all if s.status == "failed"))
        return out
