"""The BG generator's forward pass (design: 에이전트 루프; mode B = this pass alone, mode A = this pass +
``backward.py``).

Inputs: an :class:`records.Episode` (video + robot state + instruction), a detector, the models.
Per episode: initialize (detection with prompt candidates and re-detection, init LLM -> remember / goal
branches / initial statements, initial Observe), then for every event (gripper, base, keyframe): robot
state -> tracks follow / grasp target / release; detect + track; marked image; agent -> tool calls
(fallback plan on a bad reply); tools (observe, geometry, detect, look, hindsight in mode A); verifier;
goal progress; one :class:`Row` + one trace line.

Never reads ground truth: the episode has none, and nothing here imports ``oracle`` / ``evaluate``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from capx.belief.perception import Category, project_to_pixel
from capx.belief.som import MarkTable, draw_marks, mark_radius, spread_marks
from capx.belief.verify import encode_png

from .agent import Plan, PlanError, default_plan, extract_json, parse_plan
from .events import Event, EventConfig, detect_events
from .geometry import Geometry
from .graph import BeliefGraph, GoalTracker, Step, decided
from .perception import Detection, Detector
from .prompts import agent_prompt, init_prompt, observe_prompt, text_only
from .records import CameraFrame, Episode, FrameRecord
from .tracks import TrackConfig, Tracker, box_distance
from .verifier import Verifier, VerifierConfig
from .vocab import (
    ARMS_ONE,
    OPENABLE_KEYS,
    STATE_PREDS,
    TABLE,
    TOGGLEABLE_KEYS,
    VOCABULARY,
    EdgeKey,
    categories_for,
    category,
    is_container,
    is_support,
    parse_statement,
    parse_values,
    statement_key,
    statement_text,
)


@dataclass
class GenConfig:
    mode: str = "A"  # A: offline (hindsight allowed, then the backward pass), B: online (past frames only)
    camera: str = "agentview"
    view: str = "front"  # prompts.VIEWS key of the main camera
    look_views: dict[str, str] = field(default_factory=dict)  # other cameras -> view word
    relations: tuple[str, ...] = ("on", "in")  # goal / statement relations besides robot state
    extra_categories: tuple[str, ...] = ()
    max_obs: int = 10
    max_init_obs: int = 20
    max_belief_items: int = 12
    lift_dz: float = 0.03  # m: EEF this much above its grasp height -> lifted
    touch_radius: float = 0.10  # m: objects this close to the EEF at a gripper event count as touched
    agent_on: tuple[str, ...] = ("grasp", "close", "release", "close_empty", "open", "keyframe", "base_stop", "end")
    # ``close`` (grasp unknown from the gripper width): the nearest track is held provisionally and confirmed or
    # retracted once the EEF has moved ``verify_move`` m: confirmed if it is detected at the hand, retracted if it
    # is detected where it was (left behind).
    verify_move: float = 0.03  # m of EEF motion before a provisional grasp is checked (LIBERO lifts: 0.03-0.07 m)
    verify_sep: float = 0.02  # m: the "moved with the hand" and "stayed" positions must be this far apart
    verify_sep_px: float = 20.0  # the same on the 2D path (pixels)
    # bggen-fix3 C1: a provisional grasp whose object is found at its old place at the release is still a grasp when
    # the hand rose at least ``carry_lift`` m while closed and released within ``put_back_radius`` m (horizontal) of
    # where it closed: the object was lifted and put back (t02 demo_2: bowl lifted 4.3 cm and set on the same
    # plate). 0.03 m = ``lift_dz`` (the R3 ``lifted`` threshold) = ``verify_move``; LIBERO lifts are 0.03-0.07 m.
    carry_lift: float = 0.03
    put_back_radius: float = 0.08
    # bggen-fix3 A1: geometry of the released object vs the nearest supports / containers marked in the release image
    release_geo_k: int = 2
    release_geo_radius: float = 0.30  # m (3D path; the 2D path takes the nearest ``release_geo_k`` by pixel)
    # bggen-fix3 A3: R10 re-check of an 'in' conflict at the same event: another camera, then (mode A) a later
    # frame of the same quiet interval this many seconds after the event
    recheck_look: bool = True
    recheck_hindsight_s: float = 1.0
    cam_still_m: float = 0.01  # 2D path: the camera counts as unmoved since the close within 1 cm and 1 deg
    cam_still_deg: float = 1.0
    agent_retry: int = 0  # extra agent queries after a format failure before the fallback plan
    # output limit per role (the client re-queries a cut / empty reply once with twice this); the harness baseline
    max_tokens: dict[str, int] = field(default_factory=lambda: {"init": 32768, "agent": 32768, "vdm": 32768})
    # categories the instruction names that the first image does not show -> expected objects (bound on first detection)
    expect_missing: bool = True
    init_retry: int = 1
    cook_s: float | None = None  # process model: in(X, C) and toggled_on(C) this long -> cooked(X) 0.7 mem
    redetect_after: int = 2  # a category undetected this many events in a row tries its next prompt
    navigation_subtask: bool = False  # backward: rows during base motion before a segment -> move_to(X)
    save_keyframes: bool = True
    events: EventConfig = field(default_factory=EventConfig)
    tracks: TrackConfig = field(default_factory=TrackConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)

    @property
    def offline(self) -> bool:
        return self.mode.upper() == "A"


@dataclass
class Row:
    """One event / keyframe of the forward pass (all keys are edge keys; the serializer renders them)."""

    t: int
    kind: str
    arm: str | None
    obj: str | None
    observed: list[tuple[EdgeKey, str]]  # asked statements and answers (after R6), in order
    belief: dict[EdgeKey, tuple[float, str]]  # full graph after the verifier
    changed: list[EdgeKey]
    delta: str
    goal_done: int
    goal_total: int
    subtask: str  # causal estimate (mode B); the backward pass replaces it in mode A
    touched: list[str]
    held: dict[str, str | None]
    hindsight: dict[EdgeKey, str] = field(default_factory=dict)
    s4: list[str] = field(default_factory=list)
    effect: list[tuple[EdgeKey, int, int]] = field(default_factory=list)  # set by the backward pass
    notes: list[str] = field(default_factory=list)
    keyframe: str | None = None
    # R10 (bggen-fix3 P2): Observe answers in ``observed`` that the verifier did not adopt (the first 0 of a conflict
    # left pending or overturned), and the adopted value of a re-check with another camera at this event
    rechecked: dict[EdgeKey, str] = field(default_factory=dict)
    rejected: list[EdgeKey] = field(default_factory=list)
    # set by the backward B1b merge (evaluation provenance): final key -> merged alias it came from; displaced entries
    b1b_origin: dict[EdgeKey, str] = field(default_factory=dict)
    b1b_shadowed: list[tuple[EdgeKey, tuple[float, str], str | None]] = field(default_factory=list)

    def copy(self) -> Row:
        return Row(**{k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                      for k, v in self.__dict__.items()})


def _event_text(e: Event, held: dict[str, str | None]) -> str:
    arm = f" ({e.arm} arm)" if e.arm else ""
    if e.kind == "grasp":
        return f"the gripper{arm} closed on {e.obj or 'an unidentified object'}"
    if e.kind == "close":
        return f"the gripper{arm} closed, probably on {e.obj or 'an unidentified object'} (confirmed when it moves with the hand)"
    if e.kind == "release":
        return f"the gripper{arm} opened and let go of {e.obj or 'the held object'}"
    if e.kind == "close_empty":
        return f"the gripper{arm} closed on nothing" + (f" next to {e.obj}" if e.obj else "")
    if e.kind == "open":
        return f"the gripper{arm} opened (it held nothing)"
    if e.kind == "base_start":
        return "the robot base started moving"
    if e.kind == "base_stop":
        return "the robot base stopped"
    hold = ", ".join(f"{a or 'gripper'} holds {o}" for a, o in held.items() if o) or "nothing held"
    return f"periodic check, no gripper event ({hold})"


class Generator:
    def __init__(self, episode: Episode, detector: Detector, models: Any, cfg: GenConfig | None = None,
                 out_dir: str | Path | None = None) -> None:
        self.ep = episode
        self.det = detector
        self.models = models
        self.cfg = cfg or GenConfig()
        self.out = Path(out_dir) if out_dir is not None else None
        self.arm_keys = list(episode.arms)
        self.arms: tuple[str | None, ...] = tuple(self.arm_keys) if len(self.arm_keys) > 1 else ARMS_ONE
        self.graph = BeliefGraph()
        self.verifier = Verifier(self.graph, self.cfg.verifier)
        self.tracker = Tracker(self.cfg.tracks)
        self.geo = Geometry(self.tracker)
        self.goal = GoalTracker()
        self.marks = MarkTable()
        self.categories: list[Category] = []
        self.tried: dict[str, set[str]] = {}
        self.missing_streak: dict[str, int] = {}
        self.rows: list[Row] = []
        self.trace: list[dict[str, Any]] = []
        self.redetects: list[dict[str, Any]] = []
        self.touched_at: dict[str, int] = {}
        self.grasp_z: dict[str, float] = {}
        self.pending_grasp: dict[str, dict[str, Any]] = {}  # arm -> provisional grasp of a ``close`` event
        self.grasp_checks: list[dict[str, Any]] = []
        self.cook_since: dict[EdgeKey, tuple[str, int]] = {}  # cooked key -> (container, start)
        self.prev_image: dict[str, str] = {}  # cam -> last marked image (data URL)
        self.events: list[Event] = []
        self.agent_failures = 0
        self.init_info: dict[str, Any] = {}

    # --- helpers ---------------------------------------------------------------------------------
    def _arm_name(self, arm: str | None) -> str:
        return arm if arm is not None else self.arm_keys[0]

    def _arm_label(self, arm_key: str) -> str | None:
        return arm_key if len(self.arm_keys) > 1 else None

    def _cat(self, alias: str) -> str:
        return self.tracker.tracks[alias].category if alias in self.tracker.tracks else alias

    def _names(self) -> list[str]:
        return self.tracker.aliases()

    def _query(self, role: str, messages: list[dict[str, Any]]) -> tuple[str, Any, str | None, dict[str, Any]]:
        """(content, usage, error, meta). meta = finish_reason, reasoning tokens, every attempt (the client
        re-queries a cut / empty reply once with a larger limit) and ``incomplete`` (still cut / empty). An
        empty content is an error (the caller retries or falls back); a cut one is still parsed."""
        client = getattr(self.models, role)
        try:
            out = client.query(messages)
        except Exception as exc:
            return "", None, str(exc), {"finish_reason": None, "incomplete": "error"}
        text = str(out.get("content") or "")
        meta = {"finish_reason": out.get("finish_reason"), "reasoning_tokens": out.get("reasoning_tokens"),
                "incomplete": out.get("incomplete"), "attempts": out.get("attempts")}
        err = None
        if not text.strip():
            err = f"empty reply (finish_reason={out.get('finish_reason')}, after {len(out.get('attempts') or [1])} attempt(s))"
        return text, out.get("usage"), err, meta

    def _detect(self, cam: CameraFrame, cats: Sequence[Category], t: int) -> list[Detection]:
        dets = list(self.det.detect(cam, cats))
        # categories with nothing detected: the next prompt candidate (t = 0: all of them, later: after a streak)
        found = {d.category for d in dets}
        for c in list(cats):
            if c.key in found:
                self.missing_streak[c.key] = 0
                continue
            self.missing_streak[c.key] = self.missing_streak.get(c.key, 0) + 1
            if t > 0 and self.missing_streak[c.key] < self.cfg.redetect_after:
                continue
            for text in c.candidates:
                if text in self.tried.setdefault(c.key, {c.sam3_text}):
                    continue
                self.tried[c.key].add(text)
                got = [d for d in self.det.detect(cam, [c.with_query(text)]) if d.category == c.key]
                self.redetects.append({"t": t, "category": c.key, "prompt": text, "detections": len(got)})
                if got:
                    self._set_query(c.key, text)
                    dets += got
                    self.missing_streak[c.key] = 0
                    break
                if t > 0:
                    break  # later events: one new candidate per streak
        return dets

    def _detect_categories(self) -> list[str]:
        """Categories the agent may ``detect``: the instruction's / config's / init's categories (``self.categories``)
        and so every existing track's category (tracks only come from these, expected objects included). Others
        (e.g. bowl / cup in a B1K kitchen) are not offered in the prompt and are dropped by the parser."""
        return list(dict.fromkeys(c.key for c in self.categories))

    def _set_query(self, key: str, text: str) -> None:
        self.categories = [c.with_query(text) if c.key == key else c for c in self.categories]

    def _img_base(self, cam: CameraFrame) -> np.ndarray | None:
        """world <- base at the frame the image comes from (an image may be older than the event)."""
        return self.ep.robot(cam.t).base_pose

    def _to_img_base(self, cam: CameraFrame, p_world: np.ndarray) -> np.ndarray:
        b = self._img_base(cam)
        return np.asarray(p_world, float) if b is None else (np.linalg.inv(b) @ np.array([*p_world, 1.0]))[:3]

    def _eef_pixel(self, cam: CameraFrame, fr: FrameRecord, arm_key: str) -> tuple[float, float] | None:
        if cam.K is None or cam.T_base_cam is None:
            return None
        return project_to_pixel(self._to_img_base(cam, fr.eef_world(arm_key)), cam.K, cam.T_base_cam)

    def _track_pixel(self, cam: CameraFrame, alias: str, fr: FrameRecord) -> tuple[float, float] | None:
        tr = self.tracker.tracks[alias]
        if tr.seen_now and cam.name in tr.pixel:
            return tr.pixel[cam.name]
        if tr.pos is not None and cam.K is not None and cam.T_base_cam is not None:
            return project_to_pixel(self._to_img_base(cam, tr.pos), cam.K, cam.T_base_cam)
        return None

    def _marked(self, cam: CameraFrame, fr: FrameRecord, seen_only: bool = True) -> tuple[np.ndarray, str, list[str]]:
        """Marked image: only tracks detected in THIS image get a mark (verifier rule), and no mark of a
        non-held object right at the fingertip's pixel (the arm covers it; bg_est2 rule)."""
        h, w = cam.rgb.shape[:2]
        held = {a for a in (self.tracker.held(k) for k in self.arm_keys) if a}
        grip = [p for k in self.arm_keys if (p := self._eef_pixel(cam, fr, k)) is not None]
        rad = mark_radius(cam.rgb.shape) * 2.5
        shown: list[tuple[str, tuple[float, float]]] = []
        missing, near = [], []
        for mark, alias in self.marks.ordered():
            if alias not in self.tracker.tracks:
                continue
            tr = self.tracker.tracks[alias]
            px = tr.pixel.get(cam.name) if tr.seen_now else None
            if px is None and not seen_only:
                px = self._track_pixel(cam, alias, fr)
            if px is None or not (0 <= px[0] < w and 0 <= px[1] < h):
                missing.append(mark)
                continue
            if alias not in held and any(np.hypot(px[0] - g[0], px[1] - g[1]) < rad for g in grip):
                near.append(mark)
                continue
            shown.append((mark, px))
        shown = spread_marks(shown, cam.rgb.shape)
        img = draw_marks(cam.rgb, shown)
        parts = [f"{m} = {self.marks.alias(m)} ({u:.0f}, {v:.0f})" for m, (u, v) in shown]
        if missing:
            parts.append(f"{', '.join(missing)}: not detected")
        if near:
            parts.append(f"{', '.join(near)}: under the gripper, not marked")
        return img, " | ".join(parts) or "none detected", [self.marks.alias(m) for m, _ in shown]

    def _save_keyframe(self, t: int, cam: str, img: np.ndarray) -> str | None:
        if self.out is None or not self.cfg.save_keyframes:
            return None
        from PIL import Image

        d = self.out / "keyframes"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{t:06d}_{cam}.png"
        Image.fromarray(img).save(d / name)
        return f"keyframes/{name}"

    def _robot_state(self, fr: FrameRecord, t: int) -> set[EdgeKey]:
        """R3: inhand / handempty / lifted from the robot state."""
        keys: set[EdgeKey] = set()
        for k in self.arm_keys:
            lab = self._arm_label(k)
            held = self.tracker.held(k)
            he = ("handempty", lab, None)
            self.verifier.robot(he, held is None, t)
            keys.add(he)
            for a in self.tracker.tracks:
                ik = ("inhand", a, lab)
                if held == a or ik in self.graph.edges:
                    self.verifier.robot(ik, held == a, t)
                    keys.add(ik)
            if held is not None:
                lk = ("lifted", held, None)
                z0 = self.grasp_z.get(k)
                lifted = z0 is not None and fr.eef_world(k)[2] - z0 > self.cfg.lift_dz
                self.verifier.robot(lk, lifted, t)
                keys.add(lk)
        held_now = {self.tracker.held(k) for k in self.arm_keys}
        for a in self.tracker.tracks:
            lk = ("lifted", a, None)
            if lk in self.graph.edges and a not in held_now:
                self.verifier.robot(lk, False, t)
                keys.add(lk)
        # robot state owns a location edge only while the object is in the hand
        for k, ed in self.graph.edges.items():
            if k[0] in ("on", "in", "at") and ed.source == "robot" and k[1] not in held_now:
                ed.source = "mem"
        return keys

    def _support_of(self, alias: str) -> str | None:
        best = max(((k, e) for k, e in self.graph.edges.items() if k[1] == alias and k[0] in ("on", "in")),
                   key=lambda x: x[1].p, default=None)
        return best[0][2] if best is not None and best[1].p >= 0.5 else None

    def _hidden(self) -> set[str]:
        """Objects believed inside a container believed closed (R6)."""
        out = set()
        for k, e in self.graph.edges.items():
            if k[0] == "in" and e.p >= 0.5 and k[2] is not None and (self.graph.p(("open", k[2], None)) or 0.5) <= 0.3:
                out.add(k[1])
        return out

    def _rel_key(self, subj: str, target: str) -> EdgeKey:
        if target == TABLE:
            return ("on", subj, TABLE)
        return ("in" if is_container(self._cat(target)) else "on", subj, target)

    def _suggestions(self, e: Event) -> tuple[list[EdgeKey], list[tuple[str, str]]]:
        """Default-plan statements: relations of the objects around the event, re-checks, goal relations."""
        out: list[EdgeKey] = list(self.verifier.recheck_keys())
        geo: list[tuple[str, str]] = []
        focus = [a for a in [e.obj] + [self.tracker.held(k) for k in self.arm_keys] if a]
        if e.kind in ("keyframe", "end", "base_stop") and self.tracker.last_released:
            focus.append(self.tracker.last_released)
        for a in dict.fromkeys(focus):
            if a not in self.tracker.tracks:
                continue
            if self._cat(a) in OPENABLE_KEYS:
                out.append(("open", a, None))
                if self._cat(a) in TOGGLEABLE_KEYS:
                    out.append(("toggled_on", a, None))
                continue
            if any(self.tracker.held(k) == a for k in self.arm_keys):
                continue  # in the hand: robot state says where it is
            for s in self.tracker.supports():
                if s != a:
                    out.append(self._rel_key(a, s))
                    geo.append((a, s))
            out.append(("on", a, TABLE))
        out += self.goal.pending(self.graph) + self.goal.next_keys()
        out += [k for k, ed in self.graph.edges.items() if 0.3 < ed.p < 0.7 and k[0] in ("on", "in", "open")]
        return [k for k in dict.fromkeys(out) if k[1] in self.tracker.tracks or k[0] == "handempty"], geo

    # --- initialization ---------------------------------------------------------------------------
    def _expected_lines(self) -> list[str]:
        return [f"{a} ({next((c.prompt for c in self.categories if c.key == self.tracker.tracks[a].category), '?')})"
                for a in self.tracker.expected()]

    def _object_lines(self, cam: CameraFrame) -> list[str]:
        h, w = cam.rgb.shape[:2]
        lines = []
        for mark, alias in self.marks.ordered():
            tr = self.tracker.tracks[alias]
            if tr.expected:
                continue
            px = tr.pixel.get(cam.name)
            where = ""
            if px is not None:
                hor = ("left side", "left of center", "center", "right of center", "right side")[min(int(px[0] / w * 5), 4)]
                ver = ("top (far from the camera)", "middle", "bottom (near the camera)")[min(int(px[1] / h * 3), 2)]
                where = f"; {hor} of the image, {ver}"
            xyz = "" if tr.pos is None else f", xyz ({tr.pos[0]:.2f}, {tr.pos[1]:.2f}, {tr.pos[2]:.2f})"
            cat = next((c.prompt for c in self.categories if c.key == tr.category), tr.category)
            pxs = "" if px is None else f"mark at pixel ({px[0]:.0f}, {px[1]:.0f})"
            lines.append(f"- {mark} = {alias} ({cat}): {pxs}{xyz}{where}")
        return lines

    _NAME = re.compile(r"^([a-z][a-z0-9]*?)_([A-Z]{1,2})$", re.I)

    def _new_names(self, data: dict[str, Any]) -> dict[str, str]:
        """Names in the init reply that are not tracks yet but have the alias form of a known category
        (``hotdog_B``): not-yet-visible objects -> {alias: category}. They become expected tracks instead of
        making the goal unparsable (B1K t=0 sees only the countertop)."""
        known = {n.lower() for n in self._names()} | {TABLE}
        cats = {c.key for c in self.categories} | {c.key for c in VOCABULARY}
        args: list[str] = []
        for b in data.get("goal_branches") or []:
            for s in b if isinstance(b, list) else []:
                preds = [s] if isinstance(s, list) and s and isinstance(s[0], str) else s
                for p in preds if isinstance(preds, list) else []:
                    if isinstance(p, list):
                        args += [str(x) for x in p[1:]]
        for s in data.get("observe") or []:
            args += re.findall(r"[A-Za-z][A-Za-z0-9]*_[A-Za-z]{1,2}\b", str(s))
        out: dict[str, str] = {}
        for a in args:
            m = self._NAME.match(a)
            if m is None or a.lower() in known:
                continue
            cat = m.group(1).lower()
            if cat in cats:
                out.setdefault(f"{cat}_{m.group(2).upper()}", cat)
        return out

    def _parse_init(self, text: str) -> dict[str, Any]:
        data = extract_json(text)
        if not isinstance(data, dict):
            raise PlanError("reply is not a JSON object")
        new = self._new_names(data)
        names = self._names() + list(new)
        branches: list[list[Step]] = []
        for b in data.get("goal_branches") or []:
            if not isinstance(b, list) or not b:
                raise PlanError(f"branch {b!r} is not a non-empty list")
            steps: list[Step] = []
            for s in b:
                preds = [s] if isinstance(s, list) and s and isinstance(s[0], str) else s
                keys = []
                for p in preds if isinstance(preds, list) else []:
                    if not isinstance(p, list) or not p:
                        raise PlanError(f"relation {p!r} is not a list")
                    k = statement_key(str(p[0]), [str(x) for x in p[1:]], names, self.arms)
                    if k is None or k[0] not in self.cfg.relations:
                        raise PlanError(f"relation {p!r}: unknown name or relation (use {list(self.cfg.relations)})")
                    keys.append(k)
                if not keys:
                    raise PlanError(f"step {s!r} has no relation")
                steps.append(tuple(keys))
            branches.append(steps)
        obs = []
        for s in data.get("observe") or []:
            k = parse_statement(str(s), names, self.arms)
            if k is not None and k not in obs:
                obs.append(k)
        remember = [str(x)[:200] for x in data.get("remember") or [] if isinstance(x, (str, int, float))]
        used = {x for b in branches for s in b for k in s for x in k[1:]} | {x for k in obs for x in k[1:]}
        return {"branches": branches, "observe": obs, "remember": remember,
                "expected": {a: c for a, c in new.items() if a in used}}

    def _default_init_statements(self) -> list[EdgeKey]:
        out = []
        for a in self.tracker.movables():
            for s in self.tracker.supports():
                out.append(self._rel_key(a, s))
        for a, tr in self.tracker.tracks.items():
            if tr.category in OPENABLE_KEYS and "open" in self.cfg.relations:
                out.append(("open", a, None))
        return out

    def _normalize_goal(self, branches: list[list[Step]]) -> list[list[Step]]:
        """Each step keeps only the relations that BECOME true (bg_est2 rule): already-true relations are
        dropped from a step that has a new one."""
        out = []
        for b in branches:
            state = {k for k, e in self.graph.edges.items() if e.p >= 0.5}
            steps = []
            for s in b:
                new = tuple(k for k in s if k not in state)
                step = new or s
                for k in step:
                    if k[0] in ("on", "in"):
                        state -= {x for x in state if x[1] == k[1] and x[0] in ("on", "in")}
                    state.add(k)
                steps.append(step)
            if steps not in out:
                out.append(steps)
        return out

    def initialize(self) -> Row:
        cfg = self.cfg
        fr = self.ep.frame(0, [cfg.camera])
        cam = fr.cams[cfg.camera]
        self.categories = categories_for(self.ep.instruction, cfg.extra_categories)
        for c in self.categories:
            self.tried[c.key] = {c.sam3_text}
        dets = self._detect(cam, self.categories, 0)
        self.tracker.initialize(0, cam.name, dets, self._img_base(cam))
        # categories the task names that the first image does not show (B1K: fridge / microwave / hot dog are out
        # of view at t=0): one expected object each, bound to its first detection later
        if cfg.expect_missing:
            seen = {d.category for d in dets}
            for c in self.categories:
                if c.key not in seen and c.key not in cfg.extra_categories:
                    self.tracker.expect(c.key)
        self.geo.observe_sizes()
        for alias in sorted(self.tracker.tracks):
            self.marks.assign(alias, self.tracker.tracks[alias].category)
        img, marks_text, shown = self._marked(cam, fr)
        url = encode_png(img)
        self.prev_image[cam.name] = url
        kf = self._save_keyframe(0, cam.name, img)
        lines = self._object_lines(cam)
        msgs = init_prompt(self.ep.instruction, url, lines, cfg.view, cfg.relations, cfg.max_init_obs,
                           expected=self._expected_lines())
        info: dict[str, Any] = {"prompt": text_only(msgs), "replies": [], "errors": [], "attempts": [], "categories":
                                {c.key: c.sam3_text for c in self.categories}, "detections": [d.to_dict() for d in dets]}
        parsed = None
        for _ in range(cfg.init_retry + 1):
            text, usage, err, meta = self._query("init", msgs)
            info["replies"].append(text)
            info["attempts"].append({"usage": usage, **meta})
            if err:
                info["errors"].append(err)
                continue
            try:
                parsed = self._parse_init(text)
                break
            except PlanError as exc:
                info["errors"].append(str(exc))
                msgs = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": [
                    {"type": "text", "text": f"That reply could not be used: {exc}. Answer again with JSON only."}]}]
        if parsed is None:
            parsed = {"branches": [], "observe": [], "remember": [], "expected": {}}
            info["fallback"] = True
            # never silently "no goal": summary / report / CLI show it loudly
            info["failed"] = "init failed: no goal (" + ("; ".join(info["errors"][-2:]) or "no usable reply") + ")"
        elif not parsed["branches"]:
            info["failed"] = "init reply has no goal branches"
        for alias, cat in parsed.get("expected", {}).items():
            self.tracker.expect(cat, alias)
            self.marks.assign(alias, cat)
            if all(c.key != cat for c in self.categories) and (vc := category(cat)) is not None:
                vc = replace(vc, query=vc.query or vc.candidates[0])
                self.categories.append(vc)  # init named a category the instruction parser missed: detect it too
                self.tried[cat] = {vc.sam3_text}
        info["expected"] = self.tracker.expected()
        if cfg.tracks.cap_from_init and parsed["branches"]:
            # B3: a category the init's goal names has as many objects as it has names after init (detected at t=0 +
            # expected); further detections of it are one of those (``Tracker._identify`` phase 4)
            named = {self._cat(x) for b in parsed["branches"] for s in b for k in s for x in k[1:]
                     if x is not None and x in self.tracker.tracks}
            caps = {c: len(self.tracker.named(c)) for c in sorted(named)}
            self.tracker.set_caps(caps, "init names")
            info["caps"] = dict(self.tracker.caps)
        self.graph.remember = parsed["remember"]
        statements = parsed["observe"] or self._default_init_statements()
        statements = statements[: cfg.max_init_obs]
        self.verifier.begin()
        robot_keys = self._robot_state(fr, 0)
        observed: list[tuple[EdgeKey, str]] = []
        accepted: set[EdgeKey] = set()
        if statements:
            vals, obs_info = self._observe(cam, None, url, marks_text, "the task starts (first image)", statements)
            info["observe"] = obs_info
            geom = self._geometry_for(vals, cam.name)
            accepted = self.verifier.observe(vals, 0, {}, geom, self._hidden(), self._categories_map())
            observed = [(k, vals.get(k, "?")) for k in statements]
        # remembered facts: where each movable started, which supports were empty
        for a in self.tracker.movables():
            s = self._support_of(a)
            if s is not None:
                self.graph.notes[f"orig({a})"] = s
        empty = [s for s in self.tracker.supports() if not self.tracker.tracks[s].expected and not any(
            k[2] == s and k[0] in ("on", "in") and e.p >= 0.5 for k, e in self.graph.edges.items())]
        if empty:
            self.graph.notes["empty_at_start"] = ",".join(empty)
        self.goal = GoalTracker(self._normalize_goal(parsed["branches"]))
        self.goal.update(self.graph, 0)
        self.graph.age(accepted, robot_keys)
        info["goal_branches"] = [[[statement_text(k) for k in s] for s in b] for b in self.goal.branches]
        info["remember"] = self.graph.remember
        info["notes"] = dict(self.graph.notes)
        self.init_info = info
        start = Event(0, "start")
        row = self._row(start, observed, set(), {}, kf)
        self._trace(start, row, {"init": info, "marks": marks_text, "decisions": [d.to_dict() for d in self.verifier.decisions],
                                 "tracks": [t.to_dict() for t in self.tracker.tracks.values()]})
        return row

    # --- tools ------------------------------------------------------------------------------------
    def _categories_map(self) -> dict[str, str]:
        return {a: t.category for a, t in self.tracker.tracks.items()}

    def _observe(self, cam: CameraFrame, before: str | None, now: str, marks_text: str, event_text: str,
                 statements: Sequence[EdgeKey], view: str | None = None) -> tuple[dict[EdgeKey, str], dict[str, Any]]:
        stmts = [statement_text(k) for k in statements]
        msgs = observe_prompt(self.ep.instruction, before, now, marks_text, event_text, stmts, view or self.cfg.view)
        text, usage, err, meta = self._query("vdm", msgs)
        info: dict[str, Any] = {"statements": stmts, "reply": text, "error": err, "usage": usage, **meta}
        if err:
            return {}, info
        vals, notes, bad = parse_values(text, statements, self._names(), self.arms)
        info.update(notes=notes, unreadable=bad, parsed={statement_text(k): v for k, v in vals.items()})
        if not vals:
            info["error"] = "no asked statement could be read"
        return vals, info

    def _geometry_for(self, vals: dict[EdgeKey, str], cam: str, pairs: Sequence[tuple[str, str]] = (),
                      release: dict[str, np.ndarray] | None = None) -> dict[EdgeKey, dict[str, Any]]:
        out = {}
        keys = [k for k in vals if k[0] in ("on", "in")] + [self._rel_key(a, b) for a, b in pairs]
        for k in dict.fromkeys(keys):
            g = self.geo.pair(k, cam, (release or {}).get(k[1]))
            if g is not None:
                out[k] = g
        return out

    def _hindsight(self, e: Event, keys: Sequence[EdgeKey], after_s: float, next_gripper_t: int) -> dict[str, Any]:
        """Mode A only: Observe a later frame of the same quiet interval (before the next gripper event).
        Read-only: the tracker is not updated and the graph is not changed (backward pass evidence)."""
        t2 = min(e.t + max(int(round(after_s * self.ep.fps)), 1), next_gripper_t - 1, self.ep.n_frames - 1)
        if t2 <= e.t:
            return {"t": None, "values": {}, "why": "no quiet frame before the next gripper event"}
        fr = self.ep.frame(t2, [self.cfg.camera], causal=True)  # the latest image at or before t2 (< next event)
        cam = fr.cams.get(self.cfg.camera)
        if cam is None or cam.t <= e.t:
            return {"t": None, "values": {}, "why": "no stored image after the event and before the next gripper event"}
        dets = self.det.detect(cam, self.categories)
        shown = []
        for mark, alias in self.marks.ordered():
            tr = self.tracker.tracks.get(alias)
            if tr is None or cam.name not in tr.pixel:
                continue
            ref = tr.pixel[cam.name]
            near = min((d for d in dets if d.category == tr.category),
                       key=lambda d: np.hypot(d.pixel[0] - ref[0], d.pixel[1] - ref[1]), default=None)
            if near is not None and np.hypot(near.pixel[0] - ref[0], near.pixel[1] - ref[1]) < self.cfg.tracks.assoc_px * 2:
                shown.append((mark, near.pixel))
        shown = spread_marks(shown, cam.rgb.shape)
        img = draw_marks(cam.rgb, shown)
        marks_text = " | ".join(f"{m} = {self.marks.alias(m)} ({u:.0f}, {v:.0f})" for m, (u, v) in shown) or "none"
        vals, info = self._observe(cam, None, encode_png(img), marks_text, "a moment later, nothing touched", keys)
        return {"t": cam.t, "requested_t": t2, "values": {statement_text(k): v for k, v in vals.items()}, "raw": vals,
                "observe": info}

    # --- one event --------------------------------------------------------------------------------
    def step(self, e: Event, next_gripper_t: int) -> Row:
        cfg = self.cfg
        cams = [cfg.camera] + [c for c in self.ep.cameras if c != cfg.camera and c in cfg.look_views]
        fr = self.ep.frame(e.t, cams)
        self.verifier.begin()
        arm_key = self._arm_name(e.arm) if e.is_gripper else None
        rec: dict[str, Any] = {}
        touched_now: set[str] = set()
        release_pend: dict[str, Any] | None = None
        for k in self.arm_keys:
            self.tracker.follow(k, fr.eef_world(k))
        self._update_lift(e.t)
        main = fr.cams.get(cfg.camera)
        # robot-state events first: grasp target needs the tracks as they were before this frame's detection
        if e.kind in ("grasp", "close") and arm_key is not None:
            px = self._eef_pixel(main, fr, arm_key) if main is not None else None
            target, dist = self.tracker.grasp_target(fr.eef_world(arm_key), cfg.camera, px)
            rec["grasp"] = {"target": target, "distance": dist}
            if target is not None:
                self._take(e, arm_key, fr, main, target, px)
        elif e.kind in ("release", "open") and arm_key is not None:
            px = self._eef_pixel(main, fr, arm_key) if main is not None else None
            e.obj = self.tracker.release(arm_key, e.t, fr.eef_world(arm_key), px)
            self.grasp_z.pop(arm_key, None)
            release_pend = self.pending_grasp.pop(arm_key, None)  # resolved with this frame's detections below
            if e.obj:
                self.graph.events.append(f"robot -release-> {e.obj} @{e.t}")
        elif e.kind == "close_empty" and arm_key is not None:
            target, _ = self.tracker.grasp_target(fr.eef_world(arm_key), cfg.camera, None)
            e.obj = target  # pressed / touched, not held
        if e.is_gripper and arm_key is not None:
            eef = fr.eef_world(arm_key)
            for a, tr in self.tracker.tracks.items():
                if tr.pos is not None and np.linalg.norm(tr.pos - eef) <= cfg.touch_radius:
                    touched_now.add(a)
            if e.obj:
                touched_now.add(e.obj)
        touched_now |= {a for a in (self.tracker.held(k) for k in self.arm_keys) if a}
        for a in touched_now:
            self.touched_at[a] = e.t
        robot_keys = self._robot_state(fr, e.t)
        for a in {self.tracker.held(k) for k in self.arm_keys} - {None}:
            self.verifier.held_off_supports(a, e.t)  # every event while held, not only at the grasp
        # detection + tracking
        if main is None:
            rec["error"] = f"no image of camera {cfg.camera}"
            if release_pend is not None and arm_key is not None:  # cannot be checked: unverified, never dropped
                rec["grasp_verify"] = [self._mark_unverified(arm_key, release_pend, e, "no release image", {})]
            row = self._row(e, [], touched_now, {}, None)
            self._trace(e, row, rec)
            return row
        dets = self._detect(main, self.categories, e.t)
        checks = self._verify_grasps(fr, main, dets, e.t)
        if release_pend is not None and arm_key is not None:
            checks.append(self._resolve_release(arm_key, release_pend, e, fr, main, dets))
        if checks:
            rec["grasp_verify"] = checks
            if any(c["result"] == "retracted" for c in checks):
                robot_keys |= self._robot_state(fr, e.t)
        rec["track"] = self.tracker.update(e.t, main.name, dets, self._img_base(main), [c.key for c in self.categories])
        if e.kind in ("grasp", "close") and arm_key is not None and e.obj is None:
            # B1: nothing tracked within reach before this frame's detection (stale / no position): try again with
            # the positions just detected, so the held object follows the hand instead of staying behind
            px = self._eef_pixel(main, fr, arm_key)
            target, dist = self.tracker.grasp_target(fr.eef_world(arm_key), cfg.camera, px)
            rec["grasp_retry"] = {"target": target, "distance": dist}
            if target is not None:
                self.tracker.counts["grasp_retarget"] += 1
                self._take(e, arm_key, fr, main, target, px)
                touched_now.add(target)
                self.touched_at[target] = e.t
                robot_keys |= self._robot_state(fr, e.t)
                self.verifier.held_off_supports(target, e.t)
        for a in self.tracker.tracks:
            self.marks.assign(a, self.tracker.tracks[a].category)
        self.geo.observe_sizes()
        img, marks_text, shown = self._marked(main, fr)
        now_url = encode_png(img)
        kf = self._save_keyframe(e.t, main.name, img)
        held = {k: self.tracker.held(k) for k in self.arm_keys}
        ev_text = _event_text(e, {self._arm_label(k): v for k, v in held.items()})
        sugg, sugg_geo = self._suggestions(e)
        # agent: evidence plan
        plan: Plan
        agent_rec: dict[str, Any] = {}
        if e.kind in cfg.agent_on:
            msgs = agent_prompt(f"t={e.t} ({e.t / self.ep.fps:.1f} s) {e.kind}: {ev_text}",
                                self.graph.summary(), self.goal.text(self.graph),
                                self.graph.remember + [f"{k}={v}" for k, v in self.graph.notes.items()],
                                [statement_text(k) for k in self.verifier.recheck_keys()], marks_text,
                                [statement_text(k) for k in sugg], now_url,
                                [cfg.camera] + list(cfg.look_views), cfg.offline, cfg.max_obs,
                                categories=self._detect_categories())
            agent_rec["prompt"] = text_only(msgs)
            plan = None  # type: ignore[assignment]
            replies = []
            reason = ""
            for _ in range(cfg.agent_retry + 1):
                text, usage, err, meta = self._query("agent", msgs)
                replies.append({"reply": text, "error": err, "usage": usage, **meta})
                try:
                    if err:
                        raise PlanError(f"query failed: {err}")
                    plan = parse_plan(text, self._names(), self.arms, [cfg.camera] + list(cfg.look_views),
                                      self._detect_categories(), cfg.offline, cfg.max_obs)
                    break
                except PlanError as exc:
                    reason = str(exc)
                    if meta.get("incomplete") == "length":
                        reason += f" (reply cut at max_tokens, finish_reason={meta.get('finish_reason')})"
            if plan is None:
                self.agent_failures += 1
                plan = default_plan(sugg, sugg_geo, reason, cfg.max_obs)
            agent_rec["replies"] = replies
        else:
            plan = Plan(why="no agent call for this event kind")
        # verifier re-checks are always asked (R7)
        for k in self.verifier.recheck_keys():
            if k not in plan.observe:
                plan.observe.append(k)
                plan.forced.append(statement_text(k))
        tool: dict[str, Any] = {}
        if plan.detect:
            extra = []
            for key in plan.detect:
                c = next(c for c in self.categories if c.key == key)
                nxt = next((x for x in c.candidates if x not in self.tried.setdefault(key, {c.sam3_text})), None)
                if nxt is None:
                    continue
                self.tried[key].add(nxt)
                got = [d for d in self.det.detect(main, [c.with_query(nxt)]) if d.category == key]
                self.redetects.append({"t": e.t, "category": key, "prompt": nxt, "detections": len(got), "by": "agent"})
                if got:
                    self._set_query(key, nxt)
                    extra += got
            if extra:
                # B2: only the new detections; tracks matched by this frame's first pass keep that match
                tool["detect"] = self.tracker.update(e.t, main.name, extra, self._img_base(main), plan.detect, keep_seen=True)
                for a in self.tracker.tracks:
                    self.marks.assign(a, self.tracker.tracks[a].category)
                img, marks_text, shown = self._marked(main, fr)
                now_url = encode_png(img)
        obs_cam, view, before = main, cfg.view, self.prev_image.get(main.name)
        if plan.camera and plan.camera != cfg.camera:
            lc = fr.cams.get(plan.camera)
            if lc is not None and lc.K is not None and lc.T_base_cam is not None:
                limg, ltext, _ = self._marked(lc, fr, seen_only=False)
                obs_cam, view, before = lc, cfg.look_views.get(plan.camera, "wrist"), None
                now_obs_url, obs_marks = encode_png(limg), ltext
                tool["look"] = plan.camera
            else:
                tool["look"] = f"rejected: no camera pose for {plan.camera}"
        observed: list[tuple[EdgeKey, str]] = []
        accepted: set[EdgeKey] = set()
        vals: dict[EdgeKey, str] = {}
        # A1: at a release the geometry of the released object vs the nearby marked supports / containers is always
        # computed (and those statements asked), whatever the agent chose
        auto_pairs: list[tuple[str, str]] = []
        if e.kind == "release" and e.obj and arm_key is not None and e.obj in self.tracker.tracks:
            auto_pairs = self._release_candidates(e.obj, np.asarray(fr.eef_world(arm_key), float),
                                                  self._eef_pixel(main, fr, arm_key), main.name)
            for a, s in auto_pairs:
                k = self._rel_key(a, s)
                if k not in plan.observe:
                    plan.observe.append(k)
                    plan.forced.append(statement_text(k))
        if plan.observe:
            url = now_url if obs_cam is main else now_obs_url
            mt = marks_text if obs_cam is main else obs_marks
            vals, tool["observe"] = self._observe(obs_cam, before, url, mt, ev_text, plan.observe, view)
            observed = [(k, vals.get(k, "?")) for k in plan.observe]
        release = {e.obj: np.asarray(fr.eef_world(arm_key))} if e.kind == "release" and e.obj and arm_key else None
        geom = self._geometry_for(vals, main.name, list(plan.geometry) + auto_pairs, release)
        if plan.geometry or auto_pairs:
            tool["geometry"] = {statement_text(k): g for k, g in geom.items()}
        hidden = self._hidden()
        touched = dict(self.touched_at)
        accepted = self._observe_apply(vals, e.t, touched, geom, hidden)
        observed = [(k, ("?" if (k[0] in STATE_PREDS and k[1] in hidden) else v)) for k, v in observed]
        hind: dict[EdgeKey, str] = {}
        # R10: 'in' answered 0 against the geometry -> re-check now (another camera; mode A: a later quiet frame)
        # the row keeps which first answer was not adopted (``rejected``) and the adopted look re-check value
        # (``rechecked``; a hindsight re-check goes to ``hindsight``): the backward pass (B2) uses adopted answers only
        rechecked: dict[EdgeKey, str] = {}
        rejected: list[EdgeKey] = []
        if self.verifier.new_conflicts:
            conf = list(self.verifier.new_conflicts)
            done_keys, tool["recheck"], adopted = self._recheck(e, fr, obs_cam.name, conf, ev_text, next_gripper_t, hind)
            accepted |= done_keys
            for k in conf:
                a = adopted.get(k)
                if a is not None and a[1].startswith("look"):
                    rechecked[k] = a[0]
                if a is None or a[0] != "0":  # pending (0.5 conflict) or overturned: the first 0 was not adopted
                    rejected.append(k)
        # R9: release-geometry candidates the VDM left undecided: the best positive one -> expectation 0.7
        if auto_pairs:
            cmap = self._categories_map()
            und = [k for k in (self._rel_key(a, s) for a, s in auto_pairs) if vals.get(k, "?") == "?" and k in geom]
            und.sort(key=lambda k: geom[k].get("d_xy", geom[k].get("release_d_xy", 1e9)))
            for k in und:
                if self.verifier.geo_fill(k, geom[k], e.t, cmap, held={self.tracker.held(a) for a in self.arm_keys} - {None}):
                    break
        for k, v in plan.expect:
            self.verifier.expect(k, v, e.t, accepted | set(plan.observe))  # asked now, even if '?'
        self._process_model(e.t)
        if plan.hindsight and cfg.offline:
            h = self._hindsight(e, plan.hindsight, plan.hindsight_after_s, next_gripper_t)
            hind.update(h.pop("raw", {}))
            tool["hindsight"] = h
        self.graph.age(accepted, robot_keys)
        confirmed = self.goal.update(self.graph, e.t)
        self.prev_image[main.name] = now_url
        row = self._row(e, observed, touched_now, hind, kf)
        row.rechecked, row.rejected = rechecked, rejected
        rec.update(agent=agent_rec, plan=plan.to_dict(), tools=tool, marks=marks_text,
                   decisions=[d.to_dict() for d in self.verifier.decisions], goal_confirmed=confirmed,
                   tracks=[t.to_dict() for t in self.tracker.tracks.values()],
                   pending=[statement_text(k) for k in self.verifier.recheck_keys()])
        self._trace(e, row, rec)
        return row

    def _take(self, e: Event, arm_key: str, fr: FrameRecord, main: CameraFrame | None, target: str,
              px: tuple[float, float] | None) -> None:
        """The gripper holds ``target`` from this event on (``close``: provisionally, checked by co-motion)."""
        frm = self._support_of(target)
        self.tracker.hold(target, arm_key, fr.eef_world(arm_key))
        self.grasp_z[arm_key] = float(fr.eef_world(arm_key)[2])
        e.obj = target
        self.graph.events.append(f"robot -grasp-> {target} @{e.t}" + (f" (from {frm})" if frm else "")
                                 + (" (provisional)" if e.kind == "close" else ""))
        if e.kind == "close":
            tr = self.tracker.tracks[target]
            self.pending_grasp[arm_key] = {
                "obj": target, "t": e.t, "row": len(self.rows), "eef0": np.asarray(fr.eef_world(arm_key), float),
                "pos0": None if tr.pos is None else tr.pos.copy(),
                # 2D path: the object's and the fingertip's pixels at the close
                "px0": tr.pixel.get(self.cfg.camera), "eef_px0": px,
                "cam_pose0": None if main is None else self._cam_world(main),
                # highest EEF rise above the close while closed (C1: lifted and put back), frames scanned up to lift_t
                "max_lift": 0.0, "lift_t": e.t,
                # relations before the close, restored if the grasp is retracted
                "loc0": {k: (ed.p, ed.source) for k, ed in self.graph.edges.items()
                         if k[1] == target and k[0] in ("on", "in", "at")}}

    def _update_lift(self, t: int) -> None:
        """Highest EEF rise of each provisional grasp since its close, over every robot frame up to ``t``."""
        for arm, pend in self.pending_grasp.items():
            for f in range(pend["lift_t"] + 1, t + 1):
                z = float(self.ep.robot(f).eef_world(arm)[2])
                pend["max_lift"] = max(pend["max_lift"], z - float(pend["eef0"][2]))
            pend["lift_t"] = max(pend["lift_t"], t)

    def _release_candidates(self, obj: str, eef_w: np.ndarray, eef_px: tuple[float, float] | None,
                            cam: str) -> list[tuple[str, str]]:
        """A1: (released object, support) pairs for the supports / containers detected in the release image (the
        event's mark table; one right under the gripper counts though it gets no drawn mark), nearest first:
        horizontal box distance to the release EEF (3D) or pixel distance (2D); at most ``release_geo_k``."""
        cands = []
        for s, tr in self.tracker.tracks.items():
            if s == obj or not tr.seen_now or not is_support(tr.category):
                continue
            if tr.pos is not None:
                size = None if tr.size is None else np.array([tr.size[0], tr.size[1], 0.0])
                d = box_distance(np.array([eef_w[0], eef_w[1], 0.0]), np.array([tr.pos[0], tr.pos[1], 0.0]), size)
                if d <= self.cfg.release_geo_radius:
                    cands.append((0, d, s))
            elif eef_px is not None and cam in tr.pixel:
                cands.append((1, float(np.hypot(eef_px[0] - tr.pixel[cam][0], eef_px[1] - tr.pixel[cam][1])), s))
        return [(obj, s) for _, _, s in sorted(cands)[: self.cfg.release_geo_k]]

    def _recheck(self, e: Event, fr: FrameRecord, used_cam: str, keys: list[EdgeKey], ev_text: str,
                 next_gripper_t: int, hind: dict[EdgeKey, str],
                 ) -> tuple[set[EdgeKey], dict[str, Any], dict[EdgeKey, tuple[str, str]]]:
        """R10: re-ask conflicting 'in' keys now -- another camera first, then (mode A) a later frame of the same quiet
        interval. Returns (keys decided by an observation, trace info, adopted {key: (value, "look <cam>" | "hindsight")})."""
        adopted: dict[EdgeKey, tuple[str, str]] = {}
        cfg = self.cfg
        info: dict[str, Any] = {"keys": [statement_text(k) for k in keys]}
        decided_keys: set[EdgeKey] = set()
        left = list(keys)
        if cfg.recheck_look:
            for c in [cfg.camera, *cfg.look_views]:
                lc = fr.cams.get(c)
                if not left or c == used_cam or lc is None or (c != cfg.camera and (lc.K is None or lc.T_base_cam is None)):
                    continue
                img, text, _ = self._marked(lc, fr, seen_only=(c == cfg.camera))
                vals, oinfo = self._observe(lc, None, encode_png(img), text, ev_text, left,
                                            cfg.view if c == cfg.camera else cfg.look_views.get(c, "wrist"))
                info.setdefault("look", []).append({"camera": c, "values": {statement_text(k): vals.get(k, "?") for k in left},
                                                    "observe": oinfo})
                for k in list(left):
                    if self.verifier.recheck(k, vals.get(k, "?"), e.t, f"look {c}"):
                        decided_keys.add(k)
                        adopted[k] = (vals[k], f"look {c}")
                        left.remove(k)
                break  # one other camera
        if left and cfg.offline:
            h = self._hindsight(e, left, cfg.recheck_hindsight_s, next_gripper_t)
            raw = dict(h.pop("raw", {}))
            info["hindsight"] = h
            for k in list(left):
                v = raw.get(k, "?")
                if v in ("0", "1"):
                    hind[k] = v
                if self.verifier.recheck(k, v, e.t, "hindsight"):
                    decided_keys.add(k)
                    adopted[k] = (v, "hindsight")
                    left.remove(k)
        info["pending"] = [statement_text(k) for k in left]
        return decided_keys, info, adopted

    def _grasp_evidence(self, pend: dict[str, Any], arm: str, fr: FrameRecord, cam: CameraFrame,
                        dets: Sequence[Detection], hand_world: np.ndarray | None) -> tuple[str, dict[str, Any]]:
        """Compare two hypotheses for a provisional grasp with this image's detections of the object's category:
        "moved with the hand" (expected at ``hand_world`` / its pixel) and "stayed" (where it was at the close).

        Returns (verdict, info): ``with_hand`` (a detection at the hand and none left at the old place),
        ``left`` (one at the old place and none at the hand), ``ambiguous`` (both), ``gone_lifted`` (the category
        is detected, nothing at either place, the hand rose and the old place is not under the hand),
        ``no_detection`` (the category is not detected at all: occlusion / detector miss), ``too_close`` (the two
        places are not yet apart), ``no_reference`` (no 3D or 2D reference), ``camera_moved`` / ``camera_unknown``
        (2D path: pixels are compared only when the camera pose is unchanged since the close), ``unclear``. A detection that
        another known track of the same category explains (e.g. the other bowl) is not evidence."""
        tr = self.tracker.tracks[pend["obj"]]
        base = self._img_base(cam)
        cat = [d for d in dets if d.category == tr.category]
        others = [t for a, t in self.tracker.tracks.items() if a != pend["obj"] and t.category == tr.category]
        eef_w = np.asarray(fr.eef_world(arm), float)
        lifted = float(eef_w[2] - pend["eef0"][2]) >= self.cfg.lift_dz
        if pend.get("pos0") is not None and hand_world is not None:
            mode, hand, old, r, sep = "3d", np.asarray(hand_world, float), pend["pos0"], self.cfg.tracks.assoc_radius, self.cfg.verify_sep
            refs = [w for d in cat if (w := Tracker._world(d, base)) is not None]
            other_refs = [t.pos for t in others if t.pos is not None]
            occluded = float(np.linalg.norm(eef_w[:2] - old[:2])) < self.cfg.tracks.grasp_radius + 0.03
        else:
            eef_px = self._eef_pixel(cam, fr, arm)
            if pend.get("px0") is None or pend.get("eef_px0") is None or eef_px is None:
                return "no_reference", {"mode": "none"}
            # pixels at the close and now are comparable only from the same camera pose (a head camera moves)
            pose0, pose = pend.get("cam_pose0"), self._cam_world(cam)
            if pose0 is None or pose is None:
                if self.ep.mobile:
                    return "camera_unknown", {"mode": "2d"}
                mode = "2d_fixed_camera_assumed"
            else:
                moved = self._pose_change(pose0, pose)
                if moved[0] > self.cfg.cam_still_m or moved[1] > self.cfg.cam_still_deg:
                    return "camera_moved", {"mode": "2d", "cam_moved_m": round(moved[0], 3), "cam_moved_deg": round(moved[1], 2)}
                mode = "2d"
            r, sep = self.cfg.tracks.assoc_px, self.cfg.verify_sep_px
            old = np.asarray(pend["px0"], float)
            hand = old + (np.asarray(eef_px, float) - np.asarray(pend["eef_px0"], float))
            refs = [np.asarray(d.pixel, float) for d in cat]
            other_refs = [np.asarray(t.pixel[cam.name], float) for t in others if cam.name in t.pixel]
            occluded = float(np.linalg.norm(np.asarray(eef_px, float) - old)) < r
        info: dict[str, Any] = {"mode": mode, "separation": round(float(np.linalg.norm(hand - old)), 3), "detections": len(refs)}
        if np.linalg.norm(hand - old) < sep:
            return "too_close", info
        if not refs:
            return "no_detection", info

        def explained(p: np.ndarray) -> bool:
            return any(np.linalg.norm(p - o) <= 0.5 * r for o in other_refs)

        at_hand = [p for p in refs if np.linalg.norm(p - hand) <= r and np.linalg.norm(p - hand) < np.linalg.norm(p - old)
                   and not explained(p)]
        at_old = [p for p in refs if np.linalg.norm(p - old) <= r and np.linalg.norm(p - old) <= np.linalg.norm(p - hand)
                  and not explained(p)]
        info.update(at_hand=len(at_hand), at_old=len(at_old), lifted=lifted, old_under_hand=occluded)
        if at_hand and at_old:
            return "ambiguous", info
        if at_old:
            return "left", info
        if at_hand:
            return "with_hand", info
        if lifted and not occluded:
            return "gone_lifted", info
        return "unclear", info

    def _cam_world(self, cam: CameraFrame) -> np.ndarray | None:
        """world <- camera at the image's frame (None if the camera pose is unknown)."""
        if cam.T_base_cam is None:
            return None
        b = self._img_base(cam)
        return np.asarray(cam.T_base_cam, float) if b is None else np.asarray(b, float) @ np.asarray(cam.T_base_cam, float)

    @staticmethod
    def _pose_change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """(translation m, rotation deg) between two 4x4 poses."""
        dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
        return dt, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    def _retract(self, arm: str, pend: dict[str, Any], t: int, why: str) -> None:
        """The close did not hold the object: back where it was, its relations as remembered before the close."""
        tr = self.tracker.tracks.get(pend["obj"])
        if tr is not None:
            tr.held_by, tr.hold_offset, tr.released_at = None, None, None
            if pend.get("pos0") is not None:
                tr.pos = pend["pos0"]
        if self.tracker.last_released == pend["obj"]:
            self.tracker.last_released = None
        self.grasp_z.pop(arm, None)
        for k, (p, _src) in pend.get("loc0", {}).items():
            self.graph.set(k, p, "mem", t)
        self.graph.events.append(f"robot -grasp-> {pend['obj']} @{pend['t']} retracted @{t} ({why})")
        if pend["row"] < len(self.rows):
            self.rows[pend["row"]].notes.append("grasp_retracted")  # backward: a close, not a grasp

    def _check(self, arm: str, pend: dict[str, Any], t: int, result: str, info: dict[str, Any], at: str) -> dict[str, Any]:
        eef = np.asarray(self.ep.robot(t).eef_world(arm), float)
        c = {"arm": arm, "obj": pend["obj"], "t": pend["t"], "result": result, "decided_t": t, "at": at,
             "eef_moved_m": round(float(np.linalg.norm(eef - pend["eef0"])), 3), **info}
        self.grasp_checks.append(c)
        return c

    def _verify_grasps(self, fr: FrameRecord, cam: CameraFrame, dets: Sequence[Detection], t: int) -> list[dict[str, Any]]:
        """Co-motion check of provisional grasps (``close`` events) while the hand is still closed. Decided only on
        clear evidence (``_grasp_evidence``): with_hand / gone_lifted -> confirmed, left -> retracted; anything else
        (ambiguous, no detection, too close, unclear) waits for the next event (and finally the release)."""
        out = []
        for arm, pend in list(self.pending_grasp.items()):
            tr = self.tracker.tracks.get(pend["obj"])
            if tr is None:
                self.pending_grasp.pop(arm)
                continue
            if tr.category in OPENABLE_KEYS:
                # a door / lid moves on an arc while the body stays: body co-motion is no test
                self.pending_grasp.pop(arm)
                out.append(self._check(arm, pend, t, "articulated", {}, "hold"))
                continue
            if np.linalg.norm(np.asarray(fr.eef_world(arm), float) - pend["eef0"]) < self.cfg.verify_move:
                continue
            verdict, info = self._grasp_evidence(pend, arm, fr, cam, dets, tr.pos)
            pend["last"] = verdict
            result = {"with_hand": "confirmed", "gone_lifted": "confirmed_lifted", "left": "retracted"}.get(verdict)
            if result is None:
                continue
            if result == "retracted":
                self._retract(arm, pend, t, "stayed where it was")
            self.pending_grasp.pop(arm)
            out.append(self._check(arm, pend, t, result, {"evidence": verdict, **info}, "hold"))
        return out

    def _resolve_release(self, arm: str, pend: dict[str, Any], e: Event, fr: FrameRecord, cam: CameraFrame,
                         dets: Sequence[Detection]) -> dict[str, Any]:
        """A release while the grasp is still unverified: check it here, with the release image. The object
        detected at the release place (and not at the old one) -> confirmed; detected at the old place only ->
        retracted (nothing was carried: the event lets go of nothing); otherwise the place stays UNVERIFIED (the
        close and release rows are flagged; the backward pass labels the segment maybe_* and gives no Effect)."""
        tr = self.tracker.tracks.get(pend["obj"])
        if tr is None:
            return self._check(arm, pend, e.t, "unverified", {"evidence": "track gone"}, "release")
        verdict, info = self._grasp_evidence(pend, arm, fr, cam, dets, tr.pos)
        eef =np.asarray(fr.eef_world(arm), float)
        info["max_lift"] = round(float(pend.get("max_lift", 0.0)), 3)
        info["release_from_close_xy"] = round(float(np.linalg.norm(eef[:2] - pend["eef0"][:2])), 3)
        if verdict == "with_hand":
            return self._check(arm, pend, e.t, "confirmed", {"evidence": verdict, **info}, "release")
        if verdict == "left" and info["max_lift"] >= self.cfg.carry_lift and info["release_from_close_xy"] <= self.cfg.put_back_radius:
            # C1: at the old place at the release, but the closed hand rose >= carry_lift and came back down where it
            # closed: lifted and put back (co-motion while held wins over "it is where it was")
            return self._check(arm, pend, e.t, "confirmed_put_back", {"evidence": "lifted_and_put_back", **info}, "release")
        if verdict == "left":
            self._retract(arm, pend, e.t, "found where it was at the release")
            e.obj = None
            return self._check(arm, pend, e.t, "retracted", {"evidence": verdict, **info}, "release")
        return self._mark_unverified(arm, pend, e, verdict, info)

    def _mark_unverified(self, arm: str, pend: dict[str, Any], e: Event, evidence: str, info: dict[str, Any]) -> dict[str, Any]:
        """The grasp could not be checked by the release: the close row and the release are flagged (backward:
        maybe_* labels, no Effect) and the check is recorded."""
        if pend["row"] < len(self.rows):
            self.rows[pend["row"]].notes.append("grasp_unverified")
        e.info["place_unverified"] = True
        return self._check(arm, pend, e.t, "unverified", {"evidence": evidence, **info}, "release")

    def _observe_apply(self, vals: dict[EdgeKey, str], t: int, touched: dict[str, int], geom: dict[EdgeKey, Any],
                       hidden: set[str]) -> set[EdgeKey]:
        # R7 needs "touched after this key's last observation": per key, the aliases touched since then
        out: set[EdgeKey] = set()
        for k, v in vals.items():
            e = self.graph.get(k)
            since = -1 if e is None or e.obs_t is None else e.obs_t
            tset = {a for a in (k[1], k[2]) if a is not None and touched.get(a, -10) > since}
            out |= self.verifier.observe({k: v}, t, tset, geom, hidden, self._categories_map(),
                                         held={self.tracker.held(a) for a in self.arm_keys} - {None})
        return out

    def _process_model(self, t: int) -> None:
        """cooked(X) after in(X, C) and toggled_on(C) held for ``cook_s`` (source mem: a process the
        camera cannot see through a closed door)."""
        if not self.cfg.cook_s:
            return
        active: dict[EdgeKey, str] = {}
        for k, e in self.graph.edges.items():
            if k[0] == "in" and e.p >= 0.7 and k[2] is not None and (self.graph.p(("toggled_on", k[2], None)) or 0.0) >= 0.7:
                active[("cooked", k[1], None)] = k[2]
        for ck in list(self.cook_since):  # out of the container, or the appliance off: the timer restarts
            if active.get(ck) != self.cook_since[ck][0]:
                self.cook_since.pop(ck)
        for ck, c in active.items():
            t0 = self.cook_since.setdefault(ck, (c, t))[1]
            k = ("in", ck[1], c)
            if (t - t0) / self.ep.fps >= self.cfg.cook_s and (self.graph.p(ck) or 0.0) < 0.7:
                before = self.graph.p(ck)
                self.graph.set(ck, 0.7, "mem", t)
                self.verifier._log(ck, "R8-process", "process", before, f"in {k[2]} and toggled_on for {self.cfg.cook_s} s")

    # --- rows / trace --------------------------------------------------------------------------------
    def _subtask_causal(self) -> str:
        for k in self.arm_keys:
            a = self.tracker.held(k)
            if a is None:
                continue
            if self._cat(a) in OPENABLE_KEYS:
                return f"manipulate({a})"
            tgt = next((x[2] for x in self.goal.next_keys() if x[1] == a and x[0] in ("on", "in")), None)
            op = "put_in" if tgt and is_container(self._cat(tgt)) else "place"
            return f"{op}({a}, {tgt or '?'})"
        nxt = self.goal.next_keys()
        if nxt:
            x = nxt[0]
            if x[0] in ("on", "in"):
                return f"pick({x[1]}, {self._support_of(x[1]) or '?'})"
            return f"{'open' if x[0] == 'open' else x[0]}({x[1]})"
        return "done" if self.goal.known else "?"

    def _row(self, e: Event, observed: list[tuple[EdgeKey, str]], touched: set[str], hind: dict[EdgeKey, str],
             kf: str | None) -> Row:
        b = self.goal.best()
        prev = self.rows[-1].belief if self.rows else {}
        # flips: keys whose truth (P >= 0.5) changed at this row, or that appear true
        changed = [k for k, ed in self.graph.edges.items()
                   if (ed.p >= 0.5) != ((prev[k][0] if k in prev else 0.0) >= 0.5)]
        row = Row(e.t, e.kind, e.arm, e.obj, observed, {k: (ed.p, ed.source) for k, ed in self.graph.edges.items()},
                  changed, self.goal.text(self.graph), 0 if b is None else self.goal.done[b],
                  0 if b is None else len(self.goal.branches[b]), self._subtask_causal(), sorted(touched),
                  {k: self.tracker.held(k) for k in self.arm_keys}, hind, list(self.verifier.s4), keyframe=kf)
        self.rows.append(row)
        return row

    def _trace(self, e: Event, row: Row, rec: dict[str, Any]) -> None:
        line = {"t": e.t, "time_s": round(e.t / self.ep.fps, 2), "event": e.to_dict(),
                "graph": self.graph.snapshot(), "goal": self.goal.text(self.graph), "subtask_forward": row.subtask,
                "observe": [[statement_text(k), v] for k, v in row.observed], "s4": row.s4, **rec}
        self.trace.append(line)

    # --- run ---------------------------------------------------------------------------------------------
    def run(self) -> list[Row]:
        self.events = detect_events(self.ep, self.cfg.events)
        self.initialize()
        grip_t = [e.t for e in self.events if e.is_gripper]
        for e in self.events[1:]:
            nxt = next((t for t in grip_t if t > e.t), self.ep.n_frames)
            self.step(e, nxt)
        return self.rows

    def unobservable_steps(self) -> list[dict[str, Any]]:
        """A4: steps of the best branch, not confirmed, with a relation that could not be asked until the end of the
        demonstration: its object is still in the hand at the last event (on / in / at of a held object are robot-owned,
        e.g. at(basket, center) while the basket is carried to the end), or it was never detected (expected name). The
        goal decision itself is unchanged; this only separates "could not be seen" from "seen and wrong"."""
        b = self.goal.best()
        if b is None or not self.rows:
            return []
        held_end = {a for a in self.rows[-1].held.values() if a}
        unbound = set(self.tracker.expected())
        out = []
        for i in range(self.goal.done[b], len(self.goal.branches[b])):
            why = {}
            for k in self.goal.branches[b][i]:
                if k[0] in ("on", "in", "at") and k[1] in held_end:
                    why[statement_text(k)] = f"{k[1]} held until the end"
                elif any(x in unbound for x in k[1:] if x):
                    why[statement_text(k)] = "never detected"
            if why:
                out.append({"branch": b, "step": i, "why": why})
        return out

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for e in self.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        rules: dict[str, int] = {}
        for tline in self.trace:
            for d in tline.get("decisions", []):
                rules[d["rule"]] = rules.get(d["rule"], 0) + 1
        b = self.goal.best()
        reasons: dict[str, int] = {}
        for tline in self.trace:
            r = (tline.get("plan") or {}).get("fallback_reason")
            if r:
                reasons[r] = reasons.get(r, 0) + 1
        cats: dict[str, int] = {}
        for tr in self.tracker.tracks.values():
            cats[tr.category] = cats.get(tr.category, 0) + 1
        identity = {**self.tracker.counts, "tracks_per_category": cats, "expected_unbound": self.tracker.expected(),
                    "late_merges": len(self.tracker.merges), "caps": dict(self.tracker.caps),
                    "capped": self.tracker.capped[:50]}
        return {"init_failed": self.init_info.get("failed"), "track_identity": identity,
                "events": kinds, "rows": len(self.rows), "agent_fallbacks": self.agent_failures,
                "agent_fallback_reasons": reasons, "grasp_checks": self.grasp_checks,
                "verifier_rules": rules, "redetects": self.redetects, "merges": self.tracker.merges,
                "tracks": [t.to_dict() for t in self.tracker.tracks.values()],
                "goal": {"branches": [[[statement_text(k) for k in s] for s in br] for br in self.goal.branches],
                         "best": b, "done": self.goal.done, "done_t": self.goal.done_t,
                         "final_satisfied": self.goal.satisfied_final(self.graph),
                         "unobservable": self.unobservable_steps()},
                "final_graph": self.graph.snapshot(), "image_reads": self.ep.image_reads,
                "detector": {"name": getattr(self.det, "name", "?"), "calls": getattr(self.det, "calls", None)},
                "models": self.models.usage() if hasattr(self.models, "usage") else {}}


def save_trace(path: Path, trace: Sequence[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for line in trace:
            f.write(json.dumps(line, default=_default) + "\n")


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "__dict__"):
        return asdict(o) if hasattr(o, "__dataclass_fields__") else o.__dict__
    return str(o)
