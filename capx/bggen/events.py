"""Event detection from robot state only (design: 사건 감지).

* Gripper, per arm: the finger opening falls by more than ``close_delta`` and then stops (speed below
  ``stop_speed`` for ``stop_frames``). With the robot's fully-closed width known (``closed_width``): stopped
  on a plateau above it (+ ``empty_margin``) -> ``grasp``, else ``close_empty``. Unknown -> ``close``, decided
  by the generator from co-motion of the nearest track (``Generator._verify_grasps``; LIBERO-Mem: a bowl rim
  grasp closes the fingers to 4-6 mm, the same as an empty close, so width cannot decide). From a closed
  state, a rise of ``open_delta`` -> ``release`` (after a grasp / close) or ``open`` (after an empty close).
* Mobile base: speed above ``base_move_speed`` for ``stop_frames`` -> ``base_start``; below
  ``base_stop_speed`` for ``stop_frames`` -> ``base_stop``. The speed comes from the base pose track. Debounce:
  stops shorter than ``base_min_stop_s`` are merged, then moves whose start-to-stop distance is below
  ``base_min_move_m`` are dropped (B1K 00450010 raw: 69 pairs, stop median 1.02 s, move median 0.08 m).
* Keyframes: one every ``keyframe_s`` seconds since the last event of any kind (``keyframe``).

Events are causal: an event at frame t is decided from frames <= t + ``stop_frames`` (the stop has to
be seen). ``Event.t`` is the frame where the generator looks at the scene (the stop frame).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

GRIPPER_KINDS = ("grasp", "close", "close_empty", "release", "open")


@dataclass
class EventConfig:
    close_delta: float = 0.004  # m: a fall this large from the last open level counts as closing
    open_delta: float = 0.004  # m: a rise this large from the closed level counts as opening
    stop_speed: float = 0.0004  # m / frame: "stopped"
    stop_frames: int = 4
    # Robot-specific fully-closed width (fingers closed on nothing, m). Known: a close that stops above
    # closed_width + empty_margin (a plateau before fully closed) is a grasp, at or below it an empty close.
    # None (unknown for this robot / dataset): the close is reported as ``close`` and the generator decides
    # by co-motion (does the nearest track move with the EEF?).
    closed_width: float | None = None
    empty_margin: float = 0.002
    smooth: int = 3  # moving-average window (frames)
    base_move_speed: float = 0.05  # m / s
    base_stop_speed: float = 0.02  # m / s
    base_min_stop_s: float = 1.0  # debounce: a stop shorter than this between two moves is merged into one move
    base_min_move_m: float = 0.05  # debounce: a move whose start-to-stop distance is below this is dropped
    keyframe_s: float = 1.5  # keyframe every this many seconds without another event (<= 0: none)
    min_gap_frames: int = 2  # two gripper events of one arm are at least this far apart


@dataclass
class Event:
    t: int
    kind: str  # grasp | close (unknown) | close_empty | release | open | base_start | base_stop | keyframe | start | end
    arm: str | None = None
    width: float | None = None  # gripper opening at t
    eef: list[float] | None = None  # base-frame EEF at t
    obj: str | None = None  # grasp: the object (set by the tracker), release: the object let go
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_gripper(self) -> bool:
        return self.kind in GRIPPER_KINDS

    def label(self) -> str:
        arm = f"[{self.arm}]" if self.arm else ""
        obj = f" {self.obj}" if self.obj else ""
        return f"{self.kind}{arm}{obj}@{self.t}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.astype(np.float64)
    pad = np.concatenate([np.full(k - 1, x[0]), x]).astype(np.float64)
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def gripper_events(width: np.ndarray, cfg: EventConfig, arm: str | None = None,
                   eef: np.ndarray | None = None) -> list[Event]:
    """State machine over one arm's finger opening (see module docstring)."""
    w = _smooth(np.asarray(width, dtype=np.float64), cfg.smooth)
    n = len(w)
    floor = None if cfg.closed_width is None else cfg.closed_width + cfg.empty_margin
    out: list[Event] = []
    state = "open"
    ref = w[0] if n else 0.0  # open: the highest level since opening; closed: the stopped level
    closing = False
    still = 0
    last_t = -10 ** 9
    for t in range(1, n):
        v = w[t] - w[t - 1]
        if state == "open":
            ref = max(ref, w[t])
            if ref - w[t] > cfg.close_delta:
                closing = True
            if closing:
                still = still + 1 if abs(v) < cfg.stop_speed else 0
                if still >= cfg.stop_frames and t - last_t >= cfg.min_gap_frames:
                    kind = "close" if floor is None else "grasp" if w[t] > floor else "close_empty"
                    out.append(Event(t, kind, arm, float(width[t]), None if eef is None else list(map(float, eef[t])),
                                     info={"plateau_width": round(float(w[t]), 5), "closed_width": cfg.closed_width}))
                    state, ref, closing, still, last_t = ("closed" if kind == "close_empty" else "held"), w[t], False, 0, t
        else:
            if w[t] < ref and (floor is None or w[t] > floor - 1e-9) and abs(v) < cfg.stop_speed:
                ref = min(ref, w[t])  # slowly squeezing an object: keep the closed level current
            if w[t] - ref > cfg.open_delta and t - last_t >= cfg.min_gap_frames:
                # the rise started where w first left the closed level
                t0 = t
                while t0 - 1 > last_t and w[t0 - 1] - ref > 0.25 * cfg.open_delta:
                    t0 -= 1
                kind = "release" if state == "held" else "open"
                out.append(Event(t0, kind, arm, float(width[t0]), None if eef is None else list(map(float, eef[t0]))))
                state, ref, closing, still, last_t = "open", w[t], False, 0, t0
            elif state == "held" and floor is not None and w[t] <= floor:
                # the object slipped out: fingers closed on nothing
                out.append(Event(t, "release", arm, float(width[t]), None if eef is None else list(map(float, eef[t])),
                                 info={"slipped": True}))
                state, ref, last_t = "closed", w[t], t
    return out


def base_events(base_pose: np.ndarray, fps: float, cfg: EventConfig) -> list[Event]:
    xy = np.asarray(base_pose)[:, :2, 3]
    if len(xy) < 2:
        return []
    speed = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * fps
    speed = _smooth(speed, max(cfg.smooth, 5))
    step = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    segs: list[list[int]] = []  # raw [start, stop] moves
    moving = False
    run = 0
    for t in range(1, len(speed)):
        cond = speed[t] < cfg.base_stop_speed if moving else speed[t] > cfg.base_move_speed
        run = run + 1 if cond else 0
        if run >= cfg.stop_frames:
            moving = not moving
            if moving:
                segs.append([t - cfg.stop_frames + 1, len(speed) - 1])
            else:
                segs[-1][1] = t
            run = 0
    # debounce: merge moves separated by a short stop, then drop short moves
    merged: list[list[int]] = []
    for sgm in segs:
        if merged and (sgm[0] - merged[-1][1]) / fps < cfg.base_min_stop_s:
            merged[-1][1] = sgm[1]
        else:
            merged.append(list(sgm))
    out: list[Event] = []
    for i, (a, b) in enumerate(merged):
        path = float(step[a + 1:b + 1].sum())
        disp = float(np.linalg.norm(xy[b] - xy[a]))  # start-to-stop distance (robust to jitter in the path)
        if disp < cfg.base_min_move_m:
            continue
        info = {"path_m": round(path, 3), "disp_m": round(disp, 3)}
        out.append(Event(a, "base_start", info=info))
        if not (i == len(merged) - 1 and moving):  # still moving at the end: no stop
            out.append(Event(b, "base_stop", info=dict(info)))
    return out


def add_keyframes(events: Sequence[Event], n_frames: int, fps: float, keyframe_s: float) -> list[Event]:
    ev = sorted(events, key=lambda e: e.t)
    if keyframe_s <= 0:
        return list(ev)
    step = max(int(round(keyframe_s * fps)), 1)
    out: list[Event] = []
    last = 0
    for e in ev + [Event(n_frames - 1, "_end")]:
        t = last + step
        while t < e.t - step // 3:
            out.append(Event(t, "keyframe"))
            t += step
        if e.kind != "_end":
            out.append(e)
        last = e.t
    return out


def detect_events(episode: Any, cfg: EventConfig) -> list[Event]:
    """All events of an episode (``records.Episode``), sorted, with a ``start`` at frame 0 and an ``end``."""
    ev: list[Event] = []
    for arm in episode.arms:
        ev += gripper_events(episode.gripper(arm), cfg, arm if len(episode.arms) > 1 else None, episode.eef(arm))
    if episode.mobile:
        ev += base_events(episode.base_poses(), episode.fps, cfg)
    ev = [e for e in ev if 0 < e.t < episode.n_frames - 1]
    ev = add_keyframes(ev, episode.n_frames, episode.fps, cfg.keyframe_s)
    order = {"base_stop": 0, "release": 1, "open": 1, "grasp": 2, "close_empty": 2, "base_start": 3, "keyframe": 4}
    ev.sort(key=lambda e: (e.t, order.get(e.kind, 5)))
    return [Event(0, "start")] + ev + [Event(episode.n_frames - 1, "end")]
