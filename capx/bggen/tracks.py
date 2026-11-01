"""Object tracks of the BG generator: association across keyframes, names from events.

* Aliases: at initialization, per category from the image left (``bowl_A`` is the leftmost bowl).
  A track that appears later gets the next free letter.
* Association per category: 3D distance of world centers (depth path, ``assoc_radius``) or pixel
  distance (2D path, ``assoc_px``; only meaningful for a fixed camera).
* A held object follows its arm's EEF (offset fixed at the grasp).
* Same-looking objects are named by EVENTS, not appearance: an unmatched detection near the place
  where the last held object was released (``merge_radius``, any time before that object is seen
  again) is that object. If the new detection was first given a fresh alias, the fresh alias is
  merged into the released object's alias and recorded in ``merges`` (the backward pass rewrites
  earlier rows that used it).
* Grasp target: the track nearest to the EEF by point-to-box distance (``grasp_radius``); movable
  objects win over supports / containers at the same distance (a door handle grasp -> the fridge).
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .perception import Detection
from .vocab import is_support


@dataclass
class TrackConfig:
    assoc_radius: float = 0.08  # m
    assoc_px: float = 40.0
    merge_radius: float = 0.10  # m (2D path: merge_px)
    merge_px: float = 60.0
    grasp_radius: float = 0.08  # m, EEF to object box
    fixed_categories: tuple[str, ...] = ("plate",)  # never move (LIBERO-Mem plates); not grasp targets


@dataclass
class Track:
    alias: str
    category: str
    pos: np.ndarray | None = None  # world, m
    size: np.ndarray | None = None
    pixel: dict[str, tuple[float, float]] = field(default_factory=dict)  # cam -> last pixel
    box: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict, repr=False)  # cam -> last mask (2D path)
    last_seen: int | None = None
    seen_now: bool = False
    held_by: str | None = None  # arm key ("main" for a one-arm robot)
    hold_offset: np.ndarray | None = None
    released_at: tuple[int, np.ndarray | None, tuple[float, float] | None] | None = None  # (t, world eef, pixel)
    created: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"alias": self.alias, "category": self.category,
                "pos": None if self.pos is None else [round(float(x), 3) for x in self.pos],
                "size": None if self.size is None else [round(float(x), 3) for x in self.size],
                "last_seen": self.last_seen, "seen_now": self.seen_now, "held_by": self.held_by}


def box_distance(p: np.ndarray, center: np.ndarray, size: np.ndarray | None) -> float:
    half = 0.5 * (np.asarray(size) if size is not None else np.zeros(3))
    d = np.maximum(np.abs(np.asarray(p) - np.asarray(center)) - half, 0.0)
    return float(np.linalg.norm(d))


class Tracker:
    def __init__(self, cfg: TrackConfig | None = None) -> None:
        self.cfg = cfg or TrackConfig()
        self.tracks: dict[str, Track] = {}
        self.merges: list[dict[str, Any]] = []  # {"t", "from", "into", "why"}
        self.log: list[dict[str, Any]] = []
        self.last_released: str | None = None

    # --- names -----------------------------------------------------------------------------------
    def _next_alias(self, cat: str) -> str:
        used = {t.alias for t in self.tracks.values()}
        for c in string.ascii_uppercase:
            if f"{cat}_{c}" not in used:
                return f"{cat}_{c}"
        return next(f"{cat}_{a}{b}" for a in string.ascii_uppercase for b in string.ascii_uppercase
                    if f"{cat}_{a}{b}" not in used)

    def aliases(self) -> list[str]:
        return list(self.tracks)

    def movables(self) -> list[str]:
        return [a for a, t in self.tracks.items() if not is_support(t.category)]

    def supports(self) -> list[str]:
        return [a for a, t in self.tracks.items() if is_support(t.category)]

    # --- detections ---------------------------------------------------------------------------------
    @staticmethod
    def _world(det: Detection, base_pose: np.ndarray | None) -> np.ndarray | None:
        if det.center is None:
            return None
        if base_pose is None:
            return np.asarray(det.center, dtype=np.float64)
        return (base_pose @ np.array([*det.center, 1.0]))[:3]

    def initialize(self, t: int, cam: str, dets: Sequence[Detection], base_pose: np.ndarray | None = None) -> None:
        """First frame: one track per detection, aliases per category from the image left."""
        for d in sorted(dets, key=lambda d: (d.category, d.pixel[0])):
            self._new(t, cam, d, base_pose)

    def _new(self, t: int, cam: str, d: Detection, base_pose: np.ndarray | None) -> Track:
        tr = Track(self._next_alias(d.category), d.category, created=t)
        self._fill(tr, t, cam, d, base_pose)
        self.tracks[tr.alias] = tr
        return tr

    @staticmethod
    def _world_size(d: Detection, base_pose: np.ndarray | None) -> np.ndarray:
        """Base-axis extent -> world-axis extent (the axis-aligned bound of the rotated box)."""
        size = np.asarray(d.size, dtype=np.float64)
        return size if base_pose is None else np.abs(np.asarray(base_pose)[:3, :3]) @ size

    def _fill(self, tr: Track, t: int, cam: str, d: Detection, base_pose: np.ndarray | None) -> None:
        w = self._world(d, base_pose)
        if w is not None and tr.held_by is None:
            tr.pos, tr.size = w, self._world_size(d, base_pose)
        elif w is not None and tr.size is None:
            tr.size = self._world_size(d, base_pose)
        tr.pixel[cam], tr.box[cam] = d.pixel, d.box
        if d.mask is not None:
            tr.mask[cam] = d.mask
        tr.last_seen, tr.seen_now = t, True
        if tr.released_at is not None and t > tr.released_at[0]:
            tr.released_at = None

    def _dist(self, tr: Track, d: Detection, cam: str, base_pose: np.ndarray | None) -> float | None:
        w = self._world(d, base_pose)
        if w is not None and tr.pos is not None:
            return float(np.linalg.norm(w - tr.pos)) / self.cfg.assoc_radius
        if cam in tr.pixel:
            return float(np.hypot(d.pixel[0] - tr.pixel[cam][0], d.pixel[1] - tr.pixel[cam][1])) / self.cfg.assoc_px
        return None

    def update(self, t: int, cam: str, dets: Sequence[Detection], base_pose: np.ndarray | None = None,
               detected_categories: Sequence[str] | None = None) -> dict[str, Any]:
        """Associate this frame's detections; returns {"matched", "new", "merged", "dropped"}."""
        cats = set(detected_categories) if detected_categories is not None else {d.category for d in dets}
        for tr in self.tracks.values():
            if tr.category in cats:
                tr.seen_now = False
        rep: dict[str, Any] = {"matched": {}, "new": [], "merged": [], "dropped": 0}
        pairs = []
        for i, d in enumerate(dets):
            for a, tr in self.tracks.items():
                if tr.category != d.category:
                    continue
                dist = self._dist(tr, d, cam, base_pose)
                if dist is not None and dist <= 1.0:
                    pairs.append((dist, i, a))
        used_d: set[int] = set()
        used_t: set[str] = set()
        for dist, i, a in sorted(pairs):
            if i in used_d or a in used_t:
                continue
            used_d.add(i)
            used_t.add(a)
            self._fill(self.tracks[a], t, cam, dets[i], base_pose)
            rep["matched"][a] = round(dist, 3)
        for i, d in enumerate(dets):
            if i in used_d:
                continue
            # a duplicate of a track matched this frame (two masks of one object) is dropped
            if any((self._dist(self.tracks[a], d, cam, base_pose) or 9.0) <= 1.5 for a in used_t
                   if self.tracks[a].category == d.category):
                rep["dropped"] += 1
                continue
            target = self._released_near(d, cam, base_pose, exclude=used_t)
            if target is not None:
                self._fill(self.tracks[target], t, cam, d, base_pose)
                used_t.add(target)
                rec = {"t": t, "from": None, "into": target, "why": "detection at the release place of the last held object"}
                rep["merged"].append(rec)
                self.log.append({"kind": "merge_on_detect", **rec})
                continue
            tr = self._new(t, cam, d, base_pose)
            used_t.add(tr.alias)
            rep["new"].append(tr.alias)
            self.log.append({"kind": "new_track", "t": t, "alias": tr.alias})
        rep["merged"] += self._late_merges(t, cam)
        return rep

    def _released_near(self, d: Detection, cam: str, base_pose: np.ndarray | None, exclude: set[str]) -> str | None:
        """The released, not-yet-re-seen object of this category whose release place is near ``d``."""
        best, best_d = None, 1.0
        w = self._world(d, base_pose)
        for a, tr in self.tracks.items():
            if a in exclude or tr.category != d.category or tr.released_at is None or tr.held_by is not None:
                continue
            _, eef_w, px = tr.released_at
            if w is not None and eef_w is not None:
                dd = float(np.linalg.norm(w[:2] - eef_w[:2])) / self.cfg.merge_radius
            elif px is not None:
                dd = float(np.hypot(d.pixel[0] - px[0], d.pixel[1] - px[1])) / self.cfg.merge_px
            else:
                continue
            if dd <= best_d:
                best, best_d = a, dd
        return best

    def _late_merges(self, t: int, cam: str) -> list[dict[str, Any]]:
        """A track created after a release, at the release place, while the released object stays unseen:
        the same object (merged into the released alias; the fresh alias disappears)."""
        out = []
        for a, tr in list(self.tracks.items()):
            if tr.released_at is None or tr.seen_now:
                continue
            rt, eef_w, px = tr.released_at
            for b, nt in list(self.tracks.items()):
                if b == a or nt.category != tr.category or nt.created <= rt or not nt.seen_now:
                    continue
                if eef_w is not None and nt.pos is not None:
                    near = np.linalg.norm(nt.pos[:2] - eef_w[:2]) <= self.cfg.merge_radius
                elif px is not None and cam in nt.pixel:
                    near = np.hypot(nt.pixel[cam][0] - px[0], nt.pixel[cam][1] - px[1]) <= self.cfg.merge_px
                else:
                    near = False
                if near:
                    keep = self.tracks.pop(b)
                    tr.pos, tr.size, tr.pixel, tr.box, tr.mask = keep.pos, keep.size, keep.pixel, keep.box, keep.mask
                    tr.last_seen, tr.seen_now, tr.released_at = keep.last_seen, True, None
                    rec = {"t": t, "from": b, "into": a, "why": "new track at the release place of the last held object"}
                    self.merges.append(rec)
                    self.log.append({"kind": "merge_late", **rec})
                    out.append(rec)
                    break
        return out

    # --- robot interaction ----------------------------------------------------------------------
    def grasp_target(self, eef_world: np.ndarray, cam: str | None = None, eef_pixel: tuple[float, float] | None = None,
                     ) -> tuple[str | None, float | None]:
        """(alias, distance) of the object the gripper closed on, or (None, None)."""
        cands = []
        for a, tr in self.tracks.items():
            if tr.category in self.cfg.fixed_categories or tr.held_by is not None:
                continue
            if tr.pos is not None:
                d = box_distance(eef_world, tr.pos, tr.size)
                if d <= self.cfg.grasp_radius:
                    cands.append((is_support(tr.category), d, a))
            elif eef_pixel is not None and cam in tr.pixel:
                d = float(np.hypot(eef_pixel[0] - tr.pixel[cam][0], eef_pixel[1] - tr.pixel[cam][1]))
                if d <= self.cfg.merge_px:
                    cands.append((is_support(tr.category), d / self.cfg.merge_px * self.cfg.grasp_radius, a))
        if not cands:
            return None, None
        _, d, a = min(cands)
        return a, round(d, 4)

    def hold(self, alias: str, arm: str, eef_world: np.ndarray) -> None:
        tr = self.tracks[alias]
        tr.held_by = arm
        tr.hold_offset = None if tr.pos is None else tr.pos - eef_world
        tr.released_at = None

    def release(self, arm: str, t: int, eef_world: np.ndarray, eef_pixel: tuple[float, float] | None = None) -> str | None:
        for a, tr in self.tracks.items():
            if tr.held_by == arm:
                tr.held_by = None
                if tr.pos is not None or tr.hold_offset is not None:
                    tr.pos = eef_world + (tr.hold_offset if tr.hold_offset is not None else 0.0)
                tr.released_at = (t, np.asarray(eef_world, dtype=np.float64), eef_pixel)
                self.last_released = a
                return a
        return None

    def follow(self, arm: str, eef_world: np.ndarray) -> None:
        for tr in self.tracks.values():
            if tr.held_by == arm and tr.hold_offset is not None:
                tr.pos = eef_world + tr.hold_offset

    def held(self, arm: str) -> str | None:
        return next((a for a, tr in self.tracks.items() if tr.held_by == arm), None)
