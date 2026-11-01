"""Geometry tool (design: 도구 실행 · geometry).

Depth path (both tracks have 3D centers and were detected now): ``d_xy`` (center distance in the
horizontal plane), ``r`` (support radius, the median over the support's observations so far --
design 남은 약점: 반경을 여러 프레임에서 안정화), ``dz`` (center height difference). For a container
(fridge, microwave) ``inside`` says whether the object's center lies in the container's box.
The verifier turns these into the flat veto (``verifier.R5``).

2D path (no depth): mask relations in one camera -- ``overlap`` (share of the object's mask inside the
support's box), ``bottom_contact`` (the object's lowest row lies within the support's box rows and the
two overlap horizontally) -> an on / in candidate only; never a veto (the VDM decides). Plus the
event-time 3D: where the object was released (EEF position) vs the support's last 3D center.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .tracks import Tracker
from .vocab import TABLE, EdgeKey, is_container


class Geometry:
    def __init__(self, tracker: Tracker, support_radius_min: float = 0.03, radius_window: int = 9,
                 in_xy_margin: float = 0.02, in_bottom_margin: float = 0.02, in_top_margin: float = 0.04,
                 overlap_min: float = 0.5) -> None:
        self.tr = tracker
        # 'in' (container): margins around the container's box / height range. Top 4 cm: a held-size object resting
        # on the bottom of a shallow basket sticks out of it by up to half its height (ASSUMPTION, LIBERO cheese /
        # basket ~ 4-8 cm; measure from the batch2 geometry dumps)
        self.in_xy_margin, self.in_bottom_margin, self.in_top_margin = in_xy_margin, in_bottom_margin, in_top_margin
        self.overlap_min = overlap_min  # 2D path: share of the object's mask inside the support's box
        self.r_min = support_radius_min
        self.window = radius_window
        self.radii: dict[str, list[float]] = defaultdict(list)
        self.table_z: float | None = None

    def observe_sizes(self) -> None:
        """Record this frame's support radii (called after each detection)."""
        for a, t in self.tr.tracks.items():
            if t.seen_now and t.size is not None and t.held_by is None:
                self.radii[a].append(0.5 * float(max(t.size[0], t.size[1])))
                self.radii[a] = self.radii[a][-self.window:]
        zs = [float(t.pos[2] - 0.5 * t.size[2]) for t in self.tr.tracks.values()
              if t.pos is not None and t.size is not None and t.seen_now]
        if zs and self.table_z is None:
            self.table_z = min(zs)

    def radius(self, alias: str) -> float | None:
        r = self.radii.get(alias)
        if r:
            return max(float(np.median(r)), self.r_min)
        t = self.tr.tracks.get(alias)
        if t is None or t.size is None:
            return None
        return max(0.5 * float(max(t.size[0], t.size[1])), self.r_min)

    def pair(self, key: EdgeKey, cam: str | None = None, release_eef: np.ndarray | None = None,
             use_height: bool = True) -> dict[str, Any] | None:
        """``use_height=False`` (bggen-fix4 B: geometry of a release frame with no settled frame after it -- the object
        may still be falling): height is recorded but vetoes nothing -- ``inside`` = inside horizontally, ``dz`` moved to
        ``dz_unsettled`` (the flat veto then checks the horizontal distance only)."""
        pred, s, t = key
        if pred not in ("on", "in") or s not in self.tr.tracks:
            return None
        x = self.tr.tracks[s]
        if t == TABLE:
            if x.pos is None or x.size is None or self.table_z is None:
                return None
            return {"mode": "depth", "dz_table": round(float(x.pos[2] - 0.5 * x.size[2] - self.table_z), 3)}
        if t not in self.tr.tracks:
            return None
        y = self.tr.tracks[t]
        both_seen = x.seen_now and y.seen_now
        if x.pos is not None and y.pos is not None and both_seen:
            out: dict[str, Any] = {"mode": "depth", "d_xy": round(float(np.linalg.norm(x.pos[:2] - y.pos[:2])), 3),
                                   "dz": round(float(x.pos[2] - y.pos[2]), 3), "r": round(self.radius(t) or self.r_min, 3)}
            if is_container(y.category) and y.size is not None:
                # bggen-fix3 A2: inside = the object's center within the container's box horizontally and within
                # its 3D height range (the mask's lowest .. highest depth points + margins), not "dz to the
                # container's reference point" (a cheese 13 cm above a basket's median point can be inside it)
                half = 0.5 * np.asarray(y.size)[:2] + self.in_xy_margin
                in_xy = bool(np.all(np.abs(x.pos[:2] - y.pos[:2]) <= half))
                lo, hi = y.zlim if y.zlim is not None else (float(y.pos[2] - 0.5 * y.size[2]), float(y.pos[2] + 0.5 * y.size[2]))
                z = float(x.pos[2])
                in_z = lo - self.in_bottom_margin <= z <= hi + self.in_top_margin
                out.update(inside=in_xy and (in_z or not use_height), z_range=[round(lo, 3), round(hi, 3)], inside_xy=in_xy,
                           inside_z=in_z)
                out["positive"] = out["inside"]
            if not use_height:
                out["dz_unsettled"] = out.pop("dz")
                out["height_veto"] = "skipped: release frame, no settled frame (the object may still be falling)"
            return out
        out = {"mode": "2d"}
        if cam is not None and cam in x.box and cam in y.box and both_seen:
            out.update(self._mask2d(x, y, cam))
        if release_eef is not None and y.pos is not None:
            out["release_d_xy"] = round(float(np.linalg.norm(np.asarray(release_eef)[:2] - y.pos[:2])), 3)
            r = self.radius(t)
            if r is not None:
                out["r"] = round(r, 3)
        if not both_seen:
            out["unseen"] = [a for a, tt in ((s, x), (t, y)) if not tt.seen_now]
        # 2D evidence FOR the relation (never against it): released over the support, or the mask sits in its box
        released_over = "release_d_xy" in out and "r" in out and out["release_d_xy"] <= out["r"]
        if released_over or (out.get("overlap", 0.0) >= self.overlap_min and out.get("bottom_contact")):
            out["positive"] = True
        return out if len(out) > 1 else None

    @staticmethod
    def _mask2d(x: Any, y: Any, cam: str) -> dict[str, Any]:
        u0, v0, u1, v1 = y.box[cam]
        xu0, xv0, xu1, xv1 = x.box[cam]
        m = x.mask.get(cam)
        if m is not None:
            ys, xs = np.nonzero(m)
            inside = ((xs >= u0) & (xs <= u1) & (ys >= v0) & (ys <= v1)).mean() if len(xs) else 0.0
        else:
            iu = max(0.0, min(u1, xu1) - max(u0, xu0))
            iv = max(0.0, min(v1, xv1) - max(v0, xv0))
            inside = iu * iv / max((xu1 - xu0) * (xv1 - xv0), 1.0)
        h_overlap = max(0.0, min(u1, xu1) - max(u0, xu0)) / max(xu1 - xu0, 1.0)
        contact = bool(v0 <= xv1 <= v1 + 0.1 * (v1 - v0) and h_overlap > 0.5)
        return {"overlap": round(float(inside), 3), "bottom_contact": contact, "h_overlap": round(float(h_overlap), 3)}


class GeoMemory:
    """bggen-fix4 A: the last positive on / in geometry of each pair, valid while neither object has been grasped
    since and neither track has moved (``move_m`` in 3D, ``move_px`` on a fixed camera's pixels). The generator
    passes it to the verifier when an Observe 0 for in(X, C) comes at an event without positive geometry of its own
    (t08 d1 / d2: R9 2D geometry at the release, VDM 0 at the next keyframe, no geometry there -> R10 never ran).

    A depth geometry computed now that says "not inside" clears the entry (the objects were measured apart)."""

    def __init__(self, move_m: float = 0.04, move_px: float = 20.0, fixed_camera: bool = True) -> None:
        self.move_m, self.move_px, self.fixed_camera = move_m, move_px, fixed_camera
        self.entries: dict[EdgeKey, dict[str, Any]] = {}

    @staticmethod
    def _snap(tr: Any, t: int) -> dict[str, Any]:
        """Movement reference: only a position from a detection at this frame (a released object that is not detected
        has the hand's position, not its own: its fall into the container is no movement)."""
        if not tr.seen_now or tr.held_by is not None:
            return {"pos": None, "pixel": {}, "ref_t": None}
        return {"pos": None if tr.pos is None else np.asarray(tr.pos, float).copy(), "pixel": dict(tr.pixel), "ref_t": t}

    def refresh(self, tracks: dict[str, Any], t: int) -> None:
        """After each frame's detection: an object with no movement reference yet takes its first detection after the
        remembered geometry as the reference."""
        for ent in self.entries.values():
            for a, snap in ent["snap"].items():
                tr = tracks.get(a)
                if snap.get("ref_t") is None and tr is not None and t > ent["t"]:
                    new = self._snap(tr, t)
                    if new["ref_t"] is not None:
                        ent["snap"][a] = new

    def remember(self, key: EdgeKey, g: dict[str, Any], t: int, tracks: dict[str, Any], src: str,
                 positive: bool, snaps: dict[str, dict[str, Any]] | None = None) -> None:
        """Called for every computed on / in geometry. ``snaps``: the positions the geometry used (a settled frame's
        detections, bggen-fix4 B), else the tracks' current ones."""
        if key[0] not in ("on", "in") or key[2] is None or key[1] not in tracks or key[2] not in tracks:
            return
        if positive:
            snaps = snaps or {}
            self.entries[key] = {"t": t, "g": dict(g), "src": src,
                                 "snap": {a: snaps.get(a) or self._snap(tracks[a], t) for a in key[1:]}}
        elif g.get("mode") == "depth":
            # measured now in 3D (both objects detected) and not positive: the remembered positive no longer holds
            self.entries.pop(key, None)

    def lookup(self, key: EdgeKey, tracks: dict[str, Any], grasps: dict[str, list[int]], cam: str | None
               ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """(geometry to use, validity record). The geometry is the remembered one marked ``remembered``; None when
        there is no entry or it is no longer valid (the record says why)."""
        ent = self.entries.get(key)
        if ent is None:
            return None, {"valid": False, "why": "no positive geometry remembered"}
        t0 = ent["t"]
        rec: dict[str, Any] = {"from_t": t0, "src": ent["src"], "grasps_since": {}, "moved": {}, "movement": {}}
        for a in key[1:]:
            gs = [t for t in grasps.get(a, []) if t > t0]
            if gs:
                rec["grasps_since"][a] = gs
            tr = tracks.get(a)
            snap = ent["snap"].get(a) or {}
            if tr is None:
                rec["movement"][a] = "track gone"
                continue
            if tr.held_by is not None:
                rec["grasps_since"].setdefault(a, []).append("held now")
            if snap.get("pos") is not None and tr.pos is not None:
                d = float(np.linalg.norm(np.asarray(tr.pos, float) - snap["pos"]))
                rec["movement"][a] = f"{d:.3f} m"
                if d > self.move_m:
                    rec["moved"][a] = round(d, 3)
            elif self.fixed_camera and cam is not None and cam in snap.get("pixel", {}) and cam in tr.pixel:
                p0, p1 = snap["pixel"][cam], tr.pixel[cam]
                d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                rec["movement"][a] = f"{d:.1f} px"
                if d > self.move_px:
                    rec["moved"][a] = round(d, 1)
            elif snap.get("ref_t") is None:
                rec["movement"][a] = "unknown (not detected since)"
            else:
                rec["movement"][a] = "unknown (no comparable position)"
        if rec["grasps_since"] or rec["moved"]:
            rec["valid"] = False
            rec["why"] = "grasped since" if rec["grasps_since"] else "moved since"
            self.entries.pop(key, None)
            return None, rec
        rec["valid"] = True
        rec["why"] = "neither object grasped or moved since"
        return {**ent["g"], "positive": True, "remembered": rec}, rec
