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

    def pair(self, key: EdgeKey, cam: str | None = None, release_eef: np.ndarray | None = None) -> dict[str, Any] | None:
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
                out.update(inside=in_xy and in_z, z_range=[round(lo, 3), round(hi, 3)], inside_xy=in_xy, inside_z=in_z)
                out["positive"] = out["inside"]
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
