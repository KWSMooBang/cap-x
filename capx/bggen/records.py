"""Common frame records and the on-disk episode format of the BG generator (``capx.bggen``).

Every benchmark adapter (``adapters/``) turns a demonstration into the same episode directory, and the
generator reads only that directory through :class:`Episode`:

    <root>/episode/
        meta.json               instruction, fps, n_frames, arms, cameras (+ field notes from the adapter)
        proprio.npz             per frame (N = n_frames): t, eef_pos_<arm> (N,3) and eef_quat_<arm> (N,4, wxyz)
                                in the robot base frame, gripper_<arm> (N,) finger opening in m,
                                optional base_pose (N,4,4) world <- base, optional action (N,A)
        cams/<cam>/index.npz    t (M,) frames with an image, K (3,3), T (M,4,4) base <- camera (OpenCV axes:
                                x right, y down, z forward), or no K/T when unknown
        cams/<cam>/rgb/<t:06d>.png
        cams/<cam>/depth/<t:06d>.npy   float16 metres along the optical axis (missing: no depth path)
    <root>/oracle/              EVALUATION ONLY (``capx.bggen.oracle``); the generator never opens it.

All 3D quantities the generator uses are in a world frame: the base frame itself for a fixed robot
(no ``base_pose``), ``base_pose @ x_base`` for a mobile one.

Ground truth never travels with :class:`Episode`: adapters write it to ``oracle/`` and the evaluation
code loads it with ``capx.bggen.oracle.load_oracle``. ``Episode`` refuses any attribute that looks
like ground truth (tests check that the generator never touches ``oracle/``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORBIDDEN_ATTRS = ("oracle", "truth", "task_info", "annotations", "sim_state", "states")


@dataclass
class CameraFrame:
    name: str
    t: int  # frame the image comes from (<= the requested frame: causal)
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None = None  # (H, W) float32 m
    K: np.ndarray | None = None  # (3, 3)
    T_base_cam: np.ndarray | None = None  # (4, 4) base <- camera, OpenCV axes

    @property
    def has_geometry(self) -> bool:
        return self.depth is not None and self.K is not None and self.T_base_cam is not None


@dataclass
class ArmState:
    gripper: float  # finger opening, m (0 = closed on nothing)
    eef_pos: np.ndarray  # (3,) base frame
    eef_quat: np.ndarray | None = None  # (4,) wxyz, base frame


@dataclass
class FrameRecord:
    t: int
    time_s: float
    arms: dict[str, ArmState]
    base_pose: np.ndarray | None = None  # (4, 4) world <- base (mobile robots)
    action: np.ndarray | None = None
    cams: dict[str, CameraFrame] = field(default_factory=dict)

    def world(self, p_base: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_base, dtype=np.float64)
        if self.base_pose is None:
            return p
        return (self.base_pose @ np.array([*p, 1.0]))[:3]

    def eef_world(self, arm: str) -> np.ndarray:
        return self.world(self.arms[arm].eef_pos)


ImageLoader = Callable[[str, int], tuple[np.ndarray, np.ndarray | None]]  # (cam, t) -> rgb, depth


class Episode:
    """Read-only view of one demonstration: instruction + robot state + camera frames.

    ``proprio``: dict of arrays (see module docstring). ``cams``: cam -> {"t": (M,), "K": (3,3)|None,
    "T": (M,4,4)|None}. Images come from ``image_loader(cam, t)`` for a stored ``t``.
    """

    def __init__(self, meta: dict[str, Any], proprio: dict[str, np.ndarray], cams: dict[str, dict[str, Any]],
                 image_loader: ImageLoader) -> None:
        self.meta = dict(meta)
        self._p = {k: np.asarray(v) for k, v in proprio.items()}
        self._cams = cams
        self._load = image_loader
        self.image_reads = 0

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_ATTRS or name.startswith("oracle"):
            raise AttributeError(f"Episode has no {name!r}: ground truth is only in capx.bggen.oracle (evaluation)")
        raise AttributeError(name)

    # --- meta --------------------------------------------------------------------------------
    @property
    def instruction(self) -> str:
        return str(self.meta["instruction"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def arms(self) -> list[str]:
        return list(self.meta["arms"])

    @property
    def cameras(self) -> list[str]:
        return list(self._cams)

    @property
    def mobile(self) -> bool:
        return "base_pose" in self._p

    def camera_frames(self, cam: str) -> np.ndarray:
        return np.asarray(self._cams[cam]["t"], dtype=int)

    def has_depth(self, cam: str) -> bool:
        return bool(self.meta.get("cameras", {}).get(cam, {}).get("depth", False)) and self._cams[cam].get("K") is not None

    # --- robot state (every frame) -------------------------------------------------------------
    def gripper(self, arm: str) -> np.ndarray:
        return self._p[f"gripper_{arm}"].astype(np.float64)

    def eef(self, arm: str) -> np.ndarray:
        return self._p[f"eef_pos_{arm}"].astype(np.float64)

    def base_poses(self) -> np.ndarray | None:
        return self._p.get("base_pose")

    def robot(self, t: int) -> FrameRecord:
        t = int(np.clip(t, 0, self.n_frames - 1))
        arms = {}
        for a in self.arms:
            q = self._p.get(f"eef_quat_{a}")
            arms[a] = ArmState(float(self._p[f"gripper_{a}"][t]), self._p[f"eef_pos_{a}"][t].astype(np.float64),
                               None if q is None else q[t].astype(np.float64))
        base = self._p.get("base_pose")
        act = self._p.get("action")
        return FrameRecord(t, t / self.fps, arms, None if base is None else base[t].astype(np.float64),
                           None if act is None else act[t])

    # --- images (causal: the latest stored image at or before t) ------------------------------
    def image_t(self, cam: str, t: int, causal: bool = True) -> int | None:
        ts = self.camera_frames(cam)
        if len(ts) == 0:
            return None
        if causal:
            i = int(np.searchsorted(ts, t, side="right")) - 1
            return int(ts[i]) if i >= 0 else None
        return int(ts[int(np.argmin(np.abs(ts - t)))])

    def camera(self, cam: str, t: int, causal: bool = True) -> CameraFrame | None:
        ti = self.image_t(cam, t, causal)
        if ti is None:
            return None
        rgb, depth = self._load(cam, ti)
        self.image_reads += 1
        c = self._cams[cam]
        K = c.get("K")
        T = None
        if c.get("T") is not None:
            idx = int(np.searchsorted(self.camera_frames(cam), ti))
            T = np.asarray(c["T"][idx], dtype=np.float64)
        return CameraFrame(cam, ti, rgb, depth if self.has_depth(cam) else None,
                           None if K is None else np.asarray(K, dtype=np.float64), T)

    def frame(self, t: int, cams: Sequence[str] = (), causal: bool = True) -> FrameRecord:
        fr = self.robot(t)
        for c in cams:
            cf = self.camera(c, t, causal)
            if cf is not None:
                fr.cams[c] = cf
        return fr


# --- on-disk format ----------------------------------------------------------------------------

def episode_dir(root: str | Path) -> Path:
    return Path(root) / "episode"


def write_episode(root: str | Path, meta: dict[str, Any], proprio: dict[str, np.ndarray],
                  cams: dict[str, dict[str, Any]]) -> Path:
    """Write ``meta.json``, ``proprio.npz`` and per camera ``index.npz`` (images are written separately
    with :func:`write_image`). ``cams``: cam -> {"t", "K", "T"} (K / T may be None)."""
    d = episode_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default))
    np.savez_compressed(d / "proprio.npz", **proprio)
    for cam, c in cams.items():
        cd = d / "cams" / cam
        (cd / "rgb").mkdir(parents=True, exist_ok=True)
        arrays = {"t": np.asarray(c["t"], dtype=np.int64)}
        if c.get("K") is not None:
            arrays["K"] = np.asarray(c["K"], dtype=np.float64)
        if c.get("T") is not None:
            arrays["T"] = np.asarray(c["T"], dtype=np.float64)
        np.savez_compressed(cd / "index.npz", **arrays)
    return d


def write_image(root: str | Path, cam: str, t: int, rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
    from PIL import Image

    cd = episode_dir(root) / "cams" / cam
    (cd / "rgb").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(cd / "rgb" / f"{int(t):06d}.png")
    if depth is not None:
        (cd / "depth").mkdir(parents=True, exist_ok=True)
        np.save(cd / "depth" / f"{int(t):06d}.npy", np.asarray(depth, dtype=np.float16))  # ~1 mm at 1 m


def load_episode(root: str | Path) -> Episode:
    d = episode_dir(root)
    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "proprio.npz") as z:
        proprio = {k: z[k] for k in z.files}
    cams: dict[str, dict[str, Any]] = {}
    for cam in meta.get("cameras", {}):
        cd = d / "cams" / cam
        if not (cd / "index.npz").exists():
            continue
        with np.load(cd / "index.npz") as z:
            cams[cam] = {"t": z["t"], "K": z["K"] if "K" in z.files else None, "T": z["T"] if "T" in z.files else None}

    def loader(cam: str, t: int) -> tuple[np.ndarray, np.ndarray | None]:
        from PIL import Image

        cd = d / "cams" / cam
        rgb = np.asarray(Image.open(cd / "rgb" / f"{t:06d}.png").convert("RGB"))
        dp = cd / "depth" / f"{t:06d}.npy"
        return rgb, (np.load(dp).astype(np.float32) if dp.exists() else None)

    return Episode(meta, proprio, cams, loader)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
