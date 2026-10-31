"""Perception for the estimated Belief Graph (``bg_est``): camera detections -> 3D object estimates.

Only robot-usable inputs: the agentview RGB-D image with its intrinsics and extrinsics (what
``get_observation()`` returns), the robot's end-effector pose, and the task language.

* Object categories come from the task language through a small vocabulary
  (``categories_from_instruction``), e.g. "swap the 2 bowls on their plates" -> bowl, plate.
* ``Sam3PerceptionSource`` runs a SAM3 text prompt per category on the agentview RGB image and lifts
  each instance mask with depth + intrinsics + extrinsics to 3D points in the robot base frame (the
  frame the generated code works in): center = median point, size = 5-95 percentile extent.
* ``SimPerceptionSource`` is a FAKE detector for tests and CPU checks: simulator poses with noise,
  dropped detections and arm occlusion. It must never be used for a real evaluation; it announces
  itself loudly and memory logs record ``perception=sim_fake``.

Detections carry only a category, never a simulator name.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Category:
    key: str  # alias base, e.g. "bowl" -> bowl_A, bowl_B
    prompt: str  # SAM3 text prompt
    keywords: tuple[str, ...]  # matched in the task language (substring, lower case)
    size: tuple[float, float, float]  # nominal extent (m), used by the fake detector only
    container: bool = False  # In(x, this) instead of On(x, this)
    sim_keys: tuple[str, ...] = ()  # fake detector only: substrings of simulator object names


# Objects of LIBERO / LIBERO-Mem scenes. Order matters only for overlapping keywords (first wins).
VOCABULARY: tuple[Category, ...] = (
    Category("cream_cheese", "cream cheese box", ("cream cheese",), (0.10, 0.06, 0.04), sim_keys=("cream_cheese",)),
    Category("bowl", "black bowl", ("bowl",), (0.13, 0.13, 0.05), sim_keys=("bowl",)),
    Category("plate", "plate", ("plate",), (0.20, 0.20, 0.02), sim_keys=("plate",)),
    Category("bottle", "wine bottle", ("bottle",), (0.07, 0.07, 0.28), sim_keys=("bottle",)),
    Category("basket", "basket", ("basket",), (0.25, 0.18, 0.10), container=True, sim_keys=("basket",)),
    Category("mug", "mug", ("mug", "cup"), (0.10, 0.08, 0.10), sim_keys=("mug", "cup")),
    Category("ketchup", "ketchup bottle", ("ketchup",), (0.06, 0.04, 0.15), sim_keys=("ketchup",)),
    Category("soup", "alphabet soup can", ("soup",), (0.07, 0.07, 0.10), sim_keys=("soup",)),
    Category("butter", "butter box", ("butter",), (0.08, 0.05, 0.03), sim_keys=("butter",)),
    Category("milk", "milk carton", ("milk",), (0.07, 0.07, 0.15), sim_keys=("milk",)),
    Category("juice", "orange juice carton", ("juice",), (0.07, 0.07, 0.15), sim_keys=("juice",)),
    Category("tray", "tray", ("tray",), (0.30, 0.20, 0.03), container=True, sim_keys=("tray",)),
)


def categories_from_instruction(
    instruction: str,
    max_categories: int = 5,
    vocabulary: Sequence[Category] = VOCABULARY,
) -> list[Category]:
    """Categories named in the task language, in order of first mention, at most ``max_categories``."""
    text = instruction.lower()
    found: list[tuple[int, Category]] = []
    taken: list[tuple[int, int]] = []  # spans already matched by an earlier (longer) keyword
    for cat in vocabulary:
        for kw in cat.keywords:
            m = re.search(re.escape(kw), text)
            if m is None or any(a <= m.start() < b for a, b in taken):
                continue
            taken.append(m.span())
            found.append((m.start(), cat))
            break
    found.sort(key=lambda x: x[0])
    return [c for _, c in found][:max_categories]


@dataclass
class Detection:
    category: str  # Category.key
    center: np.ndarray  # (3,) robot base frame, m
    size: np.ndarray  # (3,) extent, m
    score: float
    pixel: tuple[float, float] | None = None  # mask centroid (u, v) in the agentview image


class PerceptionSource(Protocol):
    name: str
    calls: int  # detector calls so far (SAM3: one per category per detect())

    def detect(self, obs: dict[str, Any], categories: Sequence[Category]) -> list[Detection]: ...


# robot_cartesian_pos (LIBERO Franka) is this far behind the fingertip centre along the gripper's
# approach axis (local +z): measured on LIBERO-Mem, a goto_pose fingertip target 0.030 m above the
# table reads z 0.138 top-down.
TCP_OFFSET = 0.108


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) or 1.0)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def proprio_from_obs(obs: dict[str, Any], tcp_offset: float = TCP_OFFSET) -> tuple[np.ndarray, float] | None:
    """(fingertip centre xyz in the robot base frame, gripper 0 closed .. 1 open) from
    ``get_observation()``: ``robot_cartesian_pos`` moved ``tcp_offset`` along the approach axis."""
    cart = obs.get("robot_cartesian_pos") if isinstance(obs, dict) else None
    if cart is None:
        return None
    cart = np.asarray(cart, dtype=np.float64)
    pos = cart[:3].copy()
    if len(cart) >= 8:
        pos = pos + _quat_wxyz_to_mat(cart[3:7]) @ np.array([0.0, 0.0, tcp_offset])
    return pos, float(cart[-1])


def mask_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T: np.ndarray, max_depth: float = 3.0) -> np.ndarray:
    """Mask pixels -> (N, 3) points in the frame of the camera pose ``T`` (camera-to-base, OpenCV axes)."""
    if depth.ndim == 3:
        depth = depth[..., 0]
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs].astype(np.float64)
    ok = (z > 0) & (z < max_depth)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    return (pts @ np.asarray(T, dtype=np.float64).T)[:, :3]


def project_to_pixel(pos: Sequence[float], K: np.ndarray, T: np.ndarray) -> tuple[float, float] | None:
    """Base-frame point -> (u, v) pixel of the camera with pose ``T`` (None if behind the camera)."""
    p = np.linalg.inv(np.asarray(T, dtype=np.float64)) @ np.array([*pos, 1.0])
    if p[2] <= 1e-6:
        return None
    return float(K[0, 0] * p[0] / p[2] + K[0, 2]), float(K[1, 1] * p[1] / p[2] + K[1, 2])


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


class Sam3PerceptionSource:
    """SAM3 text prompt per category on the agentview RGB, masks lifted to 3D with depth.

    ``segment_fn(rgb, text_prompt=...) -> [{"mask", "score", ...}]`` defaults to cap-x's SAM3 client
    (``capx.integrations.vision.sam3.init_sam3``, service at 127.0.0.1:8114)."""

    name = "sam3"

    def __init__(
        self,
        segment_fn: Callable[..., list[dict[str, Any]]] | None = None,
        camera: str = "agentview",
        score_threshold: float = 0.4,
        min_points: int = 30,
        nms_iou: float = 0.5,
    ) -> None:
        if segment_fn is None:
            from capx.integrations.vision.sam3 import init_sam3

            segment_fn = init_sam3()
        self.segment_fn = segment_fn
        self.camera = camera
        self.score_threshold = score_threshold
        self.min_points = min_points
        self.nms_iou = nms_iou
        self.calls = 0

    def detect(self, obs: dict[str, Any], categories: Sequence[Category]) -> list[Detection]:
        cam = obs[self.camera]
        rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
        K, T = np.asarray(cam["intrinsics"]), np.asarray(cam["pose_mat"])
        out: list[Detection] = []
        for cat in categories:
            self.calls += 1
            try:
                results = self.segment_fn(rgb, text_prompt=cat.prompt) or []
            except Exception as exc:  # the service is down: no detections this time
                print(f"[bg_est] SAM3 failed for {cat.prompt!r}: {exc}", file=sys.stderr)
                results = []
            kept: list[dict[str, Any]] = []
            for r in sorted(results, key=lambda r: -float(r.get("score", 0.0))):
                if float(r.get("score", 0.0)) < self.score_threshold:
                    continue
                mask = np.asarray(r["mask"]).astype(bool)
                if any(_mask_iou(mask, k["mask"]) > self.nms_iou for k in kept):
                    continue
                kept.append({"mask": mask, "score": float(r["score"])})
            for k in kept:
                pts = mask_points(k["mask"], depth, K, T)
                if len(pts) < self.min_points:
                    continue
                lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
                ys, xs = np.nonzero(k["mask"])
                out.append(Detection(cat.key, np.median(pts, axis=0), hi - lo, k["score"],
                                     (float(xs.mean()), float(ys.mean()))))
        return out


SIM_FAKE_BANNER = (
    "\n" + "!" * 78 + "\n"
    "!!  bg_est perception=sim_fake: detections are SIMULATOR POSES + noise.          !!\n"
    "!!  For tests and CPU checks only. NOT a valid evaluation of the estimated graph. !!\n"
    + "!" * 78 + "\n"
)


@dataclass
class SimPerceptionSource:
    """FAKE detector: simulator object positions with noise, dropouts and arm occlusion.

    ``truth`` needs ``movable_objects()`` and ``positions()`` (e.g. ``LiberoMemSource``). Detections
    carry the category only. Never use for a real evaluation (see module docstring)."""

    truth: Any
    noise_std: float = 0.01  # m, per axis
    dropout: float = 0.1  # chance an object is missed
    occlusion: float = 0.5  # chance an object under the gripper (xy within occlusion_radius) is missed
    occlusion_radius: float = 0.08
    held_radius: float = 0.06  # an object this close to the EEF is in the hand and always seen
    seed: int = 0
    vocabulary: Sequence[Category] = VOCABULARY
    name: str = "sim_fake"
    calls: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        print(SIM_FAKE_BANNER, file=sys.stderr)

    def _category(self, sim_name: str, categories: Sequence[Category]) -> Category | None:
        for cat in categories:
            if any(k in sim_name for k in cat.sim_keys):
                return cat
        return None

    def detect(self, obs: dict[str, Any], categories: Sequence[Category]) -> list[Detection]:
        self.calls += len(categories)
        positions = self.truth.positions()
        proprio = proprio_from_obs(obs)
        eef = proprio[0] if proprio is not None else None
        out: list[Detection] = []
        for sim_name in self.truth.movable_objects():
            cat = self._category(sim_name, categories)
            if cat is None or sim_name not in positions:
                continue
            pos = np.asarray(positions[sim_name], dtype=np.float64)
            held = eef is not None and np.linalg.norm(pos - eef) < self.held_radius
            if not held:
                if self.rng.random() < self.dropout:
                    continue
                if (eef is not None and np.linalg.norm(pos[:2] - eef[:2]) < self.occlusion_radius
                        and eef[2] > pos[2] and self.rng.random() < self.occlusion):
                    continue
            size = np.asarray(cat.size) * (1.0 + self.rng.normal(0.0, 0.05, 3))
            # Body origins of LIBERO objects are at their bottom; a camera sees the middle.
            center = pos + np.array([0.0, 0.0, 0.5 * cat.size[2]]) + self.rng.normal(0.0, self.noise_std, 3)
            out.append(Detection(cat.key, center, np.abs(size), 0.9))
        order = self.rng.permutation(len(out))  # detection order says nothing about identity
        return [out[i] for i in order]
