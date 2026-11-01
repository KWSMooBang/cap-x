"""Detection on one camera frame: SAM3 text prompts per category, masks lifted to 3D with depth.

Same method as bg_est2's ``Sam3PerceptionSource`` (score >= 0.4, mask NMS 0.5, center = median of
the lifted mask points, size = 5-95 percentile extent), but on a :class:`records.CameraFrame` of any
camera, and the depth path is optional: without depth / intrinsics / pose a detection keeps its mask,
pixel and 2D box only (``center`` None), and the geometry tool uses the 2D path.

``FormatFakeDetector`` is a FAKE for format smoke runs on real data with no GPU: fixed boxes per
category. It says so loudly and its detections are never a valid perception result.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from capx.belief.perception import Category, mask_points

from .records import CameraFrame


@dataclass
class Detection:
    category: str
    score: float
    pixel: tuple[float, float]  # mask centroid (u, v)
    box: tuple[float, float, float, float]  # u0, v0, u1, v1
    center: np.ndarray | None = None  # (3,) base frame, m (depth path)
    size: np.ndarray | None = None  # (3,) m
    mask: np.ndarray | None = field(default=None, repr=False)
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "score": round(self.score, 3), "pixel": [round(x, 1) for x in self.pixel],
                "box": [round(x, 1) for x in self.box], "prompt": self.prompt,
                "center": None if self.center is None else [round(float(x), 4) for x in self.center],
                "size": None if self.size is None else [round(float(x), 4) for x in self.size]}


class Detector(Protocol):
    name: str
    calls: int

    def detect(self, cam: CameraFrame, categories: Sequence[Category]) -> list[Detection]: ...


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def detections_from_masks(cam: CameraFrame, cat: Category, results: Sequence[dict[str, Any]], score_threshold: float = 0.4,
                          nms_iou: float = 0.5, min_points: int = 30, max_depth: float = 8.0) -> list[Detection]:
    kept: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda r: -float(r.get("score", 0.0))):
        if float(r.get("score", 0.0)) < score_threshold:
            continue
        mask = np.asarray(r["mask"]).astype(bool)
        if mask.ndim == 3:
            mask = mask[0]
        if not mask.any() or any(_iou(mask, k["mask"]) > nms_iou for k in kept):
            continue
        kept.append({"mask": mask, "score": float(r["score"])})
    out = []
    for k in kept:
        ys, xs = np.nonzero(k["mask"])
        det = Detection(cat.key, k["score"], (float(xs.mean()), float(ys.mean())),
                        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), mask=k["mask"],
                        prompt=cat.sam3_text)
        if cam.has_geometry:
            pts = mask_points(k["mask"], cam.depth, cam.K, cam.T_base_cam, max_depth=max_depth)
            if len(pts) >= min_points:
                lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
                det.center, det.size = np.median(pts, axis=0), hi - lo
        out.append(det)
    return out


SAM3_URL = "http://127.0.0.1:8114"


def sam3_http_client(url: str | None = None, timeout_s: float = 120.0, retries: int = 3) -> Callable[..., list[dict[str, Any]]]:
    """``segment_fn(rgb, text_prompt) -> [{"mask", "score", "box", "label"}]`` for cap-x's SAM3 FastAPI service
    (``capx.serving.launch_sam3_server``: POST ``/segment`` {image_base64, text_prompt} -> {results: [{mask_base64,
    shape, box, score, label}]}). Same protocol as ``capx.integrations.vision.sam3``, but standalone: importing
    that module runs ``capx.integrations/__init__`` (every robot API -> ``capx.envs`` / simulators), which the
    generator does not need. URL: ``url``, else ``BGGEN_SAM3_URL``, else ``http://127.0.0.1:8114``."""
    import base64
    import io
    import os
    import time

    import requests
    from PIL import Image

    base = (url or os.environ.get("BGGEN_SAM3_URL") or SAM3_URL).rstrip("/")

    def segment_fn(image: np.ndarray, text_prompt: str) -> list[dict[str, Any]]:
        arr = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        buf = io.BytesIO()
        Image.fromarray(arr).convert("RGB").save(buf, format="PNG")
        payload = {"image_base64": base64.b64encode(buf.getvalue()).decode(), "text_prompt": text_prompt}
        last: Exception | None = None
        for k in range(retries + 1):
            try:
                r = requests.post(f"{base}/segment", json=payload, timeout=timeout_s)
                r.raise_for_status()
                items = r.json().get("results") or []
                break
            except Exception as exc:  # noqa: BLE001 -- service restarting: retry, then raise to the caller
                last = exc
                if k == retries:
                    raise RuntimeError(f"SAM3 service at {base} failed: {last}") from exc
                time.sleep(min(2.0 * 2 ** k, 20.0))
        out = []
        for it in items:
            mask = np.frombuffer(base64.b64decode(it["mask_base64"]), dtype=np.uint8).reshape(tuple(it["shape"])).astype(bool)
            out.append({"mask": mask, "box": it.get("box"), "score": it.get("score", 0.0), "label": it.get("label")})
        return out

    return segment_fn


class Sam3Detector:
    """``segment_fn(rgb, text_prompt=...) -> [{"mask", "score"}]``; default: the standalone SAM3 HTTP client
    (``sam3_http_client``; does not import ``capx.integrations`` / ``capx.envs``)."""

    name = "sam3"

    def __init__(self, segment_fn: Callable[..., list[dict[str, Any]]] | None = None, score_threshold: float = 0.4,
                 nms_iou: float = 0.5, min_points: int = 30, max_depth: float = 8.0) -> None:
        if segment_fn is None:
            segment_fn = sam3_http_client()
        self.segment_fn = segment_fn
        self.score_threshold, self.nms_iou, self.min_points, self.max_depth = score_threshold, nms_iou, min_points, max_depth
        self.calls = 0

    def detect(self, cam: CameraFrame, categories: Sequence[Category]) -> list[Detection]:
        out: list[Detection] = []
        for cat in categories:
            self.calls += 1
            try:
                res = self.segment_fn(cam.rgb, text_prompt=cat.sam3_text) or []
            except Exception as exc:  # service down: nothing this time
                print(f"[bggen] SAM3 failed for {cat.sam3_text!r}: {exc}", file=sys.stderr)
                res = []
            out += detections_from_masks(cam, cat, res, self.score_threshold, self.nms_iou, self.min_points,
                                         self.max_depth)
        return out


FAKE_BANNER = ("\n" + "!" * 78 + "\n!!  bggen detector=format_fake: FIXED BOXES, not perception. Format smoke only.  !!\n"
               + "!" * 78 + "\n")


class FormatFakeDetector:
    """FAKE: ``per_category`` boxes per category on a fixed grid of the image (format smoke only)."""

    name = "format_fake"

    def __init__(self, per_category: int = 2) -> None:
        self.per_category = per_category
        self.calls = 0
        print(FAKE_BANNER, file=sys.stderr)

    def detect(self, cam: CameraFrame, categories: Sequence[Category]) -> list[Detection]:
        h, w = cam.rgb.shape[:2]
        res = []
        for ci, cat in enumerate(categories):
            self.calls += 1
            masks = []
            for j in range(self.per_category):
                u = int(w * (0.2 + 0.6 * (j + 0.5) / self.per_category))
                v = int(h * (0.3 + 0.4 * (ci + 0.5) / max(len(categories), 1)))
                m = np.zeros((h, w), bool)
                m[max(v - h // 20, 0): v + h // 20, max(u - w // 20, 0): u + w // 20] = True
                masks.append({"mask": m, "score": 0.9})
            res += detections_from_masks(cam, cat, masks)
        return res
