"""Set-of-Mark images for the estimated Belief Graph v2 (``bg_est2``).

Every track gets a short mark for the whole trial: supports (plates, baskets, trays) get numbers
1, 2, 3, ...; other objects capital letters A, B, ... . Marks are given in alias order, and aliases
are given from the image left (robot base -y) at turn 0, so mark A is bowl_A and mark 1 is plate_A.
A track that appears later gets the next free mark.

``draw_marks`` draws each mark at the track's image center (the projection of its estimated 3D
center with the camera intrinsics / extrinsics) as a white label on a dark disc with a thin white
ring, readable on black bowls and on white plates. Tracks not detected in this image get no mark.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

SUPPORT_KEYS = ("plate", "basket", "tray")  # categories that hold other objects: numbered marks


def is_support(category_key: str) -> bool:
    return category_key in SUPPORT_KEYS


@dataclass
class MarkTable:
    """alias <-> mark, fixed for a trial."""

    by_alias: dict[str, str] = field(default_factory=dict)

    def assign(self, alias: str, category_key: str) -> str:
        if alias in self.by_alias:
            return self.by_alias[alias]
        used = set(self.by_alias.values())
        if is_support(category_key):
            n = 1
            while str(n) in used:
                n += 1
            mark = str(n)
        else:
            mark = next((c for c in string.ascii_uppercase if c not in used), None)
            if mark is None:  # more than 26 objects: two letters
                mark = next(a + b for a in string.ascii_uppercase for b in string.ascii_uppercase if a + b not in used)
        self.by_alias[alias] = mark
        return mark

    def assign_all(self, tracks: Sequence[tuple[str, str]]) -> None:
        """``(alias, category key)`` in alias order."""
        for alias, cat in sorted(tracks):
            self.assign(alias, cat)

    def mark(self, alias: str) -> str | None:
        return self.by_alias.get(alias)

    def alias(self, mark: str) -> str | None:
        for a, m in self.by_alias.items():
            if m.lower() == mark.lower():
                return a
        return None

    def ordered(self) -> list[tuple[str, str]]:
        """(mark, alias): letters first, then numbers in numeric order."""
        def key(item: tuple[str, str]) -> tuple[int, int, str]:
            m = item[1]
            return (1, int(m), "") if m.isdigit() else (0, len(m), m)

        return [(m, a) for a, m in sorted(self.by_alias.items(), key=key)]


def _font(size: int):
    from PIL import ImageFont

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def draw_marks(rgb: np.ndarray, marks: Sequence[tuple[str, tuple[float, float]]], scale: float | None = None) -> np.ndarray:
    """Copy of ``rgb`` with each ``(mark, (u, v))`` drawn at its pixel. Marks outside the image are skipped."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy())
    h, w = img.height, img.width
    s = scale if scale is not None else max(min(h, w) / 256.0, 0.5)
    radius = max(int(round(9 * s)), 5)
    font = _font(max(int(round(12 * s)), 8))
    draw = ImageDraw.Draw(img)
    for mark, (u, v) in marks:
        if not (0 <= u < w and 0 <= v < h):
            continue
        r = radius + (3 * len(mark) - 3 if len(mark) > 1 else 0)
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(20, 20, 160), outline=(255, 255, 255), width=max(1, int(s)))
        bbox = draw.textbbox((0, 0), mark, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((u - tw / 2 - bbox[0], v - th / 2 - bbox[1]), mark, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def mark_radius(shape: tuple[int, ...]) -> int:
    s = max(min(shape[0], shape[1]) / 256.0, 0.5)
    return max(int(round(9 * s)), 5)


def spread_marks(marks: Sequence[tuple[str, tuple[float, float]]], shape: tuple[int, ...]) -> list[tuple[str, tuple[float, float]]]:
    """Move a mark that would cover an earlier one (e.g. a cheese box inside a basket: the same center)
    down by one mark diameter until it is free, so every mark stays readable. The returned pixels are the
    drawn ones (prompts report these)."""
    r = mark_radius(shape)
    placed: list[tuple[str, tuple[float, float]]] = []
    for m, (u, v) in marks:
        for _ in range(6):
            if all((u - pu) ** 2 + (v - pv) ** 2 >= (2 * r + 2) ** 2 for _, (pu, pv) in placed):
                break
            v = v + 2 * r + 2 if v + 2 * r + 2 < shape[0] else v - 2 * r - 2
        placed.append((m, (u, v)))
    return placed


def pixels_in_image(pixels: dict[str, tuple[float, float] | None], shape: tuple[int, ...]) -> dict[str, tuple[float, float]]:
    h, w = shape[0], shape[1]
    return {a: p for a, p in pixels.items() if p is not None and 0 <= p[0] < w and 0 <= p[1] < h}
