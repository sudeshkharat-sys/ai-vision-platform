"""
test_variants.py
~~~~~~~~~~~~~~~~
Label-safe variants of a failed image (same size, nothing moves, no flips), so
the boxes of the original stay exactly right:

  1. brightness / contrast shift (darker or lighter, lower contrast)
  2. blur plus a down/up-scale (soft, low-resolution look)
  3. gamma + a little noise
"""

from __future__ import annotations

import re

import cv2
import numpy as np

MAX_VARIANTS = 3


def make_variants(img: np.ndarray, n: int, seed: int = 0) -> list:
    """Return ``n`` (0..3) variants of a BGR image, same shape as the input."""
    rng = np.random.default_rng(seed)
    n = max(0, min(int(n), MAX_VARIANTS))
    h, w = img.shape[:2]
    out = []
    if n >= 1:
        alpha = float(rng.choice([rng.uniform(0.6, 0.85), rng.uniform(1.15, 1.4)]))   # contrast
        beta = float(rng.uniform(-35, 35))                                           # brightness
        out.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    if n >= 2:
        k = int(rng.choice([3, 5]))
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        f = float(rng.uniform(0.4, 0.65))
        small = cv2.resize(blurred, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
    if n >= 3:
        g = float(rng.uniform(0.7, 1.5))
        lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
        noisy = cv2.LUT(img, lut).astype(np.int16) + rng.normal(0, 6, img.shape).astype(np.int16)
        out.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return out


def make_zoom(img: np.ndarray, boxes: list, seed: int = 0):
    """Zoom-in copy: crop a random window and resize back to the original size.

    ``boxes`` are normalized [cx, cy, w, h]. The window always contains every
    box completely, so no label is cut or lost; returns ``None`` when the
    labelled objects fill too much of the image to zoom (>~92%).
    Returns ``(zoomed image, (x0, y0, sw, sh))`` - the normalized window, used
    to remap labels with ``remap_norm``."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if boxes:
        xs0 = min(b[0] - b[2] / 2 for b in boxes); xs1 = max(b[0] + b[2] / 2 for b in boxes)
        ys0 = min(b[1] - b[3] / 2 for b in boxes); ys1 = max(b[1] + b[3] / 2 for b in boxes)
        xs0, ys0, xs1, ys1 = max(0.0, xs0), max(0.0, ys0), min(1.0, xs1), min(1.0, ys1)
    else:
        xs0 = ys0 = 0.0; xs1 = ys1 = 1.0
    scale = max(float(rng.uniform(0.6, 0.85)), (xs1 - xs0) + 0.02, (ys1 - ys0) + 0.02)
    if scale > 0.92:
        return None
    # window start range that still contains the union box
    x_lo, x_hi = max(0.0, xs1 - scale), min(1.0 - scale, xs0)
    y_lo, y_hi = max(0.0, ys1 - scale), min(1.0 - scale, ys0)
    if x_lo > x_hi or y_lo > y_hi:
        return None
    x0 = float(rng.uniform(x_lo, x_hi)); y0 = float(rng.uniform(y_lo, y_hi))
    px0, py0 = int(round(x0 * w)), int(round(y0 * h))
    px1, py1 = min(w, px0 + max(2, int(round(scale * w)))), min(h, py0 + max(2, int(round(scale * h))))
    crop = img[py0:py1, px0:px1]
    # use the real pixel window so labels stay exact after rounding
    win = (px0 / w, py0 / h, (px1 - px0) / w, (py1 - py0) / h)
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC), win


def remap_bbox(bbox, win):
    x0, y0, sw, sh = win
    return [min(1.0, max(0.0, (bbox[0] - x0) / sw)), min(1.0, max(0.0, (bbox[1] - y0) / sh)),
            min(1.0, bbox[2] / sw), min(1.0, bbox[3] / sh)]


def remap_points(points, win):
    x0, y0, sw, sh = win
    return [[min(1.0, max(0.0, (p[0] - x0) / sw)), min(1.0, max(0.0, (p[1] - y0) / sh))] for p in points]


_FAILED_PREFIX = re.compile(r"^failed_.*?_\d{8}_\d{4}_")
_VARIANT_SUFFIX = re.compile(r"_[vz]\d$")


def failed_copy_key(filename: str):
    """For a ``failed_<model>_<date>_<time>_<name>[_vN].jpg`` copy return
    ``(original stem, variant suffix)`` (nested prefixes from repeated
    add-backs are stripped); ``None`` for a normal image."""
    stem = re.sub(r"\.[^.]+$", "", filename or "")
    if not stem.startswith("failed_"):
        return None
    while _FAILED_PREFIX.match(stem):
        stem = _FAILED_PREFIX.sub("", stem, count=1)
    m = _VARIANT_SUFFIX.search(stem)
    suffix = m.group(0) if m else ""
    return (stem[:len(stem) - len(suffix)], suffix)
