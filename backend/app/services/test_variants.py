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
