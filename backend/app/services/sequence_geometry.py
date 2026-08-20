"""
sequence_geometry.py
~~~~~~~~~~~~~~~~~~~~~
Pure geometry helpers for sequence-detection step matching. No I/O, no
model calls — just box/line math, so it can be unit-tested against fake
coordinates without a camera or a trained model.

All coordinates are normalized (0-1) unless noted otherwise.
"""

from __future__ import annotations


def box_from_region_coords(region_coords: list[float]) -> tuple[float, float, float, float]:
    """A "box" region is already stored as [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = region_coords
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def overlap_ratio(detected_xyxy: tuple[float, float, float, float],
                   region_xyxy: tuple[float, float, float, float]) -> float:
    """Fraction of the DETECTED box's own area that overlaps the region box.

    This is the right metric for "is the finger on this button" — a small
    finger box sitting mostly inside a bigger (or similarly sized) key
    region scores close to 1.0, without being punished for not filling
    the entire region the way a plain IoU would.
    """
    dx1, dy1, dx2, dy2 = detected_xyxy
    rx1, ry1, rx2, ry2 = region_xyxy

    ix1, iy1 = max(dx1, rx1), max(dy1, ry1)
    ix2, iy2 = min(dx2, rx2), min(dy2, ry2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    detected_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
    if detected_area <= 0:
        return 0.0
    return inter_area / detected_area


def point_in_box(point: tuple[float, float], box_xyxy: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def segments_intersect(p1: tuple[float, float], p2: tuple[float, float],
                        p3: tuple[float, float], p4: tuple[float, float]) -> bool:
    """True if segment p1->p2 (the object's motion this frame) crosses
    segment p3->p4 (a drawn "line" region) — for tripwire-style regions."""

    def orientation(a, b, c) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c) -> bool:
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    return False


def step_matches(step: dict,
                  detected_xyxy: tuple[float, float, float, float],
                  prev_center: tuple[float, float] | None,
                  threshold: float) -> bool:
    """Does this frame's detected box satisfy the given sequence step?

    Box regions: overlap_ratio(detected, region) >= threshold.
    Line regions: the object's center crossed the line segment since the
    previous frame (needs prev_center; first frame with no history never
    matches a line region).
    """
    region_type = step["region_type"]
    coords = step["region_coords"]

    if region_type == "box":
        region_xyxy = box_from_region_coords(coords)
        return overlap_ratio(detected_xyxy, region_xyxy) >= threshold

    if region_type == "line":
        if prev_center is None:
            return False
        dx1, dy1, dx2, dy2 = detected_xyxy
        curr_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
        p3, p4 = (coords[0], coords[1]), (coords[2], coords[3])
        return segments_intersect(prev_center, curr_center, p3, p4)

    return False
