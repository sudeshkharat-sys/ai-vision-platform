"""
sequence_match.py
~~~~~~~~~~~~~~~~~~
Shared step-evaluation logic for both the video state machine
(sequence_engine.py) and the single-image tester (sequence_test.py), so
their matching rules never drift apart.

Implements the spec's intersection formula exactly:

    Intersection % = Intersection Area / Target Area x 100

using real polygon-vs-polygon pixel intersection (rasterized), not just
bounding-box overlap — segmentation masks are compared as actual masks.

Two step target types:
  - "region": Target = a user-drawn region (box, treated as a 4-point
    polygon). Object = a detected class's mask (its real predicted mask
    if the model produced one, else its bbox corners as a fallback
    polygon when only a plain detector, not a segmenter, saw it).
  - "detection_class": Target = another detected class's OWN mask (e.g.
    "M"'s predicted mask). Object = the intersection class's mask (e.g.
    "fingertip"). Neither polygon is drawn by the user — both come from
    the model's own predictions this frame.

Line regions are NOT handled here — a line crossing needs motion between
frames, which is state sequence_engine.py already tracks separately.
"""

from __future__ import annotations


def bbox_to_polygon(xyxy: tuple[float, float, float, float]) -> list[list[float]]:
    x1, y1, x2, y2 = xyxy
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# Half-width (normalized 0-1) of the thin rectangle a line region is
# buffered into for area-overlap matching — a true zero-width line has no
# area, so "Intersection Area / Target Area" needs a strip to compare
# against, not a geometric line.
LINE_OVERLAP_HALF_WIDTH = 0.02


def line_to_polygon(x1: float, y1: float, x2: float, y2: float,
                     half_width: float = LINE_OVERLAP_HALF_WIDTH) -> list[list[float]]:
    """Buffer a line segment into a thin rectangle (quad) so it can be
    matched with the same pixel-intersection formula a box region uses."""
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        # Degenerate (zero-length) line — fall back to a small square
        # around the single point so the polygon still has real area.
        return [
            [x1 - half_width, y1 - half_width], [x1 + half_width, y1 - half_width],
            [x1 + half_width, y1 + half_width], [x1 - half_width, y1 + half_width],
        ]
    nx, ny = -dy / length * half_width, dx / length * half_width
    return [
        [x1 + nx, y1 + ny], [x2 + nx, y2 + ny],
        [x2 - nx, y2 - ny], [x1 - nx, y1 - ny],
    ]


def mask_intersection_percent(target_polygon: list[list[float]],
                               other_polygon: list[list[float]],
                               resolution: int = 256) -> float:
    """Real pixel-level intersection between two normalized (0-1) polygons.

    Returns Intersection Area / Target Area x 100 (the spec's formula),
    rasterized on a resolution x resolution canvas so arbitrary —
    including concave — mask shapes are handled correctly, not just
    axis-aligned boxes.
    """
    import cv2
    import numpy as np

    def to_px(poly):
        return np.array(
            [[int(round(x * resolution)), int(round(y * resolution))] for x, y in poly],
            dtype=np.int32,
        )

    target_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    other_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(target_mask, [to_px(target_polygon)], 1)
    cv2.fillPoly(other_mask, [to_px(other_polygon)], 1)

    target_area = int(target_mask.sum())
    if target_area == 0:
        return 0.0
    intersection_area = int(np.logical_and(target_mask, other_mask).sum())
    return (intersection_area / target_area) * 100.0


def best_polygon_for_class(detections: list[dict], class_name: str) -> list[list[float]] | None:
    """Highest-confidence detection of this class this frame, as a polygon —
    its real segmentation mask if the model produced one, else its bbox
    corners (so a plain box-only detector still degrades gracefully)."""
    best, best_conf = None, -1.0
    for det in detections:
        if det["class_name"] != class_name:
            continue
        conf = det.get("conf", 0.0)
        if conf > best_conf:
            best_conf = conf
            best = det.get("mask") or bbox_to_polygon(det["xyxy"])
    return best


def evaluate_step(step: dict, detections: list[dict], threshold_pct: float) -> dict:
    """Evaluate one step's region/detection_class target against one
    frame's or image's detections. threshold_pct is on a 0-100 scale.

    Returns {matched, target_type, testable, per_class, note}.
    Only "region" (box) and "detection_class" targets are handled here —
    line regions return testable=False; the caller should keep using the
    separate motion-crossing check for those.
    """
    target_type = step.get("target_type", "region")

    if target_type == "detection_class":
        target_class = step.get("target_class")
        target_polygon = best_polygon_for_class(detections, target_class) if target_class else None
        if target_polygon is None:
            return {
                "matched": False,
                "target_type": target_type,
                "testable": True,
                "per_class": [],
                "note": f'Target class "{target_class}" was not detected in this frame.',
            }
    else:
        region_type = step.get("region_type")
        if region_type == "box":
            x1, y1, x2, y2 = step["region_coords"]
            target_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif region_type == "polygon":
            # A boundary frozen once from a class's own detected mask (see
            # /sequences/freeze-class), so the step never needs that class
            # detected live again — sidesteps a finger fully occluding the
            # key it's pressing, which would otherwise erase its mask.
            target_polygon = step["region_coords"]
        elif region_type == "line" and step.get("trigger_mode") == "overlap":
            # Opt-in: match this line region the same way a box is matched
            # (area overlap %% against the shared threshold) instead of the
            # default motion-crossing check, which needs two frames and so
            # isn't usable here or in the single-image tester.
            x1, y1, x2, y2 = step["region_coords"]
            target_polygon = line_to_polygon(x1, y1, x2, y2)
        else:
            return {
                "matched": False,
                "target_type": target_type,
                "testable": False,
                "per_class": [],
                "note": "Line regions need motion between frames — not evaluated here.",
            }

    needed = step.get("required_classes") or [step["required_class"]]
    per_class = []
    for cls in needed:
        other_polygon = best_polygon_for_class(detections, cls)
        pct = mask_intersection_percent(target_polygon, other_polygon) if other_polygon else 0.0
        per_class.append({
            "class_name": cls,
            "matched": pct >= threshold_pct,
            "percent": round(pct, 1),
        })

    return {
        "matched": bool(needed) and all(c["matched"] for c in per_class),
        "target_type": target_type,
        "testable": True,
        "per_class": per_class,
        "note": None,
    }
