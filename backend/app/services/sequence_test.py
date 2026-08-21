"""
sequence_test.py
~~~~~~~~~~~~~~~~~
Single-image "does this region/detection-class + class setup work" check
— for iterating on a sequence's steps before saving or running a full
video job. Runs whichever trained models the project has (detection AND
segmentation, if both exist) once against one image, then reports
per-step results using the same real mask-intersection matcher
(sequence_match.evaluate_step) the video runtime uses, so results here
match what a real run would do.

Line regions can't be meaningfully tested here — a crossing needs motion
between frames, which a single still image doesn't have.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ..config import settings
from .sequence_match import evaluate_step as _evaluate_step
from .seg_model import resolve_seg_model_path


def collect_detections(project_id: str, img_bgr: np.ndarray) -> list[dict]:
    """Run every trained model this project has (detector + segmenter) once
    against one image and return a flat list of normalized detections.
    Segmentation results carry a real "mask" polygon; plain detector
    results carry only "xyxy" (evaluate_step falls back to the bbox as a
    polygon for those)."""
    h, w = img_bgr.shape[:2]
    detections: list[dict] = []

    detector_path = None
    for name in ("main_best.pt", "seed_best.pt"):
        candidate = settings.model_dir.resolve() / project_id / name
        if candidate.exists():
            detector_path = candidate
            break

    seg_path = resolve_seg_model_path(project_id)

    if detector_path is not None:
        model = YOLO(str(detector_path))
        results = model.predict(img_bgr, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": names.get(cls_idx, str(cls_idx)),
                    "xyxy": (x1 / w, y1 / h, x2 / w, y2 / h),
                    "conf": float(box.conf[0]),
                })

    if seg_path is not None:
        model = YOLO(str(seg_path))
        results = model.predict(img_bgr, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            has_masks = r.masks is not None
            mask_polys = r.masks.xyn if has_masks else [None] * len(r.boxes)
            for box, poly in zip(r.boxes, mask_polys):
                cls_idx = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = {
                    "class_name": names.get(cls_idx, str(cls_idx)),
                    "xyxy": (x1 / w, y1 / h, x2 / w, y2 / h),
                    "conf": float(box.conf[0]),
                }
                if poly is not None and len(poly) >= 3:
                    det["mask"] = [[float(px), float(py)] for px, py in poly]
                detections.append(det)

    return detections


def evaluate_step(step: dict, detections: list[dict], threshold: float) -> dict:
    """threshold is a 0-1 fraction (matches the API's overlap_threshold field);
    converted to the 0-100 scale sequence_match.evaluate_step expects."""
    result = _evaluate_step(step, detections, threshold * 100.0)
    return {
        "order_index": step["order_index"],
        "label": step["label"],
        "region_type": step.get("region_type") or "detection_class",
        "testable": result["testable"],
        "passed": result["matched"] if result["testable"] else None,
        "per_class": [
            {"class_name": c["class_name"], "matched": c["matched"], "best_overlap": c["percent"] / 100.0}
            for c in result["per_class"]
        ],
        "note": result["note"],
    }
