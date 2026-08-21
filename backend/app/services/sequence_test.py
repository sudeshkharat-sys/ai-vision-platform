"""
sequence_test.py
~~~~~~~~~~~~~~~~~
Single-image "does this region+class setup work" check — for iterating on
a sequence's regions before saving or running a full video job. Runs
whichever trained models the project has (detection AND segmentation, if
both exist) once against one image, then reports per-step, per-class
overlap results using the same geometry the real video runtime uses.

Line regions can't be meaningfully tested here — a crossing needs motion
between frames, which a single still image doesn't have.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ..config import settings
from .sequence_geometry import box_from_region_coords, overlap_ratio
from .seg_model import resolve_seg_model_path


def collect_detections(project_id: str, img_bgr: np.ndarray) -> list[dict]:
    """Run every trained model this project has (detector + segmenter) once
    against one image and return a flat list of normalized detections."""
    h, w = img_bgr.shape[:2]
    detections: list[dict] = []

    detector_path = None
    for name in ("main_best.pt", "seed_best.pt"):
        candidate = settings.model_dir.resolve() / project_id / name
        if candidate.exists():
            detector_path = candidate
            break

    seg_path = resolve_seg_model_path(project_id)

    for model_path in (detector_path, seg_path):
        if model_path is None:
            continue
        model = YOLO(str(model_path))
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

    return detections


def evaluate_step(step: dict, detections: list[dict], threshold: float) -> dict:
    """Evaluate one step's region+class(es) against a single image's detections."""
    needed = step.get("required_classes") or [step["required_class"]]

    if step["region_type"] != "box":
        return {
            "order_index": step["order_index"],
            "label": step["label"],
            "region_type": step["region_type"],
            "testable": False,
            "passed": None,
            "per_class": [],
            "note": "Line regions need motion between frames to test — try this one with a real video run instead.",
        }

    region_xyxy = box_from_region_coords(step["region_coords"])
    per_class = []
    for cls in needed:
        best_score = 0.0
        for det in detections:
            if det["class_name"] != cls:
                continue
            score = overlap_ratio(det["xyxy"], region_xyxy)
            if score > best_score:
                best_score = score
        per_class.append({
            "class_name": cls,
            "matched": best_score >= threshold,
            "best_overlap": round(best_score, 3),
        })

    return {
        "order_index": step["order_index"],
        "label": step["label"],
        "region_type": step["region_type"],
        "testable": True,
        "passed": all(c["matched"] for c in per_class),
        "per_class": per_class,
        "note": None,
    }
