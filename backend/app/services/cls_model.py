"""
cls_model.py
~~~~~~~~~~~~~
Crop + classify inference: a YOLO detector finds the region of interest
(e.g. the engine bay), the region is cropped, and a YOLO-cls model
classifies the crop (e.g. full_cover / cut_cover / no_cover).

Artifacts live in <model_dir>/<project_id>/crop_cls/:
    region_best.pt   stage-1 detector (single class, REGION_CLASS)
    cls_best.pt      stage-2 classifier
    cls_meta.json    {"mode", "preprocess", "margin", "classes", ...}

meta["mode"] is "crop" (detector + crop + classifier) or "whole" (the
classifier sees the full frame; no detector) -- the second exists as a
baseline to compare against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import settings

REGION_CLASS = "engine_region"
CLS_FILES = {
    "detector": "region_best.pt",
    "classifier": "cls_best.pt",
    "meta": "cls_meta.json",
}

# (project_id, filename) -> (mtime, YOLO model)
_MODEL_CACHE: dict = {}


def cls_dir(project_id: str) -> Path:
    return settings.model_dir.resolve() / project_id / "crop_cls"


def load_cls_meta(project_id: str) -> dict | None:
    path = cls_dir(project_id) / CLS_FILES["meta"]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def crop_region(img: np.ndarray, bbox_xyxy, margin: float = 0.12) -> np.ndarray | None:
    """Crop pixel box (x1, y1, x2, y2) out of img, grown by `margin` (a
    fraction of the box's own size) on every side so the edges of a cut or
    partial cover stay in view. Returns None for a degenerate box."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
    x2, y2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]


def _load(project_id: str, filename: str):
    path = cls_dir(project_id) / filename
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    key = (project_id, filename)
    cached = _MODEL_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    from ultralytics import YOLO  # lazy -- heavy import
    model = YOLO(str(path))
    _MODEL_CACHE[key] = (mtime, model)
    return model


def _classify(model, crop: np.ndarray) -> tuple[str, float, dict]:
    res = model.predict(crop, verbose=False)[0]
    probs = res.probs
    names = res.names
    scores = {names[i]: round(float(p), 4) for i, p in enumerate(probs.data.tolist())}
    top = int(probs.top1)
    return names[top], float(probs.top1conf), scores


def predict_crop_cls(
    project_id: str,
    img: np.ndarray,
    det_conf: float = 0.25,
    cls_threshold: float = 0.6,
) -> dict:
    """Run the trained pipeline on a BGR image.

    Returns {"status", "state", "confidence", "scores", "bbox", "det_conf"}.
    status: "ok" | "uncertain" (classifier below cls_threshold)
            | "region_not_found" | "no_model".
    bbox is the stage-1 box as normalized [xc, yc, w, h] (crop mode only).
    """
    meta = load_cls_meta(project_id)
    classifier = _load(project_id, CLS_FILES["classifier"])
    if meta is None or classifier is None:
        return {"status": "no_model"}

    # Same preprocessing the models were trained on (see training task).
    if meta.get("preprocess", True):
        from ..tasks.training import clahe_gamma_sharpen
        img = clahe_gamma_sharpen(img)

    h, w = img.shape[:2]
    bbox = None
    region_conf = None
    crop = img

    if meta.get("mode", "crop") == "crop":
        detector = _load(project_id, CLS_FILES["detector"])
        if detector is None:
            return {"status": "no_model"}
        det = detector.predict(img, conf=det_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            return {"status": "region_not_found"}
        best = int(det.boxes.conf.argmax())
        x1, y1, x2, y2 = det.boxes.xyxy[best].tolist()
        region_conf = float(det.boxes.conf[best])
        bbox = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
        crop = crop_region(img, (x1, y1, x2, y2), meta.get("margin", 0.12))
        if crop is None:
            return {"status": "region_not_found"}

    state, conf, scores = _classify(classifier, crop)
    return {
        "status": "ok" if conf >= cls_threshold else "uncertain",
        "state": state,
        "confidence": round(conf, 4),
        "scores": scores,
        "bbox": bbox,
        "det_conf": None if region_conf is None else round(region_conf, 4),
    }
