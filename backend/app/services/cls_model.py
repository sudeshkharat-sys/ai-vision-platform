"""
cls_model.py
~~~~~~~~~~~~~
Crop + classify inference: a YOLO detector finds the region of interest
(e.g. the engine bay), the region is cropped, and a YOLO-cls model
classifies the crop (e.g. full_cover / cut_cover / no_cover).

The region detector is the project's own object-detection model (Main,
else Seed, as recorded in meta["detector"]) -- it is read live from
<model_dir>/<project_id>/{main,seed}_best.pt, so retraining the detector
improves this pipeline without retraining the classifier. meta["crop_class"]
names which detected class is the region to cut out.

Artifacts live in <model_dir>/<project_id>/crop_cls/:
    cls_best.pt      the classifier
    cls_meta.json    {"mode", "detector", "crop_class", "preprocess", "margin", "classes", ...}
    region_best.pt   legacy: a dedicated single-class detector (older runs)

meta["mode"] is "crop" (detector + crop + classifier) or "whole" (the
classifier sees the full frame; no detector).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import settings

CLS_FILES = {
    "detector": "region_best.pt",   # legacy only
    "classifier": "cls_best.pt",
    "meta": "cls_meta.json",
}

# (project_id, filename) -> (mtime, YOLO model)
_MODEL_CACHE: dict = {}


def cls_dir(project_id: str) -> Path:
    return settings.model_dir.resolve() / project_id / "crop_cls"


DETECTOR_FILES = {"main": "main_best.pt", "seed": "seed_best.pt"}


def resolve_detector(project_id: str, preferred: str | None = None):
    """(name, path) of the detection model to find regions with: the
    preferred one if trained, else Main, else Seed. (None, None) if the
    project has neither."""
    base = settings.model_dir.resolve() / project_id
    order = [preferred] if preferred in DETECTOR_FILES else []
    order += [n for n in ("main", "seed") if n not in order]
    for name in order:
        path = base / DETECTOR_FILES[name]
        if path.exists():
            return name, path
    return None, None


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
    return _load_path(cls_dir(project_id) / filename)


def _load_path(path: Path):
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    key = str(path)
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
        if meta.get("detector") in DETECTOR_FILES:
            _, det_path = resolve_detector(project_id, meta["detector"])
            detector = _load_path(det_path) if det_path else None
        else:   # legacy dedicated region detector
            detector = _load(project_id, CLS_FILES["detector"])
        if detector is None:
            return {"status": "no_model"}
        det = detector.predict(img, conf=det_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            return {"status": "region_not_found"}
        confs = det.boxes.conf
        crop_class = meta.get("crop_class")
        if crop_class:
            # only boxes of the chosen class count as the region
            keep = [i for i in range(len(confs)) if det.names[int(det.boxes.cls[i])] == crop_class]
            if not keep:
                return {"status": "region_not_found"}
            best = max(keep, key=lambda i: float(confs[i]))
        else:
            best = int(confs.argmax())
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
