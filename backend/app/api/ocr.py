import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..tasks.ocr_training import train_ocr_model
from ..tasks.tesseract_training import train_tesseract_model, CUSTOM_LANG
from ..tasks.crnn_training import train_crnn_model

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _bbox_to_points(bbox):
    """Four corners (TL, TR, BR, BL) of a normalized [xc, yc, w, h] bbox — the
    starting polygon for a fast, adjustable polyline instead of a plain box."""
    xc, yc, w, h = bbox
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

OCR_FILES = {
    "tflite": "ocr_model.tflite",
    "keras": "ocr_model.keras",
    "labels": "labels.txt",
    "meta": "ocr_meta.json",
}

# Tesseract engine artifacts live in <project>/ocr_tesseract/
TESS_FILES = {
    "traineddata": f"{CUSTOM_LANG}.traineddata",
    "tess_meta": "tess_meta.json",
}

# CRNN line-recognizer artifacts live in <project>/ocr_crnn/
CRNN_FILES = {
    "crnn_tflite": "ocr_crnn.tflite",
    "crnn_charset": "charset.txt",
    "crnn_meta": "crnn_meta.json",
}


class TrainOcrRequest(BaseModel):
    epochs: int = 50
    img_size: int = 64
    target_per_class: int = 300
    val_ratio: float = 0.15
    batch_size: int = 64
    learning_rate: float = 1e-3
    fine_tune: bool = False
    focus_classes: Optional[List[str]] = None
    use_pretrained: bool = True


@router.post("/train/{project_id}")
async def start_ocr_training(
    project_id: str,
    body: TrainOcrRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue the character-classifier training task for a project."""
    await get_owned_project(project_id, current_user, db)
    req = body or TrainOcrRequest()
    task = train_ocr_model.delay(
        project_id, req.epochs, req.img_size, req.target_per_class,
        req.val_ratio, req.batch_size, req.learning_rate,
        req.fine_tune, req.focus_classes, req.use_pretrained,
    )
    return {"task_id": task.id, "status": "queued"}


class TrainTesseractRequest(BaseModel):
    max_iterations: int = 800
    val_ratio: float = 0.2
    learning_rate: float = 1e-4


@router.post("/train-tesseract/{project_id}")
async def start_tesseract_training(
    project_id: str,
    body: TrainTesseractRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Queue Tesseract LSTM fine-tuning: starts from Google's pretrained
    eng.traineddata (tessdata_best) and continues training it on this
    project's labeled character boxes, grouped into text lines.
    """
    await get_owned_project(project_id, current_user, db)
    req = body or TrainTesseractRequest()
    task = train_tesseract_model.delay(
        project_id, req.max_iterations, req.val_ratio, req.learning_rate,
    )
    return {"task_id": task.id, "status": "queued"}


class TrainCrnnRequest(BaseModel):
    epochs: int = 40
    synthetic_lines: int = 3000
    composite_lines: int = 4000
    emnist_lines: int = 3000
    dotpeen_lines: int = 4000
    real_augment_copies: int = 6
    hard_image_ids: Optional[List[str]] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.15


@router.post("/train-crnn/{project_id}")
async def start_crnn_training(
    project_id: str,
    body: TrainCrnnRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Queue CRNN+CTC line-recognizer training — a real OCR that reads a whole
    cropped line at once. Knows the full 0-9/A-Z (from synthetic text) and
    fine-tunes on this project's labeled engraved lines.
    """
    await get_owned_project(project_id, current_user, db)
    req = body or TrainCrnnRequest()
    task = train_crnn_model.delay(
        project_id, epochs=req.epochs, synthetic_lines=req.synthetic_lines,
        composite_lines=req.composite_lines, emnist_lines=req.emnist_lines,
        dotpeen_lines=req.dotpeen_lines, real_augment_copies=req.real_augment_copies,
        hard_image_ids=req.hard_image_ids, batch_size=req.batch_size,
        learning_rate=req.learning_rate, val_ratio=req.val_ratio,
    )
    return {"task_id": task.id, "status": "queued"}


class TrainValueRequest(BaseModel):
    epochs: int = 40
    augment_copies: int = 40
    hard_image_ids: Optional[List[str]] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.2


@router.post("/train-value/{project_id}")
async def start_value_training(
    project_id: str,
    body: TrainValueRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Queue whole-value classifier training — for plates that can only say
    one of a few known values (badge plates: "4"/"6"/"10"/"11"). Instead
    of reading character by character, classifies the whole line crop
    into one of the values labeled in this project. Far more robust than
    OCR when the value set is closed and real photos are scarce.
    """
    await get_owned_project(project_id, current_user, db)
    from ..tasks.value_training import train_value_model
    req = body or TrainValueRequest()
    task = train_value_model.delay(
        project_id, epochs=req.epochs, augment_copies=req.augment_copies,
        hard_image_ids=req.hard_image_ids, batch_size=req.batch_size,
        learning_rate=req.learning_rate, val_ratio=req.val_ratio,
    )
    return {"task_id": task.id, "status": "queued"}


@router.get("/dataset-stats/{project_id}")
async def get_ocr_dataset_stats(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Per-character counts for the OCR dataset (single-character labels only),
    so the UI can show class balance before training.
    """
    await get_owned_project(project_id, current_user, db)

    img_q = await db.execute(
        select(Image.id).where(Image.project_id == project_id, Image.status == "annotated")
    )
    image_ids = [row[0] for row in img_q.fetchall()]
    if not image_ids:
        return {"char_counts": {}, "total_chars": 0, "annotated_images": 0}

    ann_q = await db.execute(
        select(Annotation.class_name, func.count(Annotation.id))
        .where(Annotation.image_id.in_(image_ids))
        .group_by(Annotation.class_name)
    )
    char_counts = {}
    for name, count in ann_q.fetchall():
        label = str(name).strip().upper()
        if len(label) == 1:
            char_counts[label] = char_counts.get(label, 0) + count

    return {
        "char_counts": dict(sorted(char_counts.items())),
        "total_chars": sum(char_counts.values()),
        "annotated_images": len(image_ids),
    }


@router.get("/model-status/{project_id}")
async def get_ocr_model_status(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Which OCR model files exist for this project, with size/mtime."""
    await get_owned_project(project_id, current_user, db)
    out_dir = settings.model_dir / project_id / "ocr"

    files = {}
    for key, name in OCR_FILES.items():
        path = out_dir / name
        if path.exists():
            stat = path.stat()
            files[key] = {
                "exists": True,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        else:
            files[key] = {"exists": False}

    meta = None
    meta_path = out_dir / OCR_FILES["meta"]
    if meta_path.exists():
        import json
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = None

    # Tesseract engine artifacts
    tess_dir = settings.model_dir / project_id / "ocr_tesseract"
    tess_files = {}
    for key, name in TESS_FILES.items():
        path = tess_dir / name
        if path.exists():
            stat = path.stat()
            tess_files[key] = {
                "exists": True,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        else:
            tess_files[key] = {"exists": False}
    tess_meta = None
    tess_meta_path = tess_dir / TESS_FILES["tess_meta"]
    if tess_meta_path.exists():
        try:
            tess_meta = json.loads(tess_meta_path.read_text())
        except Exception:
            tess_meta = None

    # CRNN line-recognizer artifacts
    crnn_dir = settings.model_dir / project_id / "ocr_crnn"
    crnn_files = {}
    for key, name in CRNN_FILES.items():
        path = crnn_dir / name
        if path.exists():
            stat = path.stat()
            crnn_files[key] = {
                "exists": True,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        else:
            crnn_files[key] = {"exists": False}
    crnn_meta = None
    crnn_meta_path = crnn_dir / CRNN_FILES["crnn_meta"]
    if crnn_meta_path.exists():
        try:
            crnn_meta = json.loads(crnn_meta_path.read_text())
        except Exception:
            crnn_meta = None

    return {
        "has_model": files["tflite"]["exists"],
        "files": files,
        "meta": meta,
        "tesseract": {
            "has_model": tess_files["traineddata"]["exists"],
            "files": tess_files,
            "meta": tess_meta,
        },
        "crnn": {
            "has_model": crnn_files["crnn_tflite"]["exists"],
            "files": crnn_files,
            "meta": crnn_meta,
        },
    }


@router.get("/download/{project_id}/{file_type}")
async def download_ocr_file(
    project_id: str,
    file_type: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download a trained OCR artifact:
    tflite | keras | labels | meta (CNN engine)
    traineddata | tess_meta (Tesseract engine)."""
    await get_owned_project(project_id, current_user, db)
    if file_type in TESS_FILES:
        path = settings.model_dir / project_id / "ocr_tesseract" / TESS_FILES[file_type]
        if not path.exists():
            raise HTTPException(status_code=404,
                                detail="File not found — train the Tesseract model first")
        return FileResponse(path=str(path), media_type="application/octet-stream",
                            filename=TESS_FILES[file_type])
    if file_type in CRNN_FILES:
        path = settings.model_dir / project_id / "ocr_crnn" / CRNN_FILES[file_type]
        if not path.exists():
            raise HTTPException(status_code=404,
                                detail="File not found — train the CRNN model first")
        return FileResponse(path=str(path), media_type="application/octet-stream",
                            filename=CRNN_FILES[file_type])
    if file_type not in OCR_FILES:
        raise HTTPException(status_code=400,
                            detail=f"file_type must be one of {list(OCR_FILES) + list(TESS_FILES)}")
    path = settings.model_dir / project_id / "ocr" / OCR_FILES[file_type]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found — train the OCR model first")
    return FileResponse(path=str(path), media_type="application/octet-stream",
                        filename=OCR_FILES[file_type])


# ── Test window: run the trained model on an uploaded photo ───────

# Model cache: project_id -> (mtime, keras_model, classes, img_size)
_MODEL_CACHE: dict = {}

# YOLO character-detector cache: project_id -> (mtime, yolo_model)
_YOLO_CACHE: dict = {}

# YOLO-seg (instance segmentation) cache: project_id -> (mtime, yolo_seg_model)
_YOLO_SEG_CACHE: dict = {}


def _sort_boxes_reading_order(boxes):
    """Sort (x, y, w, h) boxes into rows (top-to-bottom), left-to-right."""
    if not boxes:
        return boxes
    med_h = float(np.median([b[3] for b in boxes]))
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows = []
    for b in boxes:
        cy = b[1] + b[3] / 2
        for row in rows:
            ry = np.mean([x[1] + x[3] / 2 for x in row])
            if abs(cy - ry) < med_h * 0.6:
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: np.mean([b[1] for b in r]))
    return [b for row in rows for b in row]


PLATE_CLASS_NAME = "PLATE"  # region box that marks the whole plate/text area


def _yolo_predict_raw(project_id: str, img: np.ndarray, conf: float):
    """
    Load (and cache) the project's trained YOLO model and run inference.
    Returns the raw ultralytics results, or None if the project has no
    trained detection model. Shared by _yolo_char_boxes and
    _yolo_plate_region so both read from the one model + one forward
    pass' worth of caching logic.

    Prefers main_best.pt over seed_best.pt (see services/det_model.py) so
    auto-annotate keeps improving as the project moves past its seed batch,
    the same main->seed preference the rest of the pipeline uses.
    """
    from ..services.det_model import resolve_det_model_path, det_model_uses_preprocess

    model_path = resolve_det_model_path(project_id)
    if model_path is None:
        return None

    mtime = model_path.stat().st_mtime
    cache_key = (project_id, model_path.name)
    cached = _YOLO_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        model = cached[1]
    else:
        from ultralytics import YOLO  # lazy — heavy import
        model = YOLO(str(model_path))
        _YOLO_CACHE[cache_key] = (mtime, model)

    # Training can be run with preprocess on OR off (CLAHE+gamma+unsharp
    # baked into every training image, or not) -- feeding this model
    # whichever one it DIDN'T train on is a train/inference mismatch either
    # way. Each trainer writes a sidecar meta next to its weights recording
    # what it actually used; det_model_uses_preprocess reads the one
    # belonging to the weights file we resolved above.
    from ..tasks.training import clahe_gamma_sharpen
    if det_model_uses_preprocess(model_path):
        img = clahe_gamma_sharpen(img)

    return model.predict(img, conf=conf, verbose=False)


def _yolo_char_boxes_labeled(project_id: str, img: np.ndarray, conf: float = 0.25):
    """
    Detect character boxes with the project's trained YOLO model.
    Returns (x, y, w, h, label, confidence) tuples in reading order, or
    None when the project has no trained YOLO model — callers fall back
    to classical segmentation.

    The detector is trained per-character-class, so every detection
    already carries the character's identity. Keeping that label here is
    what lets auto-annotate write finished box+label annotations straight
    from the detector, with no second classifier pass to re-derive a
    class the detector had already predicted.

    A LEARNED detector is the fix for reflective/engraved surfaces:
    thresholding sees reflections as characters and misses real ones,
    while YOLO trained on the same labeled boxes knows what an actual
    character looks like.
    """
    results = _yolo_predict_raw(project_id, img, conf)
    if results is None:
        return None

    boxes = []
    for r in results:
        names = r.names
        for b in r.boxes:
            label = str(names.get(int(b.cls[0]), "")).strip().upper()
            if len(label) != 1:
                continue  # only single-character classes belong to OCR
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
            boxes.append((int(x1), int(y1), int(round(w)), int(round(h)),
                          label, float(b.conf[0])))
    if not boxes:
        return None
    boxes = _dedupe_cross_class_boxes(boxes)
    # _sort_boxes_reading_order only reads indices 0-3, so the label and
    # confidence ride along untouched.
    return _sort_boxes_reading_order(boxes)


def _dedupe_cross_class_boxes(boxes, iou_thresh: float = 0.4):
    """Drop the weaker box whenever two detections of DIFFERENT classes
    heavily overlap — a real, single stroke that the detector is unsure
    about (a "6" whose curve also looks a little like a "1" or an "8")
    can fire two boxes on the exact same character, one per candidate
    class. YOLO's own NMS never catches this: it only suppresses
    duplicates WITHIN one class, since here each character is trained as
    its own separate class. Left unfiltered, one physical glyph gets
    reported as two boxes/two characters downstream -- both in the
    line's decoded text and in the crop width used by the value
    classifier."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    kept = []
    for b in sorted(boxes, key=lambda b: -b[5]):  # most confident first
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _yolo_char_boxes(project_id: str, img: np.ndarray, conf: float = 0.25):
    """Geometry-only view of :func:`_yolo_char_boxes_labeled` — (x, y, w, h)
    pixel boxes in reading order, for callers that classify the crops
    themselves rather than using the detector's own labels."""
    labeled = _yolo_char_boxes_labeled(project_id, img, conf)
    if labeled is None:
        return None
    return [(x, y, w, h) for x, y, w, h, _label, _conf in labeled]


def _yolo_plate_region(project_id: str, img: np.ndarray, conf: float = 0.25):
    """
    Best "plate" region box from the project's trained YOLO model, as
    (x, y, w, h) pixels, or None if no seed model / no "plate" class /
    nothing detected. A project only has this when a "plate" label was
    added and boxed around the whole plate on training photos.

    This gives a robust text-region crop even on a photo where individual
    character boxes come back weak or empty — much better than reading the
    entire raw frame (background, glare, plate edge included).
    """
    results = _yolo_predict_raw(project_id, img, conf)
    if results is None:
        return None

    best = None  # (confidence, x, y, w, h)
    for r in results:
        names = r.names
        for b in r.boxes:
            label = str(names.get(int(b.cls[0]), "")).strip().upper()
            if label != PLATE_CLASS_NAME:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 4 or h < 4:
                continue
            score = float(b.conf[0])
            if best is None or score > best[0]:
                best = (score, int(x1), int(y1), int(round(w)), int(round(h)))
    return best[1:] if best else None


def _yolo_seg_predict_raw(project_id: str, img: np.ndarray, conf: float):
    """Load (and cache) the project's trained YOLO-seg model and run inference.
    Prefers seg_main_best.pt, falls back to seg_seed_best.pt, then the
    legacy seg_best.pt name (pre seed/main split). Returns raw ultralytics
    results, or None if no segmentation model has been trained yet."""
    from ..services.seg_model import resolve_seg_model_path
    model_path = resolve_seg_model_path(project_id)
    if model_path is None:
        return None

    mtime = model_path.stat().st_mtime
    cached = _YOLO_SEG_CACHE.get(project_id)
    if cached and cached[0] == mtime:
        model = cached[1]
    else:
        from ultralytics import YOLO  # lazy — heavy import
        model = YOLO(str(model_path))
        _YOLO_SEG_CACHE[project_id] = (mtime, model)

    return model.predict(img, conf=conf, verbose=False)


def _seg_char_instances(project_id: str, img: np.ndarray, conf: float = 0.25):
    """
    Run the trained segment model and return one entry per detected
    character instance: its box, its own dotted-mask (full-image size,
    rasterized from the predicted polygon), label and confidence — in
    reading order. Each instance mask is the raw cluster of engraving
    dots for ONE character; nothing here connects them yet, that's
    `_reconnect_dots`.

    Returns None when the project has no trained seg model or nothing
    single-character was found, so callers can fall back / error clearly.
    """
    results = _yolo_seg_predict_raw(project_id, img, conf)
    if results is None:
        return None

    H, W = img.shape[:2]
    instances = []
    for r in results:
        if r.masks is None or r.boxes is None:
            continue
        names = r.names
        for poly, b in zip(r.masks.xy, r.boxes):
            label = str(names.get(int(b.cls[0]), "")).strip().upper()
            if len(label) != 1 or poly is None or len(poly) < 3:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            instances.append({
                "box": (int(x1), int(y1), int(round(w)), int(round(h))),
                "mask": mask,
                "label": label,
                "conf": float(b.conf[0]),
            })
    if not instances:
        return None

    order = _sort_boxes_reading_order([inst["box"] for inst in instances])
    by_box = {inst["box"]: inst for inst in instances}
    return [by_box[b] for b in order]


def _reconnect_dots(mask: np.ndarray, box):
    """
    Bridge the gaps in an engraved dot-peen character so it reads as one
    connected glyph instead of loose dots.

    A dotted "0" traced by a tilted line of dots is exactly what makes it
    ambiguous (looks like a 9/8/whatever depending on which dots the
    thresholding pass happens to keep) — the fix is to close the gaps
    BETWEEN the dots with a kernel sized to the dot spacing itself, so it
    works regardless of the stroke's angle (an isotropic elliptical
    kernel, not a horizontal one — a horizontal-only kernel only bridges
    dots that happen to sit on a horizontal run).

    Returns (closed_mask, (x0, y0)) — the reconstructed binary glyph
    cropped to the instance's box (plus a small margin) and the crop's
    top-left offset in full-image coordinates.
    """
    x, y, w, h = box
    pad = int(max(w, h) * 0.18) + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1e, y1e = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
    roi = mask[y0:y1e, x0:x1e]
    if roi.size == 0 or cv2.countNonZero(roi) == 0:
        return roi, (x0, y0)

    # Estimate the dot pitch from the (still disconnected) blobs: kernel
    # radius scales with typical dot size so dense/small dots get a small
    # bridge and coarse/large dots get a bigger one.
    n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    comp_areas = [stats[i][4] for i in range(1, n) if stats[i][4] >= 2]
    dot_radius = math.sqrt(float(np.median(comp_areas)) / math.pi) if comp_areas else 2.0
    k = max(3, (int(round(dot_radius * 2.8)) | 1))  # odd, isotropic
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Trim the fuzz the closing adds to the outer silhouette without
    # reopening the gaps it just bridged.
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return closed, (x0, y0)


def _glyph_crop_from_mask(gray: np.ndarray, closed_mask: np.ndarray, offset, size: int):
    """
    Turn a reconnected binary glyph into a (size, size) grayscale crop
    matching the style `_extract_char_crop` produces from real photos, so
    it can be fed straight into the trained CNN classifier.

    Ink/background brightness are sampled from the ORIGINAL photo (not
    invented), so the synthesized stroke keeps this image's real contrast
    instead of turning into a flat cartoon glyph the classifier has never
    seen the likes of.
    """
    if closed_mask.size == 0 or closed_mask.shape[0] < 2 or closed_mask.shape[1] < 2:
        return None
    x0, y0 = offset
    h, w = closed_mask.shape
    region = gray[y0:y0 + h, x0:x0 + w]
    if region.shape != closed_mask.shape:
        return None

    ink_px = region[closed_mask > 0]
    bg_px = region[closed_mask == 0]
    ink_val = float(np.percentile(ink_px, 15)) if ink_px.size else 40.0
    bg_val = float(np.median(bg_px)) if bg_px.size else 200.0

    glyph = np.where(closed_mask > 0, ink_val, bg_val).astype(np.uint8)

    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size), int(round(bg_val)), dtype=np.uint8)
    y1, x1 = (size - nh) // 2, (size - nw) // 2
    canvas[y1:y1 + nh, x1:x1 + nw] = resized
    return canvas


def _predict_with_seg(project_id: str, img: np.ndarray, gray: np.ndarray):
    """
    Plate -> per-character dotted-mask region -> reconnect the dots into a
    solid glyph -> read with the trained CNN classifier. This is the
    dedicated path for engraved dot-peen plates: bbox/classical
    segmentation alone can't tell which dots belong to which character or
    bridge a tilted stroke, the segment model's polygon masks give an
    exact per-character dot cluster to reconnect.
    """
    model, classes, img_size = _load_project_model(project_id)

    instances = _seg_char_instances(project_id, img)
    if instances is None:
        raise HTTPException(
            status_code=422,
            detail="No segment-model detections found. Train the segment model "
                   "and draw polygon masks around the dotted characters first.",
        )

    crops, kept = [], []
    for inst in instances:
        closed, offset = _reconnect_dots(inst["mask"], inst["box"])
        crop = _glyph_crop_from_mask(gray, closed, offset, img_size)
        if crop is not None:
            crops.append(crop)
            kept.append(inst["box"])
    if not crops:
        raise HTTPException(status_code=422, detail="Could not reconstruct any characters")

    batch = np.stack(crops).astype(np.float32)[..., None] / 255.0
    probs = model.predict(batch, verbose=0)

    results, text = [], ""
    for (x, y, w, h), p in zip(kept, probs):
        idx = int(p.argmax())
        ch, conf = classes[idx], float(p[idx])
        text += ch
        results.append({"char": ch, "confidence": round(conf, 3), "box": [x, y, w, h]})
        color = (74, 222, 128) if conf >= 0.8 else (21, 170, 250) if conf >= 0.5 else (113, 113, 248)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, ch, (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    return {"text": text, "characters": results, "preview": preview,
            "num_found": len(results), "engine": "seg", "detector": "yolo-seg"}


def _load_project_model(project_id: str):
    """Load (and cache) the trained Keras model + labels for a project."""
    out_dir = settings.model_dir / project_id / "ocr"
    keras_path = out_dir / "ocr_model.keras"
    if not keras_path.exists():
        raise HTTPException(status_code=404, detail="No trained OCR model — train one first.")

    mtime = keras_path.stat().st_mtime
    cached = _MODEL_CACHE.get(project_id)
    if cached and cached[0] == mtime:
        return cached[1], cached[2], cached[3]

    import tensorflow as tf  # lazy — only loaded when the test window is used
    model = tf.keras.models.load_model(str(keras_path))
    classes = (out_dir / "labels.txt").read_text().split()
    img_size = model.input_shape[1]
    _MODEL_CACHE[project_id] = (mtime, model, classes, img_size)
    return model, classes, img_size


def _find_text_region(gray: np.ndarray):
    """
    Locate the engraved-text band inside a full photo so segmentation
    doesn't have to fight the whole scene (chassis, background, tools…).
    Morphological gradient -> threshold -> horizontal closing merges the
    characters of a line into wide blobs; the union of line-shaped blobs
    (plus margin) is the region of interest.

    Returns (x, y, w, h) in pixels — the full image when nothing
    text-like is found or the text already fills the frame.
    """
    H, W = gray.shape
    full = (0, 0, W, H)

    # Percentile-thresholded gradient: engraved strokes have the strongest
    # edges; Otsu drowns in metal-grain noise so a fixed top-percentile cut
    # is far more stable on textured surfaces.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    thr = max(float(np.percentile(grad, 93)), 20.0)
    bw = (grad > thr).astype(np.uint8) * 255

    # Connect neighbouring characters into horizontal line blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, W // 40), 3))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < W * 0.10 or h < H * 0.015 or h > H * 0.5:
            continue
        if w / h < 1.8:          # text lines are much wider than tall
            continue
        roi = bw[y:y + h, x:x + w]
        density = cv2.countNonZero(roi) / float(w * h)
        if density < 0.12:       # mostly empty box — not dense stroke texture
            continue
        # A text line is many separate strokes; a scratch/edge is one blob
        ncomp, _, cstats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        strokes = sum(1 for i in range(1, ncomp) if cstats[i][4] >= 15)
        if strokes < 3:
            continue
        lines.append((x, y, w, h, w * h * density))

    if not lines:
        return full

    # Keep the strongest line plus others with comparable height that sit
    # vertically nearby (multi-row plates) — rejects far-away texture
    lines.sort(key=lambda t: t[4], reverse=True)
    bx, by, bwd, best_h, _ = lines[0]
    kept = [
        l for l in lines
        if 0.4 * best_h <= l[3] <= 2.5 * best_h
        and abs((l[1] + l[3] / 2) - (by + best_h / 2)) <= 4 * best_h
    ]

    x0 = min(l[0] for l in kept)
    y0 = min(l[1] for l in kept)
    x1 = max(l[0] + l[2] for l in kept)
    y1 = max(l[1] + l[3] for l in kept)

    # Margin so character strokes at the edge aren't clipped
    mx, my = int((x1 - x0) * 0.06) + 4, int((y1 - y0) * 0.30) + 4
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(W, x1 + mx), min(H, y1 + my)

    # Region as big as the frame anyway -> just use the full image
    if (x1 - x0) * (y1 - y0) > 0.85 * W * H:
        return full
    return (x0, y0, x1 - x0, y1 - y0)


def _segment_characters(gray: np.ndarray):
    """
    Find character boxes in a plate photo using classical image
    processing (the same approach the mobile app will use):
    contrast-enhance, threshold both polarities, take connected
    components shaped like characters, group into rows, sort
    left-to-right.  Returns a list of (x, y, w, h) pixel boxes.
    """
    H, W = gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    enh = cv2.GaussianBlur(enh, (3, 3), 0)

    def boxes_for(thresh_img):
        n, _, stats, _ = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 30:                    # specks
                continue
            if h < H * 0.03 or h > H * 0.95: # too small / whole image
                continue
            if w > W * 0.5:                  # merged blob / border
                continue
            ar = w / h
            # Upper bound is loose on purpose: touching characters merge into
            # one wide blob (ar up to ~3.5) and are split apart further down.
            if ar < 0.04 or ar > 3.5:
                continue
            out.append((int(x), int(y), int(w), int(h)))
        return out

    # Characters can be darker or lighter than the metal, and global
    # thresholding misses low-contrast strokes — collect candidates from
    # BOTH Otsu polarities AND adaptive thresholding, then merge. This
    # rescues thin chars ('1') and chars visible in only one view ('J').
    _, otsu_inv = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_pos = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = max(15, (min(H, W) // 12) | 1)  # odd block size scaled to image
    adap_inv = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, 7)
    adap_pos = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, 7)

    def overlap(a, b):
        """Intersection over the SMALLER box — catches contained fragments."""
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        smaller = min(a[2] * a[3], b[2] * b[3])
        return (ix * iy) / smaller if smaller else 0.0

    # Primary source: the better Otsu polarity (clean global threshold).
    # The other views may only ADD boxes that (a) don't overlap something
    # already found and (b) match the primary's character height — this
    # rescues missed thin/low-contrast chars without importing noise.
    cand_inv, cand_pos = boxes_for(otsu_inv), boxes_for(otsu_pos)
    primary = cand_inv if len(cand_inv) >= len(cand_pos) else cand_pos
    secondary = cand_pos if primary is cand_inv else cand_inv
    boxes = list(primary)
    if boxes:
        ref_h = float(np.median([b[3] for b in boxes]))
        for extra in (secondary, boxes_for(adap_inv), boxes_for(adap_pos)):
            for b in extra:
                if not (0.7 * ref_h <= b[3] <= 1.4 * ref_h):
                    continue
                if all(overlap(b, kept) < 0.5 for kept in boxes):
                    boxes.append(b)
    else:
        # Otsu found nothing at all — fall back to adaptive views
        for extra in (boxes_for(adap_inv), boxes_for(adap_pos)):
            for b in extra:
                if all(overlap(b, kept) < 0.5 for kept in boxes):
                    boxes.append(b)
    if not boxes:
        return []

    # Keep boxes whose height is close to the median character height
    med_h = float(np.median([b[3] for b in boxes]))
    boxes = [b for b in boxes if 0.45 * med_h <= b[3] <= 1.8 * med_h]

    # Group into text rows by vertical overlap with the row's running band
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    rows = []
    for b in boxes:
        cy = b[1] + b[3] / 2
        placed = False
        for row in rows:
            ry = np.mean([x[1] + x[3] / 2 for x in row])
            if abs(cy - ry) < med_h * 0.6:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    # A real character sits in a row with neighbours; once a solid row
    # (3+ chars) exists, lone stray blobs elsewhere are almost always
    # scratches, screws or texture — drop single-box rows.
    if len(rows) > 1 and max(len(r) for r in rows) >= 3:
        rows = [r for r in rows if len(r) >= 2]

    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: np.mean([b[1] for b in r]))

    # ── Split merged blobs: a box much wider than the row's typical
    # character is almost always 2+ touching letters. Cut it at the
    # thinnest ink columns (the gaps between the characters). ──
    ink = otsu_inv if primary is cand_inv else otsu_pos
    ink = cv2.bitwise_or(ink, adap_inv if primary is cand_inv else adap_pos)

    def split_wide(box, med_w):
        x, y, w, h = box
        k = int(round(w / max(med_w, 1.0)))
        if k < 2 or w < 2 * 4:
            return [box]
        k = min(k, 8)
        col = ink[y:y + h, x:x + w].sum(axis=0).astype(np.float32)
        parts, prev = [], 0
        for i in range(1, k):
            target = int(w * i / k)
            lo = max(prev + 3, target - int(w * 0.12) - 1)
            hi = min(w - 3, target + int(w * 0.12) + 1)
            cut = target if lo >= hi else lo + int(np.argmin(col[lo:hi]))
            parts.append((x + prev, y, cut - prev, h))
            prev = cut
        parts.append((x + prev, y, w - prev, h))
        return [p for p in parts if p[2] >= 3]

    med_w_all = float(np.median([b[2] for row in rows for b in row]))
    out = []
    for row in rows:
        # Prefer the row's own typical width when the row is big enough
        widths = sorted(b[2] for b in row)
        med_w = float(widths[len(widths) // 2]) if len(row) >= 4 else med_w_all
        for b in row:
            if b[2] > 1.6 * med_w:
                out.extend(split_wide(b, med_w))
            else:
                out.append(b)
    return out


def _segment_in_region(gray: np.ndarray):
    """
    Full-photo pipeline: find the text band first, then segment characters
    inside it and shift the boxes back to whole-image coordinates. Lets the
    test window / auto-label work on uncropped photos.
    """
    rx, ry, rw, rh = _find_text_region(gray)
    roi = gray[ry:ry + rh, rx:rx + rw]
    boxes = _segment_characters(roi)
    if not boxes and (rw, rh) != (gray.shape[1], gray.shape[0]):
        # Region guess was wrong — retry on the full frame
        return _segment_characters(gray), (0, 0, gray.shape[1], gray.shape[0])
    return [(x + rx, y + ry, w, h) for (x, y, w, h) in boxes], (rx, ry, rw, rh)


def _segment_with_plate_hint(project_id: str, img: np.ndarray, gray: np.ndarray):
    """
    Classical character segmentation for when YOLO found no character
    boxes. Prefers the trained "plate" region (a real detection) as the
    search area over _find_text_region's morphology-based guess; falls
    back to that guess when there's no plate class trained.
    Returns (boxes, region) exactly like _segment_in_region.
    """
    plate = _yolo_plate_region(project_id, img)
    if plate:
        px, py, pw, ph = plate
        roi = gray[py:py + ph, px:px + pw]
        boxes = _segment_characters(roi)
        if boxes:
            return [(x + px, y + py, w, h) for (x, y, w, h) in boxes], plate
    return _segment_in_region(gray)


def _predict_with_tesseract(project_id: str, img: np.ndarray, gray: np.ndarray):
    """
    Test-window prediction with the fine-tuned Tesseract model: find the
    text band, group the segmented character boxes into rows, crop each
    row and read it with `tesseract --psm 7` using the project's
    <CUSTOM_LANG>.traineddata. Lines are normalized exactly like the
    training lines so inference matches training.
    """
    import subprocess
    import tempfile
    from ..tasks.tesseract_training import _crop_line

    tess_dir = settings.model_dir / project_id / "ocr_tesseract"
    if not (tess_dir / TESS_FILES["traineddata"]).exists():
        raise HTTPException(status_code=404,
                            detail="No fine-tuned Tesseract model — train one first.")

    boxes = _yolo_char_boxes(project_id, img)
    region = (0, 0, gray.shape[1], gray.shape[0])
    if boxes is None:
        boxes, region = _segment_with_plate_hint(project_id, img, gray)
    if not boxes:
        raise HTTPException(status_code=422,
                            detail="No characters found. Try a tighter crop of the plate area.")

    # Group segmented boxes into rows (same banding rule as segmentation)
    med_h = float(np.median([b[3] for b in boxes]))
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows = []
    for b in boxes:
        cy = b[1] + b[3] / 2
        for row in rows:
            ry = np.mean([x[1] + x[3] / 2 for x in row])
            if abs(cy - ry) < med_h * 0.6:
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: np.mean([b[1] for b in r]))

    results, text = [], ""
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(rows):
            # _crop_line takes (label, x1, y1, x2, y2) tuples
            line = [("?", x, y, x + w, y + h) for (x, y, w, h) in row]
            crop = _crop_line(img, line)
            if crop is None:
                continue
            png = Path(tmp) / f"row_{i}.png"
            cv2.imwrite(str(png), crop)
            r = subprocess.run([
                "tesseract", str(png), "stdout",
                "--tessdata-dir", str(tess_dir), "-l", CUSTOM_LANG,
                "--psm", "7",
                "-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            ], capture_output=True, text=True)
            import re as _re
            line_text = _re.sub(r"\s+", "", r.stdout.upper()) if r.returncode == 0 else ""
            text += line_text
            x1 = min(b[0] for b in row)
            y1 = min(b[1] for b in row)
            x2 = max(b[0] + b[2] for b in row)
            y2 = max(b[1] + b[3] for b in row)
            results.append({"char": line_text, "confidence": None,
                            "box": [x1, y1, x2 - x1, y2 - y1]})
            cv2.rectangle(img, (x1, y1), (x2, y2), (74, 222, 128), 2)
            cv2.putText(img, line_text, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74, 222, 128), 2, cv2.LINE_AA)

    rx, ry, rw, rh = region
    H, W = gray.shape
    if (rw, rh) != (W, H):
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), 1)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    return {"text": text, "characters": results, "preview": preview,
            "num_found": len(text), "engine": "tesseract"}


# CRNN model cache: project_id -> (mtime, keras_model, charset)
_CRNN_CACHE: dict = {}


def _load_crnn(project_id: str):
    crnn_dir = settings.model_dir / project_id / "ocr_crnn"
    keras_path = crnn_dir / "ocr_crnn.keras"
    if not keras_path.exists():
        raise HTTPException(status_code=404, detail="No trained CRNN model — train one first.")
    mtime = keras_path.stat().st_mtime
    cached = _CRNN_CACHE.get(project_id)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    import tensorflow as tf
    model = tf.keras.models.load_model(str(keras_path))
    charset = (crnn_dir / "charset.txt").read_text().split()
    _CRNN_CACHE[project_id] = (mtime, model, charset)
    return model, charset


def _project_allowed_chars(project_id: str) -> set | None:
    """Distinct single-character class names actually annotated anywhere
    in this project (e.g. {"0","1","4","6"}), uppercased — or None if the
    project has no single-character annotations at all (nothing to
    restrict to, so decoding falls back to the model's full charset
    unchanged, same as before this existed).

    The CRNN's charset is a fixed 0-9+A-Z (see crnn_training.py) no
    matter what a project trained on — a project that only ever drew
    "0"/"1"/"4"/"6" boxes still has a live, mostly-synthetic-trained path
    to every letter A-Z, so an undertrained model can (and does) emit
    letters like "Z" that were never part of the project's own data.
    Restricting the decode to only the classes this project actually
    uses closes that off."""
    from ..connectors.statedb_connector import StateDBConnector

    db = StateDBConnector()
    with db.get_session() as conn:
        rows = db.execute_query(
            conn,
            "SELECT DISTINCT a.class_name FROM annotations a "
            "JOIN images i ON i.id = a.image_id "
            "WHERE i.project_id = :pid AND char_length(a.class_name) = 1",
            {"pid": project_id},
        )
    chars = {r["class_name"].upper() for r in rows if r.get("class_name")}
    return chars or None


# Canonical definition lives in tasks/crnn_training.py so the CRNN's TRAINING
# crop and this inference crop de-skew by exactly the same rule — they drifted
# apart once already, and the model paid for it.
from ..tasks.crnn_training import _estimate_text_angle  # noqa: E402


def _greedy_decode_steps(probs, blank_index, charset, allowed_indices=None):
    """CTC greedy decode keeping, for every emitted character, the class
    index and the timestep that emitted it — so a format constraint can go
    back to that timestep's full softmax and re-pick among allowed classes.

    allowed_indices, when given, restricts every timestep's argmax to that
    set (plus blank_index, always kept so CTC's repeat/blank collapsing
    still works) — the model's charset is a fixed 0-9+A-Z regardless of
    what a project actually trained on, so a project with only digit
    classes could still emit any letter it was ever even weakly confused
    about. Masking here stops that at the source instead of only patching
    it up after the fact once a pattern-length guess happens to match."""
    if allowed_indices is not None:
        mask = np.full(probs.shape[1], -np.inf, dtype=np.float32)
        mask[list(allowed_indices)] = 0.0
        probs = probs + mask  # local copy — never mutates the caller's array
    best = probs.argmax(axis=1)
    out, prev = [], -1
    for t, idx in enumerate(best):
        idx = int(idx)
        if idx != prev and idx != blank_index:
            out.append((charset[idx], idx, t))
        prev = idx
    return out


def _ctc_log_prob(probs, seq, blank):
    """Exact CTC log-probability of the label sequence `seq` (class
    indices) under the (T, C) softmax matrix `probs` — the standard CTC
    forward algorithm in log space.

    This is what makes a closed-vocabulary read robust: instead of
    trusting the greedy per-timestep argmax (which invents duplicates
    like "4"->"44", drops faint characters like "66"->"6", and swaps
    one-notch lookalikes like "4"->"0" whenever a single frame flickers),
    every candidate value is scored against the WHOLE probability matrix,
    and impossible outputs are never candidates at all."""
    lp = np.log(np.maximum(probs, 1e-12))
    ext = [blank]
    for s in seq:
        ext += [s, blank]
    S, T = len(ext), lp.shape[0]
    NEG = -1e30
    a = np.full(S, NEG)
    a[0] = lp[0, ext[0]]
    if S > 1:
        a[1] = lp[0, ext[1]]
    for t in range(1, T):
        na = np.full(S, NEG)
        for s in range(S):
            v = a[s]
            if s > 0:
                v = np.logaddexp(v, a[s - 1])
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                v = np.logaddexp(v, a[s - 2])
            na[s] = v + lp[t, ext[s]]
        a = na
    out = a[S - 1]
    if S > 1:
        out = np.logaddexp(out, a[S - 2])
    return float(out)


def _best_lexicon_value(probs, values, charset, blank):
    """Pick the candidate string with the highest exact CTC probability.
    Returns (value, per-char-normalized confidence, scored list) or None
    when no candidate is encodable in the charset."""
    idx = {c: i for i, c in enumerate(charset)}
    scored = []
    for v in values:
        if not v or any(c not in idx for c in v):
            continue
        seq = [idx[c] for c in v]
        # CTC needs a frame per character plus one per adjacent repeat
        need = len(seq) + sum(1 for a, b in zip(seq, seq[1:]) if a == b)
        if need > probs.shape[0]:
            continue
        scored.append((v, _ctc_log_prob(probs, seq, blank)))
    if not scored:
        return None
    scored.sort(key=lambda kv: -kv[1])
    best_v, best_lp = scored[0]
    conf = float(np.exp(best_lp / max(1, len(best_v))))
    return best_v, round(conf, 4), [
        {"value": v, "log_prob": round(s, 2)} for v, s in scored[:5]]


def _pattern_allowed(spec_ch, charset):
    """Character-class for one pattern position: L=letter, D=digit, ?=any,
    anything else is a literal."""
    if spec_ch == "L":
        return [i for i, c in enumerate(charset) if c.isalpha()]
    if spec_ch == "D":
        return [i for i, c in enumerate(charset) if c.isdigit()]
    if spec_ch == "?":
        return list(range(len(charset)))
    return [i for i, c in enumerate(charset) if c == spec_ch.upper()]


def _apply_pattern(emits, probs, pattern, charset):
    """Re-pick each emitted character among the classes its serial-format
    position allows (argmax of that timestep's softmax restricted to the
    allowed set). Kills one-dot ambiguities like V/U or 0/8 whenever the
    schema says the position is a letter or a digit. Returns None when the
    read's length doesn't match the pattern — caller reports the mismatch
    instead of guessing."""
    if len(emits) != len(pattern):
        return None
    fixed = []
    for (ch, idx, t), spec in zip(emits, pattern):
        allowed = _pattern_allowed(spec, charset)
        if not allowed:
            return None
        if idx in allowed:
            fixed.append(ch)
        else:
            fixed.append(charset[max(allowed, key=lambda a: probs[t, a])])
    return "".join(fixed)


def _collapse_near_duplicates(emits, probs):
    """Merge a pair of adjacent emitted characters that fire abnormally
    close together in time into just the more confident one.

    CTC greedy decoding treats any two DIFFERENT non-blank classes as two
    separate characters, even if they came from timesteps a hair apart --
    but a real character normally occupies several consecutive timesteps
    on its own. When the softmax briefly flickers to a different class
    mid-glyph (common on a character the network is genuinely unsure
    about, like an "8" that could almost be read as "3" then "B"), that
    flicker gets emitted as a phantom extra character right next to the
    real one. Confirmed against real data: "TZTZG25852" -> "TZTZG253B5Z"
    (the "8" split into "3B") and similar mid-string splits. Only merges
    pairs that are close relative to this line's OWN typical spacing, so
    a genuinely tight but real two-character run isn't touched."""
    if len(emits) < 3:
        return emits
    gaps = [emits[i + 1][2] - emits[i][2] for i in range(len(emits) - 1)]
    typical = float(np.median(gaps))
    if typical <= 0:
        return emits
    out = list(emits)
    i = 0
    while i < len(out) - 1:
        gap = out[i + 1][2] - out[i][2]
        if gap < max(1.0, typical * 0.4):
            a_conf = float(probs[out[i][2], out[i][1]])
            b_conf = float(probs[out[i + 1][2], out[i + 1][1]])
            keep = out[i] if a_conf >= b_conf else out[i + 1]
            out[i:i + 2] = [keep]
        else:
            i += 1
    return out


def _trim_weak_edges(emits, probs, min_keep=2):
    """Drop leading/trailing emitted characters whose confidence is far
    below the line's own median -- real characters in a clean read score
    similarly to each other, while a character conjured from background
    texture, a scratch, or a pen mark caught in a widened crop typically
    scores much lower. Confirmed against real data: the dominant failure
    mode was the correct serial with 1-4 such characters glued onto one or
    both ends (e.g. "YYT4G55160" -> "5YYT4G55160"). Never trims below
    min_keep characters, so a short genuine read can't be trimmed away.
    min_keep is 2, not 3: badge-style plates genuinely read 1-2 characters
    ("11", "4"), and at 3 a phantom glued onto "11" ("110") sat exactly at
    the floor where trimming was forbidden to remove it."""
    if len(emits) <= min_keep:
        return emits
    confs = [float(probs[t, i]) for (_, i, t) in emits]
    median = float(np.median(confs))
    if median <= 0:
        return emits
    threshold = median * 0.55
    start, end = 0, len(emits)
    while end - start > min_keep and confs[start] < threshold:
        start += 1
    while end - start > min_keep and confs[end - 1] < threshold:
        end -= 1
    return emits[start:end]


def _decode_line_both_ways(model, crop, normalize, blank_index, charset, allowed_indices=None):
    """Read a line crop at 0° and 180° and keep the more confident read —
    a serial detected as 'vertical' is upright after one 90° rotation and
    upside-down after the other, and box geometry can't tell those apart."""
    best = None
    for flipped in (False, True):
        c = cv2.rotate(crop, cv2.ROTATE_180) if flipped else crop
        norm = normalize(c).astype(np.float32) / 255.0
        probs = model.predict(norm[None, ..., None], verbose=0)[0]
        emits = _greedy_decode_steps(probs, blank_index, charset, allowed_indices)
        confs = [float(probs[t, i]) for (_, i, t) in emits]
        conf = sum(confs) / len(confs) if confs else 0.0
        # prefer the unflipped read unless the flip is clearly better
        score = conf if not flipped else conf - 0.05
        if best is None or (emits and score > best[0]):
            best = (score, emits, probs, conf)
    _, emits, probs, _ = best
    emits = _collapse_near_duplicates(emits, probs)
    emits = _trim_weak_edges(emits, probs)
    confs = [float(probs[t, i]) for (_, i, t) in emits]
    conf = sum(confs) / len(confs) if confs else 0.0
    return emits, probs, (round(conf, 3) if emits else None)


_VALUE_CACHE = {}


def _load_value_model(project_id: str):
    value_dir = settings.model_dir / project_id / "ocr_value"
    keras_path = value_dir / "ocr_value.keras"
    if not keras_path.exists():
        raise HTTPException(status_code=404,
                            detail="No trained value classifier — train one first "
                                   "(engine=value).")
    mtime = keras_path.stat().st_mtime
    cached = _VALUE_CACHE.get(project_id)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    import tensorflow as tf
    model = tf.keras.models.load_model(str(keras_path))
    labels = (value_dir / "value_labels.txt").read_text().split()
    _VALUE_CACHE[project_id] = (mtime, model, labels)
    return model, labels


def _predict_with_value(project_id: str, img: np.ndarray, gray: np.ndarray):
    """Read a closed-vocabulary plate by classifying each whole line crop
    into one of the project's labeled values — no character-by-character
    decoding, so phantom/dropped digits are structurally impossible.

    Region priority: the trained "plate" region box FIRST — a single
    stable detection whose width doesn't move when a character detector
    hallucinates or misses a box — falling back to character-box row
    grouping only when no plate region model exists. Building the crop
    from character-box extents (the CRNN's approach) makes width a proxy
    for "how many characters", and this classifier can and does use
    width as a real feature — so one missed/extra character-box directly
    swings it toward the wrong value's width band ("10"->"1100" when a
    stray glare box widened the row; "6"->"11" when detection split one
    glyph into two boxes). Training uses the same region source (see
    value_training._plate_region_bbox) so train/infer crops match."""
    from ..tasks.crnn_training import _normalize_line

    model, labels = _load_value_model(project_id)

    plate_first = _yolo_plate_region(project_id, img)
    line_regions = []
    region_source = "full_frame"
    if plate_first:
        region_source = "yolo_plate"
        line_regions = [plate_first]
    boxes = None if plate_first else _yolo_char_boxes(project_id, img)
    if boxes:
        region_source = "yolo_chars"
        med_h = float(np.median([b[3] for b in boxes]))
        med_w = float(np.median([b[2] for b in boxes]))
        rows = []
        for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
            cy = b[1] + b[3] / 2
            for row in rows:
                ry = np.mean([x[1] + x[3] / 2 for x in row])
                if abs(cy - ry) < med_h * 0.6:
                    row.append(b)
                    break
            else:
                rows.append([b])
        rows.sort(key=lambda r: np.mean([b[1] for b in r]))
        for row in rows:
            x1 = max(0, int(round(min(b[0] for b in row) - med_w * 0.5)))
            y1 = max(0, int(round(min(b[1] for b in row) - 4)))
            x2 = min(gray.shape[1], int(round(max(b[0] + b[2] for b in row) + med_w * 0.5)))
            y2 = min(gray.shape[0], int(round(max(b[1] + b[3] for b in row) + 4)))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                line_regions.append((x1, y1, x2 - x1, y2 - y1))
    if not line_regions:
        plate = _yolo_plate_region(project_id, img)
        if plate:
            region_source = "yolo_plate"
            line_regions = [plate]
        else:
            line_regions = [(0, 0, gray.shape[1], gray.shape[0])]

    results, text_lines = [], []
    for (x, y, w, h) in line_regions:
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        norm = _normalize_line(crop).astype(np.float32) / 255.0
        probs = model.predict(norm[None, ..., None], verbose=0)[0]
        order = np.argsort(-probs)
        value = labels[int(order[0])]
        conf = round(float(probs[order[0]]), 4)
        text_lines.append(value)
        results.append({
            "char": value, "confidence": conf, "box": [x, y, w, h],
            "top_values": [{"value": labels[int(i)],
                            "prob": round(float(probs[int(i)]), 4)}
                           for i in order[:3]],
        })
        color = (74, 222, 128)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, value, (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    return {"text": "".join(text_lines), "characters": results, "preview": preview,
            "num_found": len(text_lines), "engine": "value",
            "detector": "yolo" if region_source != "full_frame" else "full_frame",
            "region_source": region_source, "known_values": labels}


def _predict_with_crnn(project_id: str, img: np.ndarray, gray: np.ndarray,
                       pattern: str = None, allowed_values: list = None):
    """
    Read text with the CRNN line recognizer. Uses the project's trained YOLO
    model to locate the text region(s) — exactly the pipeline the user
    described (YOLO gives the area, CRNN reads it).

    Region priority:
    1. Character boxes grouped into row bands — most precise when the
       character detector fires confidently.
    2. The trained "plate" region box (if that class exists) — a robust
       whole-plate crop when character boxes come back weak or empty.
    3. The whole frame, as an absolute last resort.
    """
    from ..tasks.crnn_training import _normalize_line, BLANK_INDEX

    model, charset = _load_crnn(project_id)

    allowed_chars = _project_allowed_chars(project_id)
    allowed_indices = (
        {i for i, c in enumerate(charset) if c in allowed_chars} | {BLANK_INDEX}
        if allowed_chars else None
    )

    boxes = _yolo_char_boxes(project_id, img)
    line_regions = []
    region_source = "full_frame"
    if boxes:
        region_source = "yolo_chars"

        # A plate mounted sideways (serial running vertically, common when
        # the stamped bar sits along the engine edge) puts the character
        # centers on a vertical line — the small-angle de-skew below can't
        # fix that (it caps at 60°), and a vertical line fed to the CRNN is
        # noise. Detect it from the box-center spread and rotate the whole
        # frame 90° first; whether the result is upright or upside-down is
        # settled later by reading each crop at 0° and 180°.
        if len(boxes) >= 3:
            centers0 = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in boxes], dtype=np.float32)
            if float(np.ptp(centers0[:, 1])) > float(np.ptp(centers0[:, 0])) * 1.5:
                H0 = gray.shape[0]
                gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                boxes = [(H0 - (y + h), x, h, w) for (x, y, w, h) in boxes]
                region_source = "yolo_chars_vertical"

        # Row-grouping below only tolerates ~0.6x a character's height of
        # vertical spread between boxes it considers the same line -- fine
        # for a level plate, but a plate photographed at an angle (common
        # on an assembly line, camera never perfectly square-on) spreads a
        # single real line's box centers across far more vertical distance
        # than that tolerance allows. That either fragments one line into
        # several spurious single/two-character "rows" (each read with no
        # sequence context, which is exactly the kind of isolated-character
        # misread this was chasing), or produces line crops that clip into
        # neighboring rows. De-skewing the whole frame around the character
        # cluster's own centroid first — using only detected box centers,
        # no ground-truth needed — makes the text roughly horizontal before
        # any of that grouping/cropping happens.
        angle = _estimate_text_angle(boxes)
        if abs(angle) > 2.0:
            centers = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in boxes], dtype=np.float32)
            cx, cy = float(centers[:, 0].mean()), float(centers[:, 1].mean())
            rot_h, rot_w = gray.shape[:2]
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (rot_w, rot_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.warpAffine(img, M, (rot_w, rot_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((centers.shape[0], 1), dtype=np.float32)
            rotated_centers = (M @ np.hstack([centers, ones]).T).T
            # Move each box's center onto the de-skewed frame, keeping its
            # own width/height -- rotation only needs the centers back onto
            # one horizontal line for row-grouping to work; reshaping every
            # box's extent isn't necessary for that.
            boxes = [
                (int(round(ncx - w0 / 2)), int(round(ncy - h0 / 2)), w0, h0)
                for (_, _, w0, h0), (ncx, ncy) in zip(boxes, rotated_centers)
            ]

        med_h = float(np.median([b[3] for b in boxes]))
        rows = []
        for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
            cy = b[1] + b[3] / 2
            for row in rows:
                ry = np.mean([x[1] + x[3] / 2 for x in row])
                if abs(cy - ry) < med_h * 0.6:
                    row.append(b)
                    break
            else:
                rows.append([b])
        rows.sort(key=lambda r: np.mean([b[1] for b in r]))

        # A character near either end of the line that's more blurred,
        # shadowed, or angled than the rest of the plate often falls below
        # the detector's confidence threshold and never gets a box at all --
        # cropping tightly to only the boxes that DID fire then silently
        # truncates the line before the CRNN ever sees those pixels (e.g.
        # "ZFT4H72838" -> a box around just "T4H7", read as "T4H7"). A
        # modest pad rescues that. Earlier this was 1.5x, but that pulled in
        # enough background/pen-mark/scratch pixels for the CRNN to start
        # hallucinating extra characters at the edges instead ("YYT4G55160"
        # -> "5YYT4G55160") -- confirmed across ~150 real test-on-training
        # results, a worse failure mode than the truncation it fixed.
        # Trimmed back down; _trim_weak_edges() below is the real fix for
        # missed-edge-character truncation now (reads the wider plate
        # region only when the row is suspiciously narrow, then drops
        # low-confidence edge reads instead of blindly keeping them).
        med_w = float(np.median([b[2] for b in boxes]))
        plate = _yolo_plate_region(project_id, img)
        for row in rows:
            x1 = min(b[0] for b in row) - med_w * 0.5
            y1 = min(b[1] for b in row) - 4
            x2 = max(b[0] + b[2] for b in row) + med_w * 0.5
            y2 = max(b[1] + b[3] for b in row) + 4
            if plate:
                px, py, pw, ph = plate
                row_cy = (y1 + y2) / 2
                row_w = x2 - x1
                # Only reach for the full plate width when this row looks
                # suspiciously narrow next to the detected plate box --
                # i.e. the detector likely missed several characters --
                # instead of doing it for every row unconditionally.
                # Guarded to rows that already carry several characters: a
                # 1-2 character row (badge plates like "S4"/"11") is
                # NORMALLY far narrower than its plate box, whose width is
                # mostly logo and chrome frame -- widening those crops fed
                # the stylized "S" swoosh and frame edge to the CRNN, which
                # read them as extra characters ("4" -> "64", "11" -> "110").
                if (len(row) >= 4 and py - med_h <= row_cy <= py + ph + med_h
                        and row_w < pw * 0.7):
                    x1 = min(x1, px)
                    x2 = max(x2, px + pw)
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(gray.shape[1], int(round(x2)))
            y2 = min(gray.shape[0], int(round(y2)))
            line_regions.append((x1, y1, x2 - x1, y2 - y1))
    else:
        plate = _yolo_plate_region(project_id, img)
        if plate:
            region_source = "yolo_plate"
            px, py, pw, ph = plate
            line_regions = [(px, py, pw, ph)]
        else:
            line_regions = [(0, 0, gray.shape[1], gray.shape[0])]

    results, text_lines = [], []
    pattern_applied, pattern_mismatch = False, False
    lexicon = [str(v).strip().upper() for v in (allowed_values or []) if str(v).strip()]
    lexicon_used, lexicon_scores = False, None
    for (x, y, w, h) in line_regions:
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        emits, probs, conf = _decode_line_both_ways(
            model, crop, _normalize_line, BLANK_INDEX, charset, allowed_indices)
        line_text = "".join(e[0] for e in emits)
        if lexicon:
            # Closed-vocabulary read: the plate can only say one of a few
            # known values, so score each candidate's exact CTC probability
            # against this line's whole softmax and keep the best — a
            # greedy misread like "44"/"06"/"" for a plate that can only
            # say "4"/"6"/"10"/"11" is simply not among the candidates.
            picked = _best_lexicon_value(probs, lexicon, charset, BLANK_INDEX)
            if picked is not None:
                line_text, conf, lexicon_scores = picked
                lexicon_used = True
        if pattern and not lexicon_used:
            fixed = _apply_pattern(emits, probs, pattern.strip().upper(), charset)
            if fixed is None:
                pattern_mismatch = True
            elif fixed != line_text:
                line_text = fixed
                pattern_applied = True
            else:
                pattern_applied = True
        text_lines.append(line_text)
        results.append({"char": line_text, "confidence": conf, "box": [x, y, w, h]})
        color = (74, 222, 128)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, line_text, (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    return {"text": "".join(text_lines), "characters": results, "preview": preview,
            "num_found": len(text_lines), "engine": "crnn",
            "detector": "yolo" if region_source != "full_frame" else "full_frame",
            "region_source": region_source,
            "pattern": pattern or None,
            "pattern_applied": pattern_applied,
            "pattern_mismatch": pattern_mismatch,
            "lexicon_used": lexicon_used,
            "lexicon_scores": lexicon_scores,
            "restricted_to": sorted(allowed_chars) if allowed_chars else None}


@router.get("/evaluate-on-training/{project_id}")
async def evaluate_on_training_data(
    project_id: str,
    engine: str = "crnn",
    pattern: str = None,
    allowed_values: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the trained CRNN against every already-annotated image in the
    project instead of testing one photo at a time. Each training image's
    correct text is already known -- it's just the same per-character
    boxes used to train, joined in reading order -- so this reports
    pass/fail per image directly against real ground truth, and points
    straight at the images the model still gets wrong.
    """
    await get_owned_project(project_id, current_user, db)
    if engine not in ("crnn", "value"):
        raise HTTPException(status_code=400,
                            detail="evaluate-on-training supports engine=crnn or engine=value")

    from ..tasks.training import _fetch_training_data, _group_annotations
    from ..tasks.tesseract_training import _boxes_to_lines
    from ..connectors.statedb_connector import StateDBConnector

    state_db = StateDBConnector()
    with state_db.get_session() as conn:
        proj, _, img_rows, ann_rows = _fetch_training_data(
            state_db, conn, project_id, status_filter="annotated")
    if proj is None:
        raise HTTPException(status_code=404, detail="Project not found")
    anns_by_image = _group_annotations(ann_rows)
    values_list = [v.strip() for v in allowed_values.split(",") if v.strip()] \
        if allowed_values else None

    results = []
    for img_row in img_rows:
        real_path = Path(".") / img_row["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
        img = cv2.imread(str(real_path))
        if img is None:
            continue

        scale = 1200 / max(img.shape[:2]) if max(img.shape[:2]) > 1200 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

        lines = _boxes_to_lines(anns_by_image.get(img_row["id"], []), iw, ih)
        truth = "".join("".join(c[0] for c in line) for line in lines)
        if not truth:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            pred = (_predict_with_value(project_id, img, gray) if engine == "value"
                    else _predict_with_crnn(project_id, img, gray, pattern, values_list))
        except HTTPException:
            raise
        except Exception as e:
            results.append({"image_id": img_row["id"], "filename": img_row["filename"],
                            "truth": truth, "predicted": None, "correct": False,
                            "error": str(e)})
            continue

        predicted = pred.get("text", "")
        results.append({
            "image_id": img_row["id"], "filename": img_row["filename"],
            "truth": truth, "predicted": predicted, "correct": predicted == truth,
        })

    results.sort(key=lambda r: r["correct"])  # failures first
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {
        "engine": engine, "pattern": pattern or None,
        "total": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "results": results,
    }


@router.post("/predict/{project_id}")
async def predict_ocr(
    project_id: str,
    file: UploadFile = File(...),
    engine: str = "cnn",
    pattern: str = None,
    allowed_values: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Test window: segment an uploaded photo into characters and read
    each with the trained model. Returns the text, per-character
    confidence, and a preview image with boxes drawn.
    engine=cnn (default) uses the TFLite/Keras classifier;
    engine=tesseract uses the fine-tuned traineddata;
    engine=crnn reads a whole cropped line at once (handles vertical /
    upside-down serials; optional `pattern` constrains the read to the
    serial's format — L=letter, D=digit, ?=any, e.g. LLLDLDDDDD);
    engine=seg uses the trained segment model's per-character dot masks,
    reconnects the dots into a solid glyph, then reads with the CNN
    classifier — the path for dotted/dot-peen engraved characters.
    """
    await get_owned_project(project_id, current_user, db)
    if engine == "cnn":
        model, classes, img_size = _load_project_model(project_id)

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    # Work at a sane resolution
    scale = 1200 / max(img.shape[:2]) if max(img.shape[:2]) > 1200 else 1.0
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if engine == "tesseract":
        return _predict_with_tesseract(project_id, img, gray)
    if engine == "crnn":
        values_list = [v.strip() for v in allowed_values.split(",") if v.strip()] \
            if allowed_values else None
        return _predict_with_crnn(project_id, img, gray, pattern, values_list)
    if engine == "value":
        return _predict_with_value(project_id, img, gray)
    if engine == "seg":
        return _predict_with_seg(project_id, img, gray)

    # Prefer the project's trained YOLO detector for finding characters —
    # robust on reflective metal where thresholding fires on glare/noise.
    detector = "yolo"
    boxes = _yolo_char_boxes(project_id, img)
    region = (0, 0, gray.shape[1], gray.shape[0])
    if boxes is None:
        detector = "classical"
        boxes, region = _segment_with_plate_hint(project_id, img, gray)
    if not boxes:
        raise HTTPException(status_code=422,
                            detail="No characters found. Try a tighter crop of the plate area.")

    # Crop each box exactly like training crops (margin, CLAHE, square pad)
    from ..tasks.ocr_training import _extract_char_crop
    H, W = gray.shape
    crops, kept = [], []
    for (x, y, w, h) in boxes:
        bbox = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
        crop = _extract_char_crop(img, bbox, img_size)
        if crop is not None:
            crops.append(crop)
            kept.append((x, y, w, h))
    if not crops:
        raise HTTPException(status_code=422, detail="Could not crop any characters")

    batch = np.stack(crops).astype(np.float32)[..., None] / 255.0
    probs = model.predict(batch, verbose=0)

    results, text = [], ""
    for (x, y, w, h), p in zip(kept, probs):
        idx = int(p.argmax())
        ch, conf = classes[idx], float(p[idx])
        text += ch
        results.append({"char": ch, "confidence": round(conf, 3),
                        "box": [x, y, w, h]})
        color = (74, 222, 128) if conf >= 0.8 else (21, 170, 250) if conf >= 0.5 else (113, 113, 248)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, ch, (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Show which area was auto-detected as the text band
    rx, ry, rw, rh = region
    if (rw, rh) != (W, H):
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), 1)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None

    return {"text": text, "characters": results, "preview": preview,
            "num_found": len(results), "detector": detector}


# ── Auto-label: pre-annotate pending photos with the trained model ─

class OcrAutoLabelRequest(BaseModel):
    image_ids: Optional[List[str]] = None
    min_conf: float = 0.5
    # "bbox" — plain axis-aligned boxes from the character detector.
    # "polygon" — same detector, but stores the box's own 4 corners as an
    #   editable polyline (a rectangle you can then drag), for angled
    #   LHS/RHS plate photos.
    # "segment" — the trained segmentation model's ACTUAL predicted mask
    #   outline (real per-character polygon, not a box), for projects
    #   annotated with the Segment tool. Requires a trained seed or main
    #   segmentation model to exist.
    shape: str = "bbox"


@router.post("/auto-annotate/{project_id}")
async def ocr_auto_annotate(
    project_id: str,
    body: OcrAutoLabelRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Pre-label pending photos with the project's trained annotation models
    and store the results as 'auto' annotations for human review — the
    OCR equivalent of YOLO's seed-model auto-annotate, so labeling gets
    faster after every training cycle.

    Which model does the work follows the requested shape:

    * "segment"        — the segmentation model's predicted mask outlines.
    * "bbox"/"polygon" — the detection model (main -> seed). It is trained
      per-character-class, so each detection already carries both the box
      and the character, and its labels are written straight through.

    The character classifier is only a fallback for the case where the
    project has no trained detector at all and boxes come from classical
    segmentation, which produces geometry with no label attached. Training
    an OCR model is NOT a prerequisite for auto-annotating: OCR training
    consumes these annotations, it does not produce them.
    """
    await get_owned_project(project_id, current_user, db)
    req = body or OcrAutoLabelRequest()

    from ..services.det_model import resolve_det_model_path

    model = classes = img_size = None
    if req.shape == "segment":
        from ..services.seg_model import resolve_seg_model_path
        if resolve_seg_model_path(project_id) is None:
            raise HTTPException(
                status_code=404,
                detail="No trained segmentation model found. Train the segmentation "
                       "model (seed or main) first, or use shape=bbox/polygon instead.",
            )
    elif resolve_det_model_path(project_id) is None:
        # No detector — classical segmentation finds boxes but cannot name
        # them, so a trained character classifier is the only way to label.
        try:
            model, classes, img_size = _load_project_model(project_id)
        except HTTPException:
            raise HTTPException(
                status_code=404,
                detail="No trained detection model found. Train the seed (or main) "
                       "model first so auto-annotate can label characters.",
            )

    q = select(Image).where(Image.project_id == project_id, Image.status == "pending")
    if req.image_ids:
        q = select(Image).where(Image.project_id == project_id, Image.id.in_(req.image_ids))
    result = await db.execute(q)
    images = result.scalars().all()
    if not images:
        return {"processed": 0, "labeled": 0, "detail": "No pending images."}

    from ..tasks.ocr_training import _extract_char_crop

    processed, labeled = 0, 0
    for img_row in images:
        path = settings.upload_dir.parent / img_row.filepath.lstrip("/")
        img = cv2.imread(str(path))
        if img is None:
            continue
        processed += 1

        scale = 1200 / max(img.shape[:2]) if max(img.shape[:2]) > 1200 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        added = 0

        if req.shape == "segment":
            # Real per-character mask outline from the trained seg model —
            # not a box, the actual predicted polygon (r.masks.xy).
            results = _yolo_seg_predict_raw(project_id, img, conf=req.min_conf)
            if results is None:
                continue
            for r in results:
                if r.masks is None or r.boxes is None:
                    continue
                names = r.names
                for poly, b in zip(r.masks.xy, r.boxes):
                    label = str(names.get(int(b.cls[0]), "")).strip()
                    conf = float(b.conf[0])
                    if not label or poly is None or len(poly) < 3 or conf < req.min_conf:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    if w < 2 or h < 2:
                        continue
                    bbox = [(x1 + w / 2) / W, (y1 + h / 2) / H, w / W, h / H]
                    points = [[float(px) / W, float(py) / H] for px, py in poly]
                    db.add(Annotation(
                        image_id=img_row.id,
                        class_name=label,
                        bbox=bbox,
                        annotation_type="polygon",
                        points=points,
                        source="auto",
                    ))
                    added += 1
        else:
            detected = _yolo_char_boxes_labeled(project_id, img)

            if detected is not None:
                # The detector already predicted the character for every box
                # it found — write box + label straight through.
                for (x, y, w, h, label, conf) in detected:
                    if conf < req.min_conf:
                        continue
                    bbox = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
                    db.add(Annotation(
                        image_id=img_row.id,
                        class_name=label,
                        bbox=bbox,
                        annotation_type="polygon" if req.shape == "polygon" else "bbox",
                        points=_bbox_to_points(bbox) if req.shape == "polygon" else None,
                        source="auto",
                    ))
                    added += 1
            else:
                # No detector: classical segmentation gives unlabeled
                # geometry, so the character classifier names each crop.
                boxes, _region = _segment_with_plate_hint(project_id, img, gray)
                crops, kept = [], []
                for (x, y, w, h) in boxes:
                    bbox = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
                    crop = _extract_char_crop(img, bbox, img_size)
                    if crop is not None:
                        crops.append(crop)
                        kept.append(bbox)
                if not crops:
                    continue

                batch = np.stack(crops).astype(np.float32)[..., None] / 255.0
                probs = model.predict(batch, verbose=0)

                for bbox, p in zip(kept, probs):
                    idx = int(p.argmax())
                    if float(p[idx]) < req.min_conf:
                        continue
                    db.add(Annotation(
                        image_id=img_row.id,
                        class_name=classes[idx],
                        bbox=bbox,
                        annotation_type="polygon" if req.shape == "polygon" else "bbox",
                        points=_bbox_to_points(bbox) if req.shape == "polygon" else None,
                        source="auto",
                    ))
                    added += 1

        if added:
            img_row.status = "annotated"
            labeled += added

    await db.commit()
    return {"processed": processed, "labeled": labeled,
            "detail": f"Pre-labeled {labeled} characters across {processed} photos. "
                      "Review and correct them, then retrain."}


# ── Active Learning (OCR) ───────────────────────────────────────
# Same idea as the YOLO active-learning module (score_unlabeled_images /
# suggest_for_review in tasks/active_learning.py): whichever model would do
# the auto-annotating scores the pending photos, and the ones it is least
# sure about are the most useful for a human to label next. Runs
# synchronously (scoring is cheap) rather than as a Celery job — no
# queue/polling needed for a first pass.

class OcrSuggestReviewRequest(BaseModel):
    budget: int = 10  # 0 = return every scored pending image
    shape: str = "bbox"  # "bbox" | "polygon" -> detector, "segment" -> seg model


@router.post("/active-learning/suggest/{project_id}")
async def ocr_suggest_for_review(
    project_id: str,
    body: OcrSuggestReviewRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rank pending images by model uncertainty — most useful to label first.

    Scores with whichever model ocr_auto_annotate would use for the same
    shape: the segmentation model for "segment", the detection model for
    "bbox"/"polygon", and the character classifier only as the fallback
    for a project with no trained detector.
    """
    await get_owned_project(project_id, current_user, db)
    req = body or OcrSuggestReviewRequest()

    from ..services.det_model import resolve_det_model_path

    model = classes = img_size = None
    use_seg = req.shape == "segment"
    use_det = not use_seg and resolve_det_model_path(project_id) is not None
    if use_seg:
        from ..services.seg_model import resolve_seg_model_path
        if resolve_seg_model_path(project_id) is None:
            raise HTTPException(
                status_code=404,
                detail="No trained segmentation model found. Train the segmentation "
                       "model (seed or main) first, or use shape=bbox/polygon instead.",
            )
    elif not use_det:
        try:
            model, classes, img_size = _load_project_model(project_id)
        except HTTPException:
            raise HTTPException(
                status_code=404,
                detail="No trained detection model found. Train the seed (or main) "
                       "model first so pending photos can be scored.",
            )

    result = await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "pending")
    )
    images = result.scalars().all()
    if not images:
        return {"status": "success", "total_pending": 0, "total_scored": 0, "suggestions": []}

    from ..tasks.ocr_training import _extract_char_crop

    scored = []
    for img_row in images:
        path = settings.upload_dir.parent / img_row.filepath.lstrip("/")
        img = cv2.imread(str(path))
        if img is None:
            continue

        scale = 1200 / max(img.shape[:2]) if max(img.shape[:2]) > 1200 else 1.0
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        if use_seg:
            confs = []
            results = _yolo_seg_predict_raw(project_id, img, conf=0.0)
            if results is not None:
                for r in results:
                    if r.masks is None or r.boxes is None:
                        continue
                    for b in r.boxes:
                        confs.append(float(b.conf[0]))

            if not confs:
                scored.append({
                    "image_id": img_row.id,
                    "filename": img_row.filename,
                    "char_count": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "reason": "No characters detected — segmentation may be struggling on this photo",
                })
                continue

            avg_conf = float(np.mean(confs))
            min_conf = float(np.min(confs))
            scored.append({
                "image_id": img_row.id,
                "filename": img_row.filename,
                "char_count": len(confs),
                "avg_confidence": avg_conf,
                "min_confidence": min_conf,
                "reason": f"{len(confs)} chars — lowest confidence {min_conf:.0%}, avg {avg_conf:.0%}",
            })
            continue

        if use_det:
            detected = _yolo_char_boxes_labeled(project_id, img, conf=0.0)
            confs = [c for *_box, _label, c in detected] if detected else []

            if not confs:
                scored.append({
                    "image_id": img_row.id,
                    "filename": img_row.filename,
                    "char_count": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": 0.0,
                    "reason": "No characters detected — the detector may be struggling on this photo",
                })
                continue

            avg_conf = float(np.mean(confs))
            min_conf = float(np.min(confs))
            scored.append({
                "image_id": img_row.id,
                "filename": img_row.filename,
                "char_count": len(confs),
                "avg_confidence": avg_conf,
                "min_confidence": min_conf,
                "reason": f"{len(confs)} chars — lowest confidence {min_conf:.0%}, avg {avg_conf:.0%}",
            })
            continue

        boxes, _region = _segment_with_plate_hint(project_id, img, gray)

        crops = []
        for (x, y, w, h) in boxes:
            bbox = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
            crop = _extract_char_crop(img, bbox, img_size)
            if crop is not None:
                crops.append(crop)

        if not crops:
            # No characters found at all is the hardest case — segmentation
            # itself may be failing on this photo (angle, glare, low contrast).
            scored.append({
                "image_id": img_row.id,
                "filename": img_row.filename,
                "char_count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "reason": "No characters detected — segmentation may be struggling on this photo",
            })
            continue

        batch = np.stack(crops).astype(np.float32)[..., None] / 255.0
        probs = model.predict(batch, verbose=0)
        max_probs = probs.max(axis=1)
        avg_conf = float(max_probs.mean())
        min_conf = float(max_probs.min())

        scored.append({
            "image_id": img_row.id,
            "filename": img_row.filename,
            "char_count": len(crops),
            "avg_confidence": avg_conf,
            "min_confidence": min_conf,
            "reason": f"{len(crops)} chars — lowest confidence {min_conf:.0%}, avg {avg_conf:.0%}",
        })

    # No-detection images first (hardest), then ascending average confidence
    scored.sort(key=lambda s: (s["char_count"] > 0, s["avg_confidence"]))

    budget = req.budget if req.budget > 0 else len(scored)
    return {
        "status": "success",
        "total_pending": len(images),
        "total_scored": len(scored),
        "suggestions": scored[:budget],
    }
