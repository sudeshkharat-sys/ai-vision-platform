import base64
import json
from datetime import datetime

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

router = APIRouter(prefix="/ocr", tags=["ocr"])

OCR_FILES = {
    "tflite": "ocr_model.tflite",
    "keras": "ocr_model.keras",
    "labels": "labels.txt",
    "meta": "ocr_meta.json",
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

    return {"has_model": files["tflite"]["exists"], "files": files, "meta": meta}


@router.get("/download/{project_id}/{file_type}")
async def download_ocr_file(
    project_id: str,
    file_type: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download a trained OCR artifact: tflite | keras | labels | meta."""
    await get_owned_project(project_id, current_user, db)
    if file_type not in OCR_FILES:
        raise HTTPException(status_code=400,
                            detail=f"file_type must be one of {list(OCR_FILES)}")
    path = settings.model_dir / project_id / "ocr" / OCR_FILES[file_type]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found — train the OCR model first")
    return FileResponse(path=str(path), media_type="application/octet-stream",
                        filename=OCR_FILES[file_type])


# ── Test window: run the trained model on an uploaded photo ───────

# Model cache: project_id -> (mtime, keras_model, classes, img_size)
_MODEL_CACHE: dict = {}


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
            if ar < 0.04 or ar > 1.6:        # not character-shaped ("1" is thin)
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
    return [b for row in rows for b in row]


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


@router.post("/predict/{project_id}")
async def predict_ocr(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Test window: segment an uploaded photo into characters and read
    each with the trained model. Returns the text, per-character
    confidence, and a preview image with boxes drawn.
    """
    await get_owned_project(project_id, current_user, db)
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
    boxes, region = _segment_in_region(gray)
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
            "num_found": len(results)}


# ── Auto-label: pre-annotate pending photos with the trained model ─

class OcrAutoLabelRequest(BaseModel):
    image_ids: Optional[List[str]] = None
    min_conf: float = 0.5


@router.post("/auto-annotate/{project_id}")
async def ocr_auto_annotate(
    project_id: str,
    body: OcrAutoLabelRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Use the trained OCR model to pre-label pending photos: segment each
    plate into characters, predict each character, and store the boxes
    as 'auto' annotations for human review/correction. The OCR
    equivalent of YOLO's seed-model auto-annotate — labeling gets
    faster after every training cycle.
    """
    await get_owned_project(project_id, current_user, db)
    model, classes, img_size = _load_project_model(project_id)
    req = body or OcrAutoLabelRequest()

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

        boxes, _region = _segment_in_region(gray)
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

        added = 0
        for bbox, p in zip(kept, probs):
            idx = int(p.argmax())
            if float(p[idx]) < req.min_conf:
                continue
            db.add(Annotation(
                image_id=img_row.id,
                class_name=classes[idx],
                bbox=bbox,
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
