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
            if area < 40:                    # specks
                continue
            if h < H * 0.04 or h > H * 0.95: # too small / whole image
                continue
            if w > W * 0.5:                  # merged blob / border
                continue
            ar = w / h
            if ar < 0.05 or ar > 1.6:        # not character-shaped ("1" is thin)
                continue
            out.append((int(x), int(y), int(w), int(h)))
        return out

    # Characters can be darker or lighter than the metal — try both,
    # keep whichever polarity finds more character-like shapes.
    _, inv = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, pos = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cand_inv, cand_pos = boxes_for(inv), boxes_for(pos)
    boxes = cand_inv if len(cand_inv) >= len(cand_pos) else cand_pos
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
    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: np.mean([b[1] for b in r]))
    return [b for row in rows for b in row]


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
    boxes = _segment_characters(gray)
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

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None

    return {"text": text, "characters": results, "preview": preview,
            "num_found": len(results)}
