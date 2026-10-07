import base64
import io
import json
import os
import re
import uuid
import zipfile
from collections import Counter
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..services.cls_model import CLS_FILES, cls_dir, load_cls_meta, predict_crop_cls
from ..config import settings
from ..tasks.celery_app import celery_app
from ..tasks.crop_cls_training import build_label_preview, train_crop_cls_model

router = APIRouter(prefix="/crop-cls", tags=["crop-cls"])


class TrainCropClsRequest(BaseModel):
    # "crop": detector finds the region, classifier sees the crop.
    # "whole": classifier sees the full frame (baseline for comparison).
    mode: str = "crop"
    # Only boxes with these class_names count as the region (default: all boxes).
    region_classes: Optional[List[str]] = None
    det_model_name: str = "yolo11s.pt"
    cls_model_name: str = "yolo11s-cls.pt"
    det_epochs: int = 60
    cls_epochs: int = 40
    det_imgsz: int = 640
    cls_imgsz: int = 224
    margin: float = 0.12
    preprocess: bool = True
    batch: int = 32
    aug_fliplr: float = 0.5
    # Nested-box labelling: the crop_class box (e.g. "engine") is cut out and
    # labelled by the label_classes box inside it; empty_label if none is.
    crop_class: Optional[str] = None
    label_classes: Optional[List[str]] = None
    empty_label: Optional[str] = "no_cover"
    min_overlap: float = 0.5
    augment: bool = True
    rotate_360: bool = False
    rotate_copies: int = 4


class PreviewRequest(BaseModel):
    crop_class: str
    label_classes: List[str]
    empty_label: Optional[str] = "no_cover"
    min_overlap: float = 0.5
    margin: float = 0.12


@router.post("/preview/{project_id}")
async def preview_label_set(
    project_id: str,
    body: PreviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dry-run the classification set: counts per label, skipped conflicts /
    empties and sample crops, without training anything."""
    await get_owned_project(project_id, current_user, db)
    if not body.label_classes:
        raise HTTPException(status_code=400, detail="label_classes must not be empty")
    result = await run_in_threadpool(
        build_label_preview, project_id, body.crop_class, body.label_classes,
        body.empty_label, body.min_overlap, body.margin)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/train/{project_id}")
async def start_crop_cls_training(
    project_id: str,
    body: TrainCropClsRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue crop+classify (or whole-image classify) training. Poll the
    returned task_id via GET /pipeline/task-status/{task_id}."""
    await get_owned_project(project_id, current_user, db)
    req = body or TrainCropClsRequest()
    if req.mode not in ("crop", "whole"):
        raise HTTPException(status_code=400, detail="mode must be 'crop' or 'whole'")
    if req.crop_class and not req.label_classes:
        raise HTTPException(status_code=400, detail="label_classes is required with crop_class")
    task = train_crop_cls_model.delay(project_id, **req.model_dump())
    return {"task_id": task.id, "status": "queued"}


@router.get("/model-status/{project_id}")
async def crop_cls_model_status(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    d = cls_dir(project_id)
    meta = load_cls_meta(project_id)
    return {
        "has_classifier": (d / CLS_FILES["classifier"]).exists(),
        "has_detector": (d / CLS_FILES["detector"]).exists(),
        "meta": meta,
    }


@router.post("/predict/{project_id}")
async def crop_cls_predict(
    project_id: str,
    file: UploadFile = File(...),
    det_conf: float = 0.25,
    cls_threshold: float = 0.6,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the trained pipeline on an uploaded image. Returns the state,
    both confidences, the region box (normalized xc,yc,w,h) and a preview
    image with the box + label drawn. status is one of ok | uncertain |
    region_not_found | no_model."""
    await get_owned_project(project_id, current_user, db)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    result = await run_in_threadpool(predict_crop_cls, project_id, img, det_conf, cls_threshold)
    if result["status"] == "no_model":
        raise HTTPException(status_code=404, detail="No trained crop+classify model for this project")

    preview = img.copy()
    h, w = preview.shape[:2]
    label = result.get("state") or result["status"]
    if result.get("confidence") is not None:
        label = f"{label} {result['confidence']:.2f}"
    if result.get("bbox"):
        xc, yc, bw, bh = result["bbox"]
        p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        color = (0, 200, 0) if result["status"] == "ok" else (0, 165, 255)
        cv2.rectangle(preview, p1, p2, color, max(2, w // 400))
        cv2.putText(preview, label, (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1500), color, 2)
    else:
        cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    ok, buf = cv2.imencode(".jpg", preview)
    result["preview"] = (
        "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    )
    return result


@router.post("/stop/{task_id}")
async def stop_crop_cls_training(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    """Ask a running crop+classify training to stop at the next epoch (same
    Redis flag the epoch callback already watches) and revoke it if queued."""
    import redis as redis_lib
    try:
        redis_lib.from_url(settings.redis_url, socket_connect_timeout=2).setex(
            f"stop_training:{task_id}", 300, "1")
    except Exception:
        pass
    celery_app.control.revoke(task_id, terminate=False)
    return {"status": "stopping"}


# ── Folder-of-images import (one class per folder, whole-image labels) ──────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_ZIP_FILES = 20000
_MAX_ZIP_ENTRY = 50 * 1024 * 1024


def _clean_label(raw) -> Optional[str]:
    label = re.sub(r"\s+", " ", str(raw or "")).strip()[:100]
    return label if label and label not in (".", "..") else None


def label_from_path(path: str) -> Optional[str]:
    """Class = the folder an image sits in: 'cut/a.jpg' and 'data/cut/a.jpg'
    both give 'cut'. A bare filename has no folder, so no label."""
    parts = [p for p in path.replace("\\", "/").split("/") if p and p != "__MACOSX"]
    return _clean_label(parts[-2]) if len(parts) >= 2 else None


def _decode_ok(contents: bytes) -> tuple[int, int]:
    """Validate with PIL like /images/upload does; returns (width, height)."""
    from PIL import Image as PILImage
    with PILImage.open(io.BytesIO(contents)) as im:
        im.load()
        if im.format not in {"JPEG", "PNG", "GIF", "WEBP", "BMP"}:
            raise ValueError(f"{im.format} images can't be displayed by browsers")
        if im.mode == "CMYK":
            raise ValueError("CMYK images render black in browsers -- convert to RGB")
        return im.size


@router.post("/import-folders/{project_id}")
async def import_class_folders(
    project_id: str,
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Import classification images where the FOLDER is the class.

    Send either image files plus ``labels`` (a JSON array, one class per
    file, taken from each file's folder in the browser), or .zip files
    laid out as ``<class>/<image>``. Every image becomes an annotated
    image carrying one whole-frame box labelled with its class, so the
    "whole image" classifier mode can train straight from it. New classes
    are added to the project's class list.
    """
    project = await get_owned_project(project_id, current_user, db)

    try:
        given = json.loads(labels) if labels else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="labels must be a JSON array")
    if not isinstance(given, list):
        raise HTTPException(status_code=400, detail="labels must be a JSON array")

    # (filename, label, bytes) work list
    work, failed = [], []
    for idx, f in enumerate(files):
        name = f.filename or "file"
        data = await f.read()
        if name.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                failed.append({"filename": name, "reason": "not a valid zip"})
                continue
            entries = [i for i in zf.infolist() if not i.is_dir()]
            if len(entries) > _MAX_ZIP_FILES:
                failed.append({"filename": name, "reason": "too many files in zip"})
                continue
            for info in entries:
                if os.path.splitext(info.filename)[1].lower() not in _IMG_EXTS:
                    continue
                if info.file_size > _MAX_ZIP_ENTRY:
                    failed.append({"filename": info.filename, "reason": "file too large"})
                    continue
                label = label_from_path(info.filename)
                if not label:
                    failed.append({"filename": info.filename, "reason": "not inside a class folder"})
                    continue
                work.append((os.path.basename(info.filename), label, zf.read(info)))
        else:
            label = _clean_label(given[idx]) if idx < len(given) else label_from_path(name)
            if not label:
                failed.append({"filename": name, "reason": "no class folder / label"})
                continue
            work.append((os.path.basename(name), label, data))

    if not work and not failed:
        raise HTTPException(status_code=400, detail="No images found")

    project_dir = settings.upload_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    existing = list(project.classes or [])
    per_class: Counter = Counter()

    for fname, label, data in work:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _IMG_EXTS:
            failed.append({"filename": fname, "reason": f"unsupported type {ext or '?'}"})
            continue
        try:
            if not data:
                raise ValueError("file is empty")
            width, height = _decode_ok(data)
        except Exception as e:
            failed.append({"filename": fname, "reason": str(e)})
            continue
        unique = f"{uuid.uuid4()}{ext}"
        (project_dir / unique).write_bytes(data)
        img = Image(project_id=project_id, filename=fname,
                    filepath=f"/uploads/{project_id}/{unique}",
                    width=width, height=height, status="annotated")
        db.add(img)
        await db.flush()
        db.add(Annotation(image_id=img.id, class_name=label, bbox=[0.5, 0.5, 1.0, 1.0],
                          annotation_type="bbox", state=label, source="folder"))
        if label not in existing:
            existing.append(label)
        per_class[label] += 1

    project.classes = existing
    await db.commit()
    return {
        "imported": sum(per_class.values()),
        "per_class": dict(per_class),
        "failed": failed,
        "classes": existing,
    }
