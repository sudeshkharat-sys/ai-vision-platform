import base64
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..database import get_db
from ..models.user import User
from ..services.cls_model import CLS_FILES, cls_dir, load_cls_meta, predict_crop_cls
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
