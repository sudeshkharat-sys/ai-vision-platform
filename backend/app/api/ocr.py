from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
