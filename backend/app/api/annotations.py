from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.project import Project
from ..models.user import User
from ..schemas.base import AnnotationCreate, AnnotationResponse
from ..api.auth import get_current_user
from ..api.deps import get_owned_image, get_owned_annotation
from typing import List
from pydantic import BaseModel


class ClassifyBody(BaseModel):
    class_name: str

class BboxUpdateBody(BaseModel):
    bbox: list

class PointsUpdateBody(BaseModel):
    points: List[List[float]]


router = APIRouter(prefix="/annotations", tags=["annotations"])


def _points_to_bbox(points: list) -> list:
    """Axis-aligned bounding box [x_center, y_center, w, h] enclosing normalized points.

    Keeps the legacy `bbox` column in sync for polygon annotations so consumers
    that only understand boxes (YOLO training export, NMS, etc.) keep working.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def _bbox_to_points(bbox: list) -> list:
    """Four corners (TL, TR, BR, BL) of a normalized [xc, yc, w, h] bbox."""
    xc, yc, w, h = bbox
    x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


async def _ensure_class_registered(db: AsyncSession, project_id: str, class_name: str):
    """Add class_name to project.classes if it isn't there yet.

    Training builds data.yaml straight from project.classes and silently
    skips any annotation whose class_name isn't in that list (see
    _write_split in tasks/training.py) — so a brand-new class typed while
    annotating must be registered here, or it vanishes from every future
    training run with no error.
    """
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if project and class_name not in project.classes:
        project.classes = [*project.classes, class_name]


@router.post("", response_model=AnnotationResponse)
async def create_annotation(
    data: AnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify the target image belongs to the current user
    image = await get_owned_image(data.image_id, current_user, db)
    await _ensure_class_registered(db, image.project_id, data.class_name)

    bbox = data.bbox
    if data.annotation_type == "polygon" and data.points and bbox is None:
        bbox = _points_to_bbox(data.points)

    annotation = Annotation(
        image_id=data.image_id,
        class_name=data.class_name,
        bbox=bbox,
        annotation_type=data.annotation_type,
        points=data.points,
        source=data.source,
    )
    db.add(annotation)
    image.status = "annotated"
    await db.commit()
    await db.refresh(annotation)
    return annotation


@router.get("/image/{image_id}", response_model=List[AnnotationResponse])
async def list_image_annotations(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_image(image_id, current_user, db)
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    return result.scalars().all()


@router.patch("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation_bbox(
    annotation_id: str,
    data: BboxUpdateBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the bounding box of an annotation (for drag/resize editing)."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    ann.bbox = data.bbox
    await db.commit()
    await db.refresh(ann)
    return ann


@router.patch("/{annotation_id}/points", response_model=AnnotationResponse)
async def update_annotation_points(
    annotation_id: str,
    data: PointsUpdateBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the vertices of a polygon/polyline annotation (vertex drag editing)."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    if len(data.points) < 3:
        raise HTTPException(status_code=400, detail="A polygon needs at least 3 points.")
    ann.points = data.points
    ann.annotation_type = "polygon"
    ann.bbox = _points_to_bbox(data.points)
    await db.commit()
    await db.refresh(ann)
    return ann


@router.patch("/{annotation_id}/convert-to-polygon", response_model=AnnotationResponse)
async def convert_annotation_to_polygon(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Replace a box annotation with its 4-corner polygon equivalent, ready to
    drag into shape — the starting point for "Replace box with polyline"."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    if ann.annotation_type != "polygon":
        if not ann.bbox:
            raise HTTPException(status_code=400, detail="Annotation has no bbox to convert.")
        ann.points = _bbox_to_points(ann.bbox)
        ann.annotation_type = "polygon"
        await db.commit()
        await db.refresh(ann)
    return ann


@router.patch("/image/{image_id}/convert-to-polygon", response_model=List[AnnotationResponse])
async def convert_image_annotations_to_polygon(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bulk-convert every box annotation on an image to a polygon — backs the
    canvas toolbar's "Replace box with polyline" button."""
    await get_owned_image(image_id, current_user, db)
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    anns = result.scalars().all()
    for ann in anns:
        if ann.annotation_type != "polygon" and ann.bbox:
            ann.points = _bbox_to_points(ann.bbox)
            ann.annotation_type = "polygon"
    await db.commit()
    for ann in anns:
        await db.refresh(ann)
    return anns


@router.patch("/{annotation_id}/classify", response_model=AnnotationResponse)
async def classify_annotation(
    annotation_id: str,
    data: ClassifyBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Assign a class name to an AI-detected annotation and promote it to manual."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    image = await get_owned_image(ann.image_id, current_user, db)
    await _ensure_class_registered(db, image.project_id, data.class_name)
    ann.class_name = data.class_name
    ann.source = "manual"
    await db.commit()
    await db.refresh(ann)
    return ann


@router.patch("/{annotation_id}/verify")
async def verify_annotation(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark an AI annotation as verified by changing source to 'manual'."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    ann.source = "manual"
    await db.commit()
    return {"status": "verified", "id": annotation_id}


@router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an annotation (used for 'Reject')."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    image_id = ann.image_id
    await db.delete(ann)

    # If no annotations remain, set image status back to pending
    count_res = await db.execute(
        select(Annotation).where(Annotation.image_id == image_id)
    )
    if not count_res.scalars().first():
        img_res = await db.execute(select(Image).where(Image.id == image_id))
        img = img_res.scalar_one_or_none()
        if img:
            img.status = "pending"

    await db.commit()
    return {"status": "deleted"}
