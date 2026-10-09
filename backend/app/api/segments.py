from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..services.data_segments import assign_segments, normalize_segments

router = APIRouter(prefix="/segments", tags=["segments"])

_SAMPLES = 6


class SegmentsBody(BaseModel):
    # Raw rule dicts: {name, label, counts: {class: {min, max}}, allow_other, total}
    segments: Optional[List[dict]] = None


def _clean(segments):
    try:
        return normalize_segments(segments)
    except (ValueError, TypeError, AttributeError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/{project_id}")
async def get_segments(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await get_owned_project(project_id, current_user, db)
    return {"segments": project.segments or [], "classes": list(project.classes or [])}


@router.put("/{project_id}")
async def save_segments(
    project_id: str,
    body: SegmentsBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await get_owned_project(project_id, current_user, db)
    project.segments = _clean(body.segments or [])
    await db.commit()
    return {"segments": project.segments}


@router.post("/{project_id}/preview")
async def preview_segments(
    project_id: str,
    body: SegmentsBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dry run: how many annotated images each segment would take, a few
    sample images per segment, and which annotation combinations match no
    segment. ``segments`` defaults to the saved ones (unsaved edits can be
    previewed by sending them). Nothing is changed."""
    project = await get_owned_project(project_id, current_user, db)
    segments = _clean(body.segments if body.segments is not None else (project.segments or []))

    imgs = (await db.execute(
        select(Image.id, Image.filename, Image.filepath)
        .where(Image.project_id == project_id, Image.status == "annotated")
    )).all()
    ids = [r.id for r in imgs]
    anns = defaultdict(list)
    if ids:
        rows = (await db.execute(
            select(Annotation.image_id, Annotation.class_name)
            .where(Annotation.image_id.in_(ids))
        )).all()
        for r in rows:
            anns[r.image_id].append({"class_name": r.class_name})

    assignment, summary = assign_segments(anns, segments, image_ids=ids)
    samples = defaultdict(list)
    for r in imgs:
        seg = assignment.get(r.id)
        if seg and len(samples[seg["name"]]) < _SAMPLES:
            samples[seg["name"]].append(
                {"id": r.id, "filename": r.filename, "filepath": r.filepath})
    for s in summary["segments"]:
        s["samples"] = samples.get(s["name"], [])
    summary["annotated_images"] = len(ids)
    return summary
