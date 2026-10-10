import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..config import settings
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.project import Project
from ..services.image_files import resolve_image_file
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
    known = set(project.classes or [])
    by_name = {s["name"]: s for s in segments}
    for s in summary["segments"]:
        s["samples"] = samples.get(s["name"], [])
        # class names in a rule that this project doesn't have (rules are case-sensitive)
        s["unknown_classes"] = [c for c in by_name[s["name"]]["counts"] if c not in known]
    summary["annotated_images"] = len(ids)
    return summary


class CopyBody(BaseModel):
    segments: Optional[List[dict]] = None      # default: the saved ones
    segment_names: Optional[List[str]] = None  # only these (default: all given)
    name: Optional[str] = None


@router.post("/{project_id}/create-copy")
async def create_segment_copy(
    project_id: str,
    body: CopyBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Make a NEW project holding only the images that match the chosen
    segments (files copied, every annotation on those images kept, image
    status kept). The source project is not touched. Models and jobs are not
    copied; the copy starts untrained like any new project."""
    source = await get_owned_project(project_id, current_user, db)
    segments = _clean(body.segments if body.segments is not None else (source.segments or []))
    if body.segment_names is not None:
        segments = [s for s in segments if s["name"] in set(body.segment_names)]
    if not segments:
        raise HTTPException(status_code=422, detail="Pick at least one segment")

    imgs = (await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "annotated")
    )).scalars().all()
    anns = defaultdict(list)
    if imgs:
        for a in (await db.execute(
                select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
            anns[a.image_id].append(a)
    assignment, summary = assign_segments(
        {k: [{"class_name": a.class_name} for a in v] for k, v in anns.items()},
        segments, image_ids=[i.id for i in imgs])
    if not assignment:
        raise HTTPException(status_code=422, detail="No annotated images match these segments")

    name = (body.name or "").strip() or f"{source.name} ({', '.join(s['name'] for s in segments)})"[:250]
    clash = (await db.execute(
        select(Project.name).where(Project.user_id == current_user.id, Project.name.like(f"{name}%"))
    )).scalars().all()
    final, n = name, 2
    while final in set(clash):
        final, n = f"{name} {n}", n + 1

    new_project = Project(name=final, description=source.description,
                          classes=list(source.classes or []), project_type=source.project_type,
                          segments=segments, user_id=current_user.id)
    db.add(new_project)
    await db.flush()
    dest_dir = settings.upload_dir / new_project.id
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    for img in imgs:
        if img.id not in assignment:
            continue
        src_file = resolve_image_file(project_id, img.filepath)
        if src_file is None:
            missing.append(img.filename)      # never create a row without its picture
            continue
        new_file = f"{uuid.uuid4()}{src_file.suffix}"
        shutil.copy2(src_file, dest_dir / new_file)
        new_img = Image(project_id=new_project.id, filename=img.filename,
                        filepath=f"/uploads/{new_project.id}/{new_file}",
                        width=img.width, height=img.height, status=img.status)
        db.add(new_img)
        await db.flush()
        for a in anns.get(img.id, []):
            db.add(Annotation(
                image_id=new_img.id, class_name=a.class_name,
                bbox=list(a.bbox) if a.bbox else None, source=a.source,
                annotation_type=a.annotation_type,
                points=list(a.points) if a.points else None, state=a.state))
        copied += 1
    await db.commit()
    return {"project_id": new_project.id, "name": final, "images": copied,
            "per_segment": summary["segments"], "left_out": summary["unmatched"],
            "missing_files": len(missing), "missing_examples": missing[:5]}


class ImagesBody(BaseModel):
    segments: Optional[List[dict]] = None   # default: the saved ones
    segment: str                            # segment name, or "__unmatched__"
    offset: int = 0
    limit: int = 24


@router.post("/{project_id}/images")
async def segment_images(
    project_id: str,
    body: ImagesBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The images a segment would take (or, with "__unmatched__", the ones no
    segment takes), each with its boxes, so the selection can be checked by eye."""
    project = await get_owned_project(project_id, current_user, db)
    segments = _clean(body.segments if body.segments is not None else (project.segments or []))
    imgs = (await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "annotated")
        .order_by(Image.created_at, Image.id)
    )).scalars().all()
    anns = defaultdict(list)
    if imgs:
        for a in (await db.execute(
                select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
            anns[a.image_id].append(a)
    assignment, _ = assign_segments(
        {k: [{"class_name": a.class_name} for a in v] for k, v in anns.items()},
        segments, image_ids=[i.id for i in imgs])
    if body.segment == "__unmatched__":
        chosen = [i for i in imgs if i.id not in assignment]
    else:
        chosen = [i for i in imgs if assignment.get(i.id, {}).get("name") == body.segment]
    limit = max(1, min(body.limit, 60))
    page = chosen[max(0, body.offset): max(0, body.offset) + limit]
    return {
        "total": len(chosen),
        "images": [{
            "id": i.id, "filename": i.filename, "filepath": i.filepath,
            "width": i.width, "height": i.height,
            "annotations": [{"class_name": a.class_name, "bbox": a.bbox, "points": a.points,
                             "annotation_type": a.annotation_type} for a in anns.get(i.id, [])],
        } for i in page],
    }
