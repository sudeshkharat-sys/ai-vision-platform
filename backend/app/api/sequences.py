"""
sequences.py
~~~~~~~~~~~~
CRUD API for region sequences — ordered sets of regions (box/line) that a
tracked object must pass through, in order, to complete a "sequence"
(e.g. a bulb visiting socket 1 -> 2 -> 3, or a finger visiting key-zones
S -> U -> D -> E -> S -> H).

Also includes running a saved sequence against an uploaded video: kicks off
a Celery task (app.tasks.sequence_run) that detects objects frame-by-frame
and feeds them into the SequenceRunState state machine (app.services). The
live/webcam path is still not implemented — see the design doc.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config import settings
from ..database import get_db
from ..models.sequence import RegionSequence, SequenceRun
from ..models.user import User
from ..schemas.base import (
    RegionSequenceCreate,
    RegionSequenceUpdate,
    RegionSequenceResponse,
    SequenceRunResponse,
    SequenceTestImageRequest,
    SequenceTestImageResponse,
)
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_sequence, get_owned_video, get_owned_sequence_run, get_owned_image

router = APIRouter(prefix="/sequences", tags=["sequences"])


@router.post("/project/{project_id}", response_model=RegionSequenceResponse)
async def create_sequence(
    project_id: str,
    body: RegionSequenceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new region sequence for a project."""
    await get_owned_project(project_id, current_user, db)

    seq = RegionSequence(
        project_id=project_id,
        name=body.name,
        mode=body.mode,
        overlap_threshold=body.overlap_threshold,
        steps=[step.model_dump() for step in body.steps],
        ref_image_width=body.ref_image_width,
        ref_image_height=body.ref_image_height,
    )
    db.add(seq)
    await db.commit()
    await db.refresh(seq)
    return seq


@router.get("/project/{project_id}", response_model=List[RegionSequenceResponse])
async def list_project_sequences(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all region sequences belonging to a project."""
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(
        select(RegionSequence)
        .where(RegionSequence.project_id == project_id)
        .order_by(RegionSequence.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{sequence_id}", response_model=RegionSequenceResponse)
async def get_sequence(
    sequence_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_owned_sequence(sequence_id, current_user, db)


@router.put("/{sequence_id}", response_model=RegionSequenceResponse)
async def update_sequence(
    sequence_id: str,
    body: RegionSequenceUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    seq = await get_owned_sequence(sequence_id, current_user, db)

    if body.name is not None:
        seq.name = body.name
    if body.mode is not None:
        seq.mode = body.mode
    if body.overlap_threshold is not None:
        seq.overlap_threshold = body.overlap_threshold
    if body.steps is not None:
        seq.steps = [step.model_dump() for step in body.steps]
    if body.ref_image_width is not None:
        seq.ref_image_width = body.ref_image_width
    if body.ref_image_height is not None:
        seq.ref_image_height = body.ref_image_height

    await db.commit()
    await db.refresh(seq)
    return seq


@router.delete("/{sequence_id}", status_code=204)
async def delete_sequence(
    sequence_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    seq = await get_owned_sequence(sequence_id, current_user, db)
    await db.delete(seq)
    await db.commit()


@router.post("/{sequence_id}/run/{video_id}", response_model=SequenceRunResponse)
async def run_sequence(
    sequence_id: str,
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Kick off a Celery task that runs this sequence against a video.
    Returns the SequenceRun row immediately (status 'pending'); poll
    GET /sequences/runs/{run_id} to watch progress."""
    seq = await get_owned_sequence(sequence_id, current_user, db)
    await get_owned_video(video_id, current_user, db)

    run = SequenceRun(
        sequence_id=seq.id,
        video_id=video_id,
        total_steps=len(seq.steps),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    from ..tasks.sequence_run import run_sequence_on_video
    task = run_sequence_on_video.delay(run.id)
    run.task_id = task.id
    await db.commit()
    await db.refresh(run)
    return run


@router.get("/runs/{run_id}", response_model=SequenceRunResponse)
async def get_sequence_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll the status/results of a sequence run."""
    return await get_owned_sequence_run(run_id, current_user, db)


@router.get("/{sequence_id}/runs", response_model=List[SequenceRunResponse])
async def list_sequence_runs(
    sequence_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all runs of a sequence, most recent first."""
    await get_owned_sequence(sequence_id, current_user, db)
    result = await db.execute(
        select(SequenceRun)
        .where(SequenceRun.sequence_id == sequence_id)
        .order_by(SequenceRun.created_at.desc())
    )
    return result.scalars().all()


@router.post("/test-image/{project_id}", response_model=SequenceTestImageResponse)
async def test_sequence_on_image(
    project_id: str,
    body: SequenceTestImageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Quick single-image check: run the project's trained model(s) once
    against one existing image and report, per step, whether that step's
    region+class(es) would currently match — without saving a sequence or
    running a full video job. Line regions can't be tested this way (no
    motion in a single image) and are reported as not testable."""
    await get_owned_project(project_id, current_user, db)
    image = await get_owned_image(body.image_id, current_user, db)

    import cv2
    from ..tasks.auto_annotate import _resolve_image_path
    from ..services.sequence_test import collect_detections, evaluate_step

    real_path = _resolve_image_path(image.filepath)
    if real_path is None:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    img_bgr = cv2.imread(str(real_path))
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    detections = collect_detections(project_id, img_bgr)
    if not detections and not any(
        (settings.model_dir.resolve() / project_id / name).exists()
        for name in ("main_best.pt", "seed_best.pt", "seg_main_best.pt", "seg_seed_best.pt", "seg_best.pt")
    ):
        raise HTTPException(status_code=404, detail="No trained model found for this project yet.")

    results = [
        evaluate_step(step.model_dump(), detections, body.overlap_threshold)
        for step in body.steps
    ]
    return {"image_id": body.image_id, "results": results}
