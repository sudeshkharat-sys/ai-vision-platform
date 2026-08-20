"""
sequences.py
~~~~~~~~~~~~
CRUD API for region sequences — ordered sets of regions (box/line) that a
tracked object must pass through, in order, to complete a "sequence"
(e.g. a bulb visiting socket 1 -> 2 -> 3, or a finger visiting key-zones
S -> U -> D -> E -> S -> H).

This is the builder/storage layer only. Running a sequence against a live
or uploaded video (object tracking + geometry + step matching) is a
separate, not-yet-implemented pipeline — see the design doc.
"""

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database import get_db
from ..models.sequence import RegionSequence
from ..models.user import User
from ..schemas.base import RegionSequenceCreate, RegionSequenceUpdate, RegionSequenceResponse
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_sequence

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
        steps=[step.model_dump() for step in body.steps],
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
    if body.steps is not None:
        seq.steps = [step.model_dump() for step in body.steps]

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
