"""
videos.py
~~~~~~~~~
API endpoints for video upload and frame extraction.

Workflow
--------
1. POST /videos/upload/{project_id}   — save video file, insert Video row
2. GET  /videos/project/{project_id}  — list all videos in a project
3. GET  /videos/{video_id}            — single video status/metadata
4. POST /videos/{video_id}/extract-frames — kick off the Celery frame-extraction task
5. POST /videos/{video_id}/stop-extraction — cancel a running extraction
6. DELETE /videos/{video_id}          — remove video + extracted frame Image rows
7. GET  /videos/{video_id}/frame      — preview one frame at a timestamp (not saved)
8. POST /videos/{video_id}/capture-frame — save one frame at a timestamp as an Image
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from ..database import get_db
from ..models.video import Video
from ..models.image import Image
from ..models.user import User
from ..schemas.base import VideoResponse, VideoFrameExtractionRequest, VideoRotateRequest, ImageResponse, VideoFrameCaptureRequest
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_video

router = APIRouter(prefix="/videos", tags=["videos"])


def _read_frame_at(video: Video, t: float):
    """Open the video file, seek to timestamp t (seconds) and decode that
    frame. Returns (frame_bgr, width, height) or None if the file/frame
    can't be read."""
    from ..tasks.video_processing import _resolve_video_path
    import cv2

    path = _resolve_video_path(video.filepath)
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or video.fps or 25.0
    frame_no = max(0, int(round(t * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    return frame, w, h

# Allowed video MIME types / extensions
_ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}


@router.post("/upload/{project_id}", response_model=VideoResponse)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a single video file and create a Video record."""
    await get_owned_project(project_id, current_user, db)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {sorted(_ALLOWED_VIDEO_EXTS)}",
        )

    video_dir = settings.upload_dir / project_id / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = video_dir / unique_filename

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    file_size = file_path.stat().st_size

    db_video = Video(
        project_id=project_id,
        original_filename=file.filename or unique_filename,
        filepath=f"/uploads/{project_id}/videos/{unique_filename}",
        file_size=file_size,
        status="uploaded",
    )
    db.add(db_video)
    await db.commit()
    await db.refresh(db_video)
    return db_video


@router.get("/project/{project_id}", response_model=List[VideoResponse])
async def list_project_videos(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all videos belonging to a project."""
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(
        select(Video)
        .where(Video.project_id == project_id)
        .order_by(Video.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single video by ID (useful for polling extraction status)."""
    return await get_owned_video(video_id, current_user, db)


@router.get("/{video_id}/frame")
async def preview_video_frame(
    video_id: str,
    t: float = 0.0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Decode and return a single frame at timestamp t (seconds) as a JPEG,
    without saving anything — for scrubbing a video to find a frame to use
    as a reference (e.g. drawing Sequence Detection regions) without first
    running full frame extraction into the training image pool."""
    video = await get_owned_video(video_id, current_user, db)
    result = _read_frame_at(video, t)
    if result is None:
        raise HTTPException(status_code=404, detail="Could not read a frame at that timestamp.")
    frame, _, _ = result

    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/{video_id}/capture-frame", response_model=ImageResponse)
async def capture_video_frame(
    video_id: str,
    body: VideoFrameCaptureRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save one frame at timestamp t as a normal Image row (status
    'pending', same as any uploaded photo) — so any frame of an uploaded
    video can be used as a reference frame (e.g. for Sequence Detection)
    on demand, without running bulk frame extraction or requiring it to
    already be part of the training/annotation set."""
    video = await get_owned_video(video_id, current_user, db)
    result = _read_frame_at(video, body.t)
    if result is None:
        raise HTTPException(status_code=404, detail="Could not read a frame at that timestamp.")
    frame, w, h = result

    import cv2
    frames_dir = settings.upload_dir / video.project_id / "video_frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_uuid = str(uuid.uuid4())
    frame_path = frames_dir / f"{frame_uuid}.jpg"
    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    rel_path = f"/uploads/{video.project_id}/video_frames/{video_id}/{frame_uuid}.jpg"
    stem = Path(video.original_filename).stem
    display_name = f"{stem}_t{body.t:.2f}s.jpg"

    db_image = Image(
        id=frame_uuid, project_id=video.project_id,
        filename=display_name, filepath=rel_path,
        width=w, height=h, status="pending",
    )
    db.add(db_image)
    await db.commit()
    await db.refresh(db_image)
    return db_image


@router.post("/{video_id}/extract-frames", response_model=VideoResponse)
async def extract_frames(
    video_id: str,
    body: VideoFrameExtractionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Kick off a Celery task to extract frames from the video.
    Returns the video record immediately (status will be 'extracting').
    The frontend should poll GET /videos/{video_id} to watch progress.
    """
    video = await get_owned_video(video_id, current_user, db)

    if video.status == "extracting":
        raise HTTPException(status_code=409, detail="Extraction already in progress")

    from ..tasks.video_processing import extract_video_frames

    task = extract_video_frames.delay(
        video_id,
        sample_every_n=body.sample_every_n,
        max_frames=body.max_frames,
    )

    video.status = "extracting"
    video.frames_extracted = 0
    video.task_id = task.id
    await db.commit()
    await db.refresh(video)
    return video


@router.post("/{video_id}/rotate", response_model=VideoResponse)
async def rotate_video_endpoint(
    video_id: str,
    body: VideoRotateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Kick off a Celery task that physically rotates the video file
    (90°/180°), re-encoding it and updating width/height. Poll
    GET /videos/{video_id} for status ('rotating' -> 'uploaded'/'failed')."""
    video = await get_owned_video(video_id, current_user, db)

    if video.status in ("extracting", "rotating"):
        raise HTTPException(status_code=409, detail="Video is busy — wait for the current operation to finish")
    if body.direction not in ("cw", "ccw", "180"):
        raise HTTPException(status_code=400, detail="direction must be 'cw', 'ccw', or '180'")

    from ..tasks.video_processing import rotate_video

    task = rotate_video.delay(video_id, direction=body.direction)

    video.status = "rotating"
    video.task_id = task.id
    await db.commit()
    await db.refresh(video)
    return video


@router.post("/{video_id}/stop-extraction", response_model=VideoResponse)
async def stop_extraction(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running frame-extraction task."""
    video = await get_owned_video(video_id, current_user, db)

    if video.status != "extracting":
        raise HTTPException(status_code=409, detail="No extraction in progress")

    if video.task_id:
        from ..tasks.celery_app import celery_app
        celery_app.control.revoke(video.task_id, terminate=True, signal="SIGTERM")

    video.status = "stopped"
    video.task_id = None
    await db.commit()
    await db.refresh(video)
    return video


@router.delete("/{video_id}", status_code=204)
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a video and all Image rows that were extracted from it."""
    video = await get_owned_video(video_id, current_user, db)

    # Delete extracted frame Image rows (identified by filepath pattern)
    frame_path_prefix = f"/uploads/{video.project_id}/video_frames/{video_id}/"
    await db.execute(
        delete(Image).where(Image.filepath.like(f"{frame_path_prefix}%"))
    )

    # Remove the video file from disk
    rel = video.filepath.lstrip("/")
    for anchor in [
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]:
        try:
            if anchor.exists():
                anchor.unlink()
                break
        except Exception:
            pass

    # Remove extracted frames directory
    frames_dir = settings.upload_dir / video.project_id / "video_frames" / video_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)

    await db.delete(video)
    await db.commit()
