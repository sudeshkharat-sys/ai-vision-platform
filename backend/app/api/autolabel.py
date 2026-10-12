"""
autolabel.py
~~~~~~~~~~~~
Auto-label a project's videos with an uploaded detector .pt on this PC.

  POST   /autolabel/{project_id}/model    upload the detector .pt
  POST   /autolabel/{project_id}/run      start scripts/video_autolabel.py (optional preview window)
  POST   /autolabel/{project_id}/stop     ask the running job to stop
  GET    /autolabel/{project_id}/status   progress written by the script
  POST   /autolabel/{project_id}/send     copy REVIEWED frames to a training project

The job is a local subprocess (backend runs on the user's PC), so it can use
the GPU and open a preview window.
"""

import json
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
from ..models.user import User
from ..services.image_files import resolve_image_file

router = APIRouter(prefix="/autolabel", tags=["autolabel"])

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_SCRIPT = _BACKEND_DIR / "scripts" / "video_autolabel.py"
_procs = {}  # project_id -> Popen


def _dir(project_id: str) -> Path:
    return settings.model_dir.resolve() / project_id


def _running(project_id: str) -> bool:
    p = _procs.get(project_id)
    return p is not None and p.poll() is None


@router.post("/{project_id}/model")
async def upload_model(project_id: str, file: UploadFile = File(...),
                       current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    if not (file.filename or "").lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Upload a YOLO .pt file")
    d = _dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "autolabel.pt", "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        from ultralytics import YOLO
        names = list(YOLO(str(d / "autolabel.pt")).names.values())
    except Exception as e:  # noqa: BLE001
        (d / "autolabel.pt").unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Not a valid YOLO model: {e}")
    (d / "autolabel_model.json").write_text(json.dumps({"filename": file.filename, "classes": names}))
    return {"filename": file.filename, "classes": names}


class RunBody(BaseModel):
    fps: float = 3.0
    conf: float = 0.25
    window: bool = True
    dedupe: bool = True
    video_ids: Optional[List[str]] = None


@router.post("/{project_id}/run")
async def run(project_id: str, body: RunBody, current_user: User = Depends(get_current_user),
              db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    if _running(project_id):
        raise HTTPException(status_code=409, detail="Auto-label is already running for this project")
    if not (_dir(project_id) / "autolabel.pt").exists():
        raise HTTPException(status_code=400, detail="Upload the detector .pt first")
    cmd = [sys.executable, str(_SCRIPT), "--project-id", project_id,
           "--fps", str(body.fps), "--conf", str(body.conf)]
    if body.window:
        cmd.append("--window")
    if not body.dedupe:
        cmd.append("--no-dedupe")
    for v in body.video_ids or []:
        cmd += ["--video-id", v]
    log = open(_dir(project_id) / "autolabel.log", "w")
    _procs[project_id] = subprocess.Popen(cmd, cwd=str(_BACKEND_DIR), stdout=log, stderr=subprocess.STDOUT)
    return {"started": True}


@router.post("/{project_id}/stop")
async def stop(project_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    d = _dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "autolabel_stop").write_text("stop")
    return {"stopping": True}


@router.get("/{project_id}/status")
async def status(project_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    d = _dir(project_id)
    out = {"running": _running(project_id), "model": None, "status": None}
    if (d / "autolabel_model.json").exists() and (d / "autolabel.pt").exists():
        out["model"] = json.loads((d / "autolabel_model.json").read_text())
    if (d / "autolabel_status.json").exists():
        try:
            out["status"] = json.loads((d / "autolabel_status.json").read_text())
        except Exception:  # noqa: BLE001  (read mid-write)
            pass
    p = _procs.get(project_id)
    if p is not None and p.poll() not in (None, 0) and out["status"] and out["status"].get("state") == "running":
        out["status"]["state"] = "failed"
        out["status"]["message"] = "The job crashed - see data/models/<project>/autolabel.log"
    return out


class SendBody(BaseModel):
    target_project_id: Optional[str] = None   # existing project, or
    new_name: Optional[str] = None            # a new project with this name


@router.post("/{project_id}/send")
async def send_reviewed(project_id: str, body: SendBody, current_user: User = Depends(get_current_user),
                        db: AsyncSession = Depends(get_db)):
    """Copy frames that were REVIEWED (annotated, no machine 'auto' box left) with
    all their boxes into a training project; the video id stays in the path."""
    source = await get_owned_project(project_id, current_user, db)
    if body.target_project_id:
        target = await get_owned_project(body.target_project_id, current_user, db)
        if target.id == source.id:
            raise HTTPException(status_code=400, detail="Pick a different project")
    elif (body.new_name or "").strip():
        target = Project(name=body.new_name.strip()[:250], description=source.description,
                         classes=[], project_type=source.project_type, user_id=current_user.id)
        db.add(target)
        await db.flush()
    else:
        raise HTTPException(status_code=422, detail="Choose a target project or a new project name")

    imgs = (await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "annotated",
                            Image.filepath.like("%/video_frames/%"))
    )).scalars().all()
    anns = defaultdict(list)
    if imgs:
        for a in (await db.execute(select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
            anns[a.image_id].append(a)

    copied = skipped_unreviewed = 0
    missing = []
    classes = list(target.classes or [])
    for img in imgs:
        boxes = anns.get(img.id, [])
        if not boxes or any(a.source == "auto" for a in boxes):
            skipped_unreviewed += 1
            continue
        src = resolve_image_file(project_id, img.filepath)
        if src is None:
            missing.append(img.filename)
            continue
        parts = img.filepath.replace("\\", "/").split("/")
        vid = parts[parts.index("video_frames") + 1]
        dest_dir = settings.upload_dir / target.id / "video_frames" / vid
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_file = f"{uuid.uuid4()}{src.suffix}"
        shutil.copy2(src, dest_dir / new_file)
        new_img = Image(project_id=target.id, filename=img.filename,
                        filepath=f"/uploads/{target.id}/video_frames/{vid}/{new_file}",
                        width=img.width, height=img.height, status="annotated")
        db.add(new_img)
        await db.flush()
        for a in boxes:
            db.add(Annotation(image_id=new_img.id, class_name=a.class_name,
                              bbox=list(a.bbox) if a.bbox else None, source="manual",
                              annotation_type=a.annotation_type,
                              points=list(a.points) if a.points else None, state=a.state))
            if a.class_name not in classes:
                classes.append(a.class_name)
        copied += 1
    target.classes = classes
    await db.commit()
    return {"target_project_id": target.id, "target_name": target.name, "copied": copied,
            "skipped_unreviewed": skipped_unreviewed, "missing_files": len(missing)}
