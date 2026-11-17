"""
rectify.py
~~~~~~~~~~
Testing panel > Rectify tab. Finds auto-labelled boxes whose interior is
unusually dark (torch light missed) so a person can re-check them first.
Self-contained: reads annotations, only writes when the user clicks a button,
and keeps an undo log. Does not touch training / auto-annotate code.

  GET  /rectify/{pid}/states            states by source (to pick what to scan)
  POST /rectify/{pid}/scan              start a background darkness scan
  GET  /rectify/{pid}/status            progress
  GET  /rectify/{pid}/results           ranked boxes + normal range
  GET  /rectify/{pid}/crop/{ann_id}     JPEG of the box (enhance=1 brightens)
  POST /rectify/{pid}/relabel           change a box's state (marks it manual)
  POST /rectify/{pid}/confirm           keep label, mark reviewed (manual)
  POST /rectify/{pid}/undo              restore the previous label/source
"""
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..config import settings
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User

router = APIRouter(prefix="/rectify", tags=["rectify"])

_JOBS: dict = {}          # project_id -> job dict
_LOCK = threading.Lock()


def _state_of(a) -> str:
    return (a.state or "").strip() or a.class_name


def _image_path(filepath: str) -> Optional[Path]:
    rel = filepath.lstrip("/")
    for p in (Path(filepath), settings.upload_dir.parent / rel,
              settings.upload_dir / filepath.replace("/uploads/", "", 1)):
        try:
            if p.exists():
                return p
        except OSError:
            pass
    return None


def _box_px(img, bbox):
    h, w = img.shape[:2]
    xc, yc, bw, bh = bbox
    x1, y1 = max(int((xc - bw / 2) * w), 0), max(int((yc - bh / 2) * h), 0)
    x2, y2 = min(int((xc + bw / 2) * w), w), min(int((yc + bh / 2) * h), h)
    return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


def _scan(pid: str, rows: list, job: dict):
    """rows: dicts with ann_id, image_id, filename, filepath, state, bbox."""
    res = []
    for i, r in enumerate(rows):
        if job.get("stop"):
            break
        job["done"] = i
        p = _image_path(r["filepath"])
        img = cv2.imread(str(p)) if p else None
        box = _box_px(img, r["bbox"]) if img is not None else None
        if box is None:
            continue
        x1, y1, x2, y2 = box
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
        inside = v[y1:y2, x1:x2]
        total = float(v.sum()); area = v.size
        out_n = max(area - inside.size, 1)
        mi = float(inside.mean())
        mo = (total - float(inside.sum())) / out_n
        res.append({
            "ann_id": r["ann_id"], "image_id": r["image_id"], "filename": r["filename"],
            "state": r["state"], "inside_mean": round(mi, 1), "outside_mean": round(mo, 1),
            "ratio": round(mi / max(mo, 1.0), 3),
            "dark_pct": round(float((inside < 60).mean() * 100), 1),
            "dark_score": round(255 - mi, 1),
        })
    summary = {}
    if res:
        dk = np.array([d["dark_score"] for d in res])
        med = float(np.median(dk))
        mad = float(np.median(np.abs(dk - med)) * 1.4826) or 1.0
        q1, q3 = np.percentile(dk, [25, 75])
        fence = float(q3 + 1.5 * (q3 - q1))
        for d in res:
            d["z"] = round((d["dark_score"] - med) / mad, 2)
            d["outlier"] = bool(d["dark_score"] > fence or d["z"] > 3.5)
        res.sort(key=lambda d: (not d["outlier"], -d["z"]))
        summary = {"median": round(med, 1), "mad": round(mad, 2), "fence": round(fence, 1),
                   "outliers": sum(d["outlier"] for d in res), "total": len(res)}
    job.update(items=res, summary=summary, done=len(rows), state="done")


@router.get("/{project_id}/states")
async def states(project_id: str, db: AsyncSession = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    rows = (await db.execute(
        select(Annotation).join(Image, Annotation.image_id == Image.id)
        .where(Image.project_id == project_id))).scalars().all()
    c = Counter((_state_of(a), a.source) for a in rows)
    return [{"state": s, "source": src, "count": n} for (s, src), n in sorted(c.items())]


class ScanBody(BaseModel):
    state: Optional[str] = None      # None = every state
    source: str = "auto"             # "auto" (suspects) or "all"


@router.post("/{project_id}/scan")
async def scan(project_id: str, body: ScanBody, db: AsyncSession = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    with _LOCK:
        if _JOBS.get(project_id, {}).get("state") == "running":
            raise HTTPException(409, "A scan is already running")
    q = select(Annotation, Image).join(Image, Annotation.image_id == Image.id) \
        .where(Image.project_id == project_id)
    if body.source == "auto":
        q = q.where(Annotation.source == "auto")
    rows = []
    for a, im in (await db.execute(q)).all():
        bbox = a.bbox
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        if not bbox or len(bbox) != 4:
            continue
        if body.state and _state_of(a) != body.state:
            continue
        rows.append({"ann_id": a.id, "image_id": im.id, "filename": im.filename,
                     "filepath": im.filepath, "state": _state_of(a), "bbox": bbox})
    job = {"state": "running", "done": 0, "total": len(rows), "items": [], "summary": {}}
    with _LOCK:
        _JOBS[project_id] = job
    threading.Thread(target=_scan, args=(project_id, rows, job), daemon=True).start()
    return {"started": True, "total": len(rows)}


@router.get("/{project_id}/status")
async def status(project_id: str, db: AsyncSession = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    j = _JOBS.get(project_id)
    if not j:
        return {"state": "idle"}
    return {"state": j["state"], "done": j["done"], "total": j["total"]}


@router.get("/{project_id}/results")
async def results(project_id: str, db: AsyncSession = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    j = _JOBS.get(project_id)
    if not j or j["state"] != "done":
        return {"summary": {}, "items": []}
    return {"summary": j["summary"], "items": j["items"]}


async def _owned_ann(project_id, ann_id, db):
    row = (await db.execute(
        select(Annotation, Image).join(Image, Annotation.image_id == Image.id)
        .where(Annotation.id == ann_id, Image.project_id == project_id))).first()
    if not row:
        raise HTTPException(404, "Annotation not found")
    return row


@router.get("/{project_id}/crop/{ann_id}")
async def crop(project_id: str, ann_id: str, enhance: int = 1,
               db: AsyncSession = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    a, im = await _owned_ann(project_id, ann_id, db)
    bbox = json.loads(a.bbox) if isinstance(a.bbox, str) else a.bbox

    def work():
        p = _image_path(im.filepath)
        img = cv2.imread(str(p)) if p else None
        if img is None:
            return None
        box = _box_px(img, bbox)
        if not box:
            return None
        x1, y1, x2, y2 = box
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        c = img[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
        if enhance:
            lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(3.0, (4, 4)).apply(lab[:, :, 0])
            c = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        scale = 360 / max(c.shape[:2])
        c = cv2.resize(c, None, fx=scale, fy=scale)
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    data = await run_in_threadpool(work)
    if not data:
        raise HTTPException(404, "Image unreadable")
    return Response(content=data, media_type="image/jpeg")


# ── edits (with undo log) ───────────────────────────────────────────────────
def _log_path(project_id: str) -> Path:
    d = settings.upload_dir / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "rectify_log.json"


def _log_read(project_id: str) -> dict:
    try:
        return json.loads(_log_path(project_id).read_text())
    except Exception:
        return {}


def _log_prev(project_id: str, a: Annotation):
    log = _log_read(project_id)
    # keep the FIRST original so repeated edits still undo to the true original
    log.setdefault(a.id, {"class_name": a.class_name, "state": a.state, "source": a.source})
    _log_path(project_id).write_text(json.dumps(log))


class RelabelBody(BaseModel):
    ann_id: str
    new_state: str


class AnnBody(BaseModel):
    ann_id: str


@router.post("/{project_id}/relabel")
async def relabel(project_id: str, body: RelabelBody, db: AsyncSession = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    from .annotations import _ensure_class_registered
    await get_owned_project(project_id, current_user, db)
    a, _ = await _owned_ann(project_id, body.ann_id, db)
    new = body.new_state.strip()
    if not new:
        raise HTTPException(400, "new_state required")
    _log_prev(project_id, a)
    if (a.state or "").strip():
        a.state = new
    else:
        a.class_name = new
        await _ensure_class_registered(db, project_id, new)
    a.source = "manual"
    await db.commit()
    return {"ok": True}


@router.post("/{project_id}/confirm")
async def confirm(project_id: str, body: AnnBody, db: AsyncSession = Depends(get_db),
                  current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    a, _ = await _owned_ann(project_id, body.ann_id, db)
    _log_prev(project_id, a)
    a.source = "manual"
    await db.commit()
    return {"ok": True}


@router.post("/{project_id}/undo")
async def undo(project_id: str, body: AnnBody, db: AsyncSession = Depends(get_db),
               current_user: User = Depends(get_current_user)):
    await get_owned_project(project_id, current_user, db)
    a, _ = await _owned_ann(project_id, body.ann_id, db)
    prev = _log_read(project_id).get(body.ann_id)
    if not prev:
        raise HTTPException(404, "Nothing to undo")
    a.class_name, a.state, a.source = prev["class_name"], prev["state"], prev["source"]
    await db.commit()
    return {"ok": True}
