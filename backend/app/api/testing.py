"""
testing.py
~~~~~~~~~~
Testing panel: try a trained model on new images / videos, or score it
against the project's own labelled (training) data, then send the images it
fails on to a fresh project for another round of training.

  GET  /testing/{pid}/sources                  models that can be tested
  POST /testing/{pid}/model                    upload a detector / segmentation .pt
  POST /testing/{pid}/image                    run on one uploaded image
  POST /testing/{pid}/video                    run on an uploaded video (background)
  GET  /testing/{pid}/video/status|output|flagged/{name}
  POST /testing/{pid}/training-data/run        score the model on the labelled images (background)
  GET  /testing/{pid}/training-data/status|results
  POST /testing/{pid}/training-data/stop
  POST /testing/{pid}/training-data/send-failed  copy failed images (+ labels) to a project

Sources: detector (project Main/Seed), segmentation, classifier (crop+classify
pipeline) or uploaded (test_model.pt). Runs happen in this process (the
backend runs on the user's PC); one job per kind per project.
"""

import base64
import json
import shutil
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
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
from ..services.cls_model import _load_path, load_cls_meta, predict_crop_cls
from ..services.data_segments import assign_segments, normalize_segments
from ..services.det_model import det_model_uses_preprocess, resolve_det_model_path
from ..services.image_files import resolve_image_file
from ..services.seg_model import resolve_seg_model_path, seg_model_uses_preprocess
from ..services.test_eval import score_classification, score_detections, summarize

router = APIRouter(prefix="/testing", tags=["testing"])

SOURCES = ("detector", "segmentation", "classifier", "uploaded")
_jobs: dict = {}            # (project_id, "video" | "data") -> status dict
_job_lock = threading.Lock()


def _pdir(pid: str) -> Path:
    return settings.model_dir.resolve() / pid


def _out_dir(pid: str) -> Path:
    d = _pdir(pid) / "test_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _uploaded_path(pid: str) -> Path:
    return _pdir(pid) / "test_model.pt"


def _uploaded_meta(pid: str) -> Optional[dict]:
    p = _pdir(pid) / "test_model.json"
    try:
        return json.loads(p.read_text()) if p.exists() and _uploaded_path(pid).exists() else None
    except Exception:  # noqa: BLE001
        return None


def _resolve(pid: str, source: str, preprocess: Optional[bool] = None):
    """(weights path | None, preprocess flag) for a box-producing source."""
    if source == "detector":
        p = resolve_det_model_path(pid)
        return p, (det_model_uses_preprocess(p) if p else False)
    if source == "segmentation":
        p = resolve_seg_model_path(pid)
        return p, (seg_model_uses_preprocess(p) if p else False)
    if source == "uploaded":
        p = _uploaded_path(pid)
        return (p if p.exists() else None), bool(preprocess)
    raise HTTPException(status_code=400, detail=f"Unknown source '{source}'")


def _preprocess(img, on: bool):
    if not on:
        return img
    from ..tasks.training import clahe_gamma_sharpen  # lazy -- heavy import
    return clahe_gamma_sharpen(img)


def _predict_boxes(model, img, conf: float, preprocess: bool) -> list:
    res = model.predict(_preprocess(img, preprocess), conf=conf, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    for xywhn, c, k in zip(res.boxes.xywhn.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        out.append({"class_name": res.names[int(k)], "bbox": [round(v, 5) for v in xywhn], "conf": round(float(c), 4)})
    return out


def _draw(img, boxes, color=(0, 200, 0), label_fn=None):
    h, w = img.shape[:2]
    th = max(2, w // 400)
    for b in boxes:
        xc, yc, bw, bh = b["bbox"]
        p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        cv2.rectangle(img, p1, p2, color, th)
        label = label_fn(b) if label_fn else f"{b['class_name']} {b.get('conf', 0):.2f}"
        cv2.putText(img, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, w / 1600), color, max(1, th - 1))
    return img


def _draw_cls(img, r):
    label = r.get("state") or r["status"]
    if r.get("confidence") is not None:
        label = f"{label} {r['confidence']:.2f}"
    ok = r["status"] == "ok"
    color = (0, 200, 0) if ok else (0, 165, 255)
    if r.get("bbox"):
        _draw(img, [{"class_name": "", "bbox": r["bbox"]}], color, lambda b: label)
    else:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


def _jpeg_b64(img) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def _get_model(path: Path):
    return _load_path(path)


# ── sources / upload ────────────────────────────────────────────────

@router.get("/{project_id}/sources")
async def sources(project_id: str, current_user: User = Depends(get_current_user),
                  db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    det, seg = resolve_det_model_path(project_id), resolve_seg_model_path(project_id)
    cls_ready = load_cls_meta(project_id) is not None
    return {
        "detector": {"available": det is not None, "file": det.name if det else None},
        "segmentation": {"available": seg is not None, "file": seg.name if seg else None},
        "classifier": {"available": cls_ready,
                       "detail": (load_cls_meta(project_id) or {}).get("mode")},
        "uploaded": {"available": _uploaded_meta(project_id) is not None,
                     "meta": _uploaded_meta(project_id)},
    }


@router.post("/{project_id}/model")
async def upload_model(project_id: str, file: UploadFile = File(...),
                       current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    if not (file.filename or "").lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Upload a YOLO .pt file")
    d = _pdir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "test_model.upload"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        from ultralytics import YOLO
        m = YOLO(str(tmp))
        task = getattr(m, "task", "detect")
        names = list(m.names.values())
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Not a valid YOLO model: {e}")
    if task not in ("detect", "segment"):
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Only detection / segmentation models can be uploaded here (this one is '{task}')")
    tmp.replace(_uploaded_path(project_id))
    meta = {"filename": file.filename, "task": task, "classes": names}
    (d / "test_model.json").write_text(json.dumps(meta))
    return meta


# ── single image ────────────────────────────────────────────────────

@router.post("/{project_id}/image")
async def test_image(project_id: str, source: str = "detector", conf: float = 0.25,
                     preprocess: bool = False, cls_threshold: float = 0.6,
                     file: UploadFile = File(...),
                     current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    def work():
        if source == "classifier":
            r = predict_crop_cls(project_id, img, conf, cls_threshold)
            if r["status"] == "no_model":
                raise HTTPException(status_code=404, detail="No trained crop+classify model for this project")
            return {"kind": "classifier", **r, "preview": _jpeg_b64(_draw_cls(img.copy(), r))}
        path, pre = _resolve(project_id, source, preprocess)
        if path is None:
            raise HTTPException(status_code=404, detail=f"No {source} model for this project")
        dets = _predict_boxes(_get_model(path), img, conf, pre)
        return {"kind": "boxes", "detections": dets, "preview": _jpeg_b64(_draw(img.copy(), dets))}

    return await run_in_threadpool(work)


# ── video ───────────────────────────────────────────────────────────

class VideoOpts(BaseModel):
    source: str = "detector"
    conf: float = 0.25
    preprocess: bool = False
    cls_threshold: float = 0.6
    fps: float = 5.0            # frames per second analysed
    flag_below: float = 0.4     # a frame is flagged when nothing is found / best confidence is below this


def _write_status(key, **kw):
    with _job_lock:
        _jobs.setdefault(key, {}).update(kw)


@router.post("/{project_id}/video")
async def test_video(project_id: str, source: str = "detector", conf: float = 0.25,
                     preprocess: bool = False, cls_threshold: float = 0.6, fps: float = 5.0,
                     flag_below: float = 0.4, file: UploadFile = File(...),
                     current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    key = (project_id, "video")
    if _jobs.get(key, {}).get("state") == "running":
        raise HTTPException(status_code=409, detail="A video test is already running for this project")
    if source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"Unknown source '{source}'")
    if source == "classifier":
        if load_cls_meta(project_id) is None:
            raise HTTPException(status_code=404, detail="No trained crop+classify model for this project")
        path, pre = None, False
    else:
        path, pre = _resolve(project_id, source, preprocess)
        if path is None:
            raise HTTPException(status_code=404, detail=f"No {source} model for this project")

    out = _out_dir(project_id)
    for old in out.glob("flag_*.jpg"):
        old.unlink(missing_ok=True)
    src_video = out / f"input{Path(file.filename or 'v.mp4').suffix or '.mp4'}"
    with open(src_video, "wb") as f:
        shutil.copyfileobj(file.file, f)
    _jobs[key] = {"state": "running", "stop": False, "frames": 0, "total": 0, "flagged": [],
                  "no_detection": 0, "started": time.time(), "source": source,
                  "filename": file.filename}
    threading.Thread(target=_video_worker, daemon=True, args=(
        project_id, key, src_video, path, pre, VideoOpts(source=source, conf=conf, preprocess=preprocess,
                                                          cls_threshold=cls_threshold, fps=fps,
                                                          flag_below=flag_below))).start()
    return {"started": True}


def _video_worker(pid, key, src_video, path, pre, o: VideoOpts):
    out = _out_dir(pid)
    writer = None
    try:
        cap = cv2.VideoCapture(str(src_video))
        if not cap.isOpened():
            raise RuntimeError("Could not open the video")
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(vfps / max(o.fps, 0.1))))
        _write_status(key, total=total // step if total else 0)
        model = _get_model(path) if path else None
        idx = seen = 0
        while True:
            if _jobs[key].get("stop"):
                break
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            flagged_reason = None
            if o.source == "classifier":
                r = predict_crop_cls(pid, frame, o.conf, o.cls_threshold)
                view = _draw_cls(frame.copy(), r)
                if r["status"] != "ok":
                    flagged_reason = r["status"]
            else:
                dets = _predict_boxes(model, frame, o.conf, pre)
                view = _draw(frame.copy(), dets)
                if not dets:
                    flagged_reason = "no_detection"
                    _write_status(key, no_detection=_jobs[key]["no_detection"] + 1)
                elif min(d["conf"] for d in dets) < o.flag_below:
                    flagged_reason = "low_confidence"
            if writer is None:
                h, w = view.shape[:2]
                writer = cv2.VideoWriter(str(out / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, vfps / step), (w, h))
            writer.write(view)
            if flagged_reason and len(_jobs[key]["flagged"]) < 300:
                name = f"flag_{idx:06d}.jpg"
                cv2.imwrite(str(out / name), frame)          # the raw frame, ready for re-labelling
                _jobs[key]["flagged"].append({"name": name, "frame": idx, "reason": flagged_reason,
                                              "time_s": round(idx / vfps, 2)})
            seen += 1
            _write_status(key, frames=seen)
            idx += 1
        cap.release()
        _write_status(key, state="stopped" if _jobs[key].get("stop") else "done",
                      has_output=writer is not None, finished=time.time())
    except Exception as e:  # noqa: BLE001
        _write_status(key, state="failed", message=str(e))
    finally:
        if writer is not None:
            writer.release()


@router.get("/{project_id}/video/status")
async def video_status(project_id: str, current_user: User = Depends(get_current_user),
                       db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    return _jobs.get((project_id, "video")) or {"state": "idle"}


@router.post("/{project_id}/video/stop")
async def video_stop(project_id: str, current_user: User = Depends(get_current_user),
                     db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    _write_status((project_id, "video"), stop=True)
    return {"stopping": True}


@router.get("/{project_id}/video/output")
async def video_output(project_id: str, current_user: User = Depends(get_current_user),
                       db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    p = _out_dir(project_id) / "output.mp4"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No output video yet")
    return FileResponse(str(p), media_type="video/mp4", filename="tested_video.mp4")


@router.get("/{project_id}/video/flagged/{name}")
async def video_flagged(project_id: str, name: str, current_user: User = Depends(get_current_user),
                        db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    if "/" in name or "\\" in name or not name.startswith("flag_"):
        raise HTTPException(status_code=400, detail="Bad name")
    p = _out_dir(project_id) / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(p), media_type="image/jpeg")


# ── test on the project's own labelled data ─────────────────────────

class DataRunBody(BaseModel):
    source: str = "detector"
    conf: float = 0.25
    iou: float = 0.5
    allow_extra: bool = False       # False = a false extra box also fails the image
    preprocess: bool = False        # only for an uploaded model
    cls_threshold: float = 0.6
    limit: Optional[int] = None     # test only the first N images (quick check)


def _data_results_file(pid):
    return _pdir(pid) / "test_data_results.json"


@router.post("/{project_id}/training-data/run")
async def run_data_test(project_id: str, body: DataRunBody, current_user: User = Depends(get_current_user),
                        db: AsyncSession = Depends(get_db)):
    project = await get_owned_project(project_id, current_user, db)
    key = (project_id, "data")
    if _jobs.get(key, {}).get("state") == "running":
        raise HTTPException(status_code=409, detail="A test is already running for this project")
    if body.source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"Unknown source '{body.source}'")

    path, pre = (None, False)
    if body.source != "classifier":
        path, pre = _resolve(project_id, body.source, body.preprocess)
        if path is None:
            raise HTTPException(status_code=404, detail=f"No {body.source} model for this project")
    elif load_cls_meta(project_id) is None:
        raise HTTPException(status_code=404, detail="No trained crop+classify model for this project")

    imgs = (await db.execute(select(Image).where(Image.project_id == project_id).order_by(Image.created_at))).scalars().all()
    anns = defaultdict(list)
    if imgs:
        for a in (await db.execute(select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
            anns[a.image_id].append({"class_name": a.class_name, "bbox": a.bbox})

    labels = {}
    if body.source == "classifier":
        segs = normalize_segments(project.segments or [])
        if not segs:
            raise HTTPException(status_code=400, detail="The classifier is tested against Data Segments - define them first (Training Hub -> Data Segments)")
        assignment, _ = assign_segments(anns, segs, image_ids=[i.id for i in imgs])
        labels = {iid: s["label"] for iid, s in assignment.items()}
        imgs = [i for i in imgs if i.id in labels]
    else:
        imgs = [i for i in imgs if any(a["bbox"] for a in anns.get(i.id, []))]
    if body.limit:
        imgs = imgs[: body.limit]
    if not imgs:
        raise HTTPException(status_code=400, detail="No labelled images to test on")

    items = [{"id": i.id, "filename": i.filename, "filepath": i.filepath, "width": i.width, "height": i.height,
              "gts": [a for a in anns.get(i.id, []) if a["bbox"]], "label": labels.get(i.id)} for i in imgs]
    _jobs[key] = {"state": "running", "stop": False, "done": 0, "total": len(items), "failed": 0,
                  "source": body.source, "started": time.time()}
    _data_results_file(project_id).unlink(missing_ok=True)
    threading.Thread(target=_data_worker, daemon=True, args=(project_id, key, items, path, pre, body)).start()
    return {"started": True, "total": len(items)}


def _data_worker(pid, key, items, path, pre, o: DataRunBody):
    rows = []
    try:
        model = _get_model(path) if path else None
        for it in items:
            if _jobs[key].get("stop"):
                break
            f = resolve_image_file(pid, it["filepath"])
            img = cv2.imread(str(f)) if f else None
            row = {"id": it["id"], "filename": it["filename"], "width": it["width"], "height": it["height"],
                   "filepath": it["filepath"]}
            if img is None:
                row.update(passed=False, error="file missing or unreadable", hit=0, missed=len(it["gts"]),
                           wrong_class=0, extra=0, problems=[], preds=[], gts=it["gts"])
            elif o.source == "classifier":
                r = predict_crop_cls(pid, img, o.conf, o.cls_threshold)
                sc = score_classification(it["label"], r)
                row.update(sc, bbox=r.get("bbox"), hit=int(sc["passed"]), missed=0, wrong_class=0, extra=0,
                           problems=[], preds=[], gts=[])
            else:
                preds = _predict_boxes(model, img, o.conf, pre)
                sc = score_detections(it["gts"], preds, o.iou, o.allow_extra)
                row.update(sc, preds=preds, gts=it["gts"])
            rows.append(row)
            _write_status(key, done=len(rows), failed=sum(1 for r in rows if not r["passed"]))
        stopped = _jobs[key].get("stop")
        summary = summarize(rows)
        _data_results_file(pid).write_text(json.dumps({
            "source": o.source, "options": o.model_dump(), "summary": summary, "rows": rows,
            "finished": time.time(), "stopped": bool(stopped)}))
        _write_status(key, state="stopped" if stopped else "done", summary=summary, finished=time.time())
    except Exception as e:  # noqa: BLE001
        _write_status(key, state="failed", message=str(e))


@router.get("/{project_id}/training-data/status")
async def data_status(project_id: str, current_user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    return _jobs.get((project_id, "data")) or {"state": "idle"}


@router.post("/{project_id}/training-data/stop")
async def data_stop(project_id: str, current_user: User = Depends(get_current_user),
                    db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    _write_status((project_id, "data"), stop=True)
    return {"stopping": True}


@router.get("/{project_id}/training-data/results")
async def data_results(project_id: str, only_failed: bool = False,
                       current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await get_owned_project(project_id, current_user, db)
    f = _data_results_file(project_id)
    if not f.exists():
        raise HTTPException(status_code=404, detail="No finished test yet")
    data = json.loads(f.read_text())
    if only_failed:
        data["rows"] = [r for r in data["rows"] if not r["passed"]]
    return data


class SendFailedBody(BaseModel):
    target_project_id: Optional[str] = None
    new_name: Optional[str] = None
    image_ids: Optional[list] = None        # default: every failed image of the last test
    include_model: bool = True              # copy the tested model so training can start from its weights


@router.post("/{project_id}/training-data/send-failed")
async def send_failed(project_id: str, body: SendFailedBody, current_user: User = Depends(get_current_user),
                      db: AsyncSession = Depends(get_db)):
    """Copy the failed images with all their labels into a project (new or
    existing) so you can fix labels / add data and train again."""
    source = await get_owned_project(project_id, current_user, db)
    f = _data_results_file(project_id)
    if not f.exists():
        raise HTTPException(status_code=404, detail="No finished test yet")
    data = json.loads(f.read_text())
    failed = {r["id"] for r in data["rows"] if not r["passed"]}
    if body.image_ids is not None:
        failed &= set(body.image_ids)
    if not failed:
        raise HTTPException(status_code=400, detail="No failed images to send")

    if body.target_project_id:
        target = await get_owned_project(body.target_project_id, current_user, db)
        if target.id == source.id:
            raise HTTPException(status_code=422, detail="Pick a different project")
    elif (body.new_name or "").strip():
        target = Project(name=body.new_name.strip()[:250], description=source.description, classes=[],
                         project_type=source.project_type, segments=list(source.segments or []),
                         user_id=current_user.id)
        db.add(target)
        await db.flush()
    else:
        raise HTTPException(status_code=422, detail="Choose a target project or a new project name")

    imgs = (await db.execute(select(Image).where(Image.project_id == project_id, Image.id.in_(failed)))).scalars().all()
    anns = defaultdict(list)
    for a in (await db.execute(select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
        anns[a.image_id].append(a)

    dest_dir = settings.upload_dir / target.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    classes, copied, missing = list(target.classes or []), 0, 0
    for img in imgs:
        src = resolve_image_file(project_id, img.filepath)
        if src is None:
            missing += 1
            continue
        parts = img.filepath.replace("\\", "/").split("/")
        sub = Path(*parts[parts.index("video_frames"):-1]) if "video_frames" in parts else Path()
        (dest_dir / sub).mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4()}{src.suffix}"
        shutil.copy2(src, dest_dir / sub / name)
        rel = "/".join(["uploads", target.id, *sub.parts, name])
        new_img = Image(project_id=target.id, filename=img.filename, filepath=f"/{rel}",
                        width=img.width, height=img.height, status=img.status)
        db.add(new_img)
        await db.flush()
        for a in anns.get(img.id, []):
            db.add(Annotation(image_id=new_img.id, class_name=a.class_name,
                              bbox=list(a.bbox) if a.bbox else None, source=a.source,
                              annotation_type=a.annotation_type, points=list(a.points) if a.points else None,
                              state=a.state))
            if a.class_name not in classes:
                classes.append(a.class_name)
        copied += 1
    target.classes = classes
    await db.commit()

    weights_copied = None
    if body.include_model and data.get("source") in ("detector", "segmentation", "uploaded"):
        try:
            wp, _ = _resolve(project_id, data["source"])
            if wp is not None:
                wd = settings.model_dir.resolve() / target.id / "custom_weights"
                wd.mkdir(parents=True, exist_ok=True)
                weights_copied = f"tested_{data['source']}_{wp.name}"
                shutil.copy2(wp, wd / weights_copied)
        except Exception:  # noqa: BLE001  (the images are the important part)
            weights_copied = None
    return {"target_project_id": target.id, "target_name": target.name, "copied": copied,
            "missing_files": missing, "weights": weights_copied}
