"""
sequence_run.py
~~~~~~~~~~~~~~~~
Celery task that runs a saved RegionSequence against an uploaded video:
detect objects frame-by-frame with the project's trained YOLO model, feed
the detections into the SequenceRunState state machine, and record how far
the sequence got (or whether it completed).

This is the offline/batch path only — see the design doc for the live
WebSocket path, which is a separate, not-yet-built piece.

No cross-frame object tracking (ByteTrack) is used yet: each frame, the
matcher picks the highest-confidence detection of a needed class as "the"
object being followed. That's a reasonable approximation for a
single-object demo/pilot and is the seam where a real tracker would slot
in later without changing the state machine's interface.

Both a detector (seed/main) AND a segmenter (seg_seed/seg_main), if the
project has trained both, run every sampled frame — the segmenter's
output carries real mask polygons (used for pixel-level intersection);
the detector's boxes are a fallback for classes it sees that the
segmenter doesn't.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from loguru import logger
from sqlalchemy import text
from ultralytics import YOLO

from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from ..services.sequence_engine import SequenceRunState
from ..services.seg_model import resolve_seg_model_path
from .celery_app import celery_app

# Sample every Nth frame — full-frame inference on every frame is unnecessary
# for a checkpoint-style sequence (regions are visited for many frames, not
# a single instant) and keeps a multi-minute video from taking too long.
FRAME_STRIDE = 3


def _resolve_path(filepath: str) -> Path | None:
    rel = filepath.lstrip("/")
    candidates = [
        Path(filepath),
        Path(os.getcwd()) / rel,
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]
    for p in candidates:
        try:
            if p.resolve().exists():
                return p.resolve()
        except Exception:
            continue
    return None


def _load_models(project_id: str) -> tuple[YOLO | None, YOLO | None]:
    """Returns (detector, segmenter) — either may be None if not trained."""
    detector = None
    project_model_dir = settings.model_dir / project_id
    for name in ("main_best.pt", "seed_best.pt"):
        path = project_model_dir / name
        if path.exists():
            detector = YOLO(str(path))
            break

    segmenter = None
    seg_path = resolve_seg_model_path(project_id)
    if seg_path is not None:
        segmenter = YOLO(str(seg_path))

    return detector, segmenter


def _detect_frame(detector: YOLO | None, segmenter: YOLO | None, frame) -> list[dict]:
    """Run whichever models exist against one frame, returning normalized
    detections. Segmenter results carry a real "mask" polygon."""
    h, w = frame.shape[:2]
    detections: list[dict] = []

    if detector is not None:
        results = detector.predict(frame, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": names.get(cls_idx, str(cls_idx)),
                    "xyxy": (x1 / w, y1 / h, x2 / w, y2 / h),
                    "conf": float(box.conf[0]),
                })

    if segmenter is not None:
        results = segmenter.predict(frame, verbose=False, conf=0.25)
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            has_masks = r.masks is not None
            mask_polys = r.masks.xyn if has_masks else [None] * len(r.boxes)
            for box, poly in zip(r.boxes, mask_polys):
                cls_idx = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = {
                    "class_name": names.get(cls_idx, str(cls_idx)),
                    "xyxy": (x1 / w, y1 / h, x2 / w, y2 / h),
                    "conf": float(box.conf[0]),
                }
                if poly is not None and len(poly) >= 3:
                    det["mask"] = [[float(px), float(py)] for px, py in poly]
                detections.append(det)

    return detections


@celery_app.task(bind=True, name="run_sequence_on_video")
def run_sequence_on_video(self, run_id: str) -> dict:
    db = StateDBConnector()

    with db.get_session() as conn:
        run_rows = db.execute_query(
            conn, "SELECT * FROM sequence_runs WHERE id = :id", {"id": run_id}
        )
        if not run_rows:
            return {"error": "Sequence run not found"}
        run = run_rows[0]

        seq_rows = db.execute_query(
            conn, "SELECT * FROM region_sequences WHERE id = :id", {"id": run["sequence_id"]}
        )
        video_rows = db.execute_query(
            conn, "SELECT * FROM videos WHERE id = :id", {"id": run["video_id"]}
        )

    if not seq_rows or not video_rows:
        _fail(db, run_id, "Sequence or video no longer exists")
        return {"error": "Sequence or video not found"}

    sequence = seq_rows[0]
    video = video_rows[0]
    steps = sequence["steps"]

    with db.get_session() as conn:
        conn.execute(
            text(
                "UPDATE sequence_runs SET status = 'running', total_steps = :total, task_id = :task_id WHERE id = :id"
            ),
            {"total": len(steps), "task_id": self.request.id, "id": run_id},
        )

    detector, segmenter = _load_models(sequence["project_id"])
    if detector is None and segmenter is None:
        _fail(db, run_id, "No trained model found for this project — train a model before running a sequence.")
        return {"error": "No trained model"}

    video_path = _resolve_path(video["filepath"])
    if video_path is None:
        _fail(db, run_id, "Video file not found on disk")
        return {"error": "Video file not found"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _fail(db, run_id, "Could not open video file")
        return {"error": "Could not open video"}

    state = SequenceRunState(
        steps=steps,
        mode=sequence["mode"],
        overlap_threshold=sequence["overlap_threshold"],
    )

    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % FRAME_STRIDE != 0:
                continue

            detections = _detect_frame(detector, segmenter, frame)

            event = state.process_frame(frame_number, detections)
            if event is not None:
                with db.get_session() as conn:
                    conn.execute(
                        text(
                            "UPDATE sequence_runs SET current_step = :cs, step_events = :events WHERE id = :id"
                        ),
                        {
                            "cs": state.current_step,
                            "events": json.dumps([e.__dict__ for e in state.events]),
                            "id": run_id,
                        },
                    )

            if state.is_complete:
                break
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sequence run {} failed", run_id)
        _fail(db, run_id, str(exc)[:1000])
        return {"error": str(exc)}
    finally:
        cap.release()

    with db.get_session() as conn:
        conn.execute(
            text(
                "UPDATE sequence_runs SET status = 'complete', passed = :passed, "
                "current_step = :cs, step_events = :events, finished_at = now() WHERE id = :id"
            ),
            {
                "passed": state.is_complete,
                "cs": state.current_step,
                "events": json.dumps([e.__dict__ for e in state.events]),
                "id": run_id,
            },
        )

    return {"passed": state.is_complete, "current_step": state.current_step, "total_steps": len(steps)}


def _fail(db: StateDBConnector, run_id: str, message: str) -> None:
    with db.get_session() as conn:
        conn.execute(
            text(
                "UPDATE sequence_runs SET status = 'error', error = :error, finished_at = now() WHERE id = :id"
            ),
            {"error": message, "id": run_id},
        )
