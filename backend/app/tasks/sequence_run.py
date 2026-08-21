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


_DEBUG_COLORS = {
    "matched": (74, 222, 128),          # green
    "wrong_region_reset": (68, 68, 239),  # red (BGR)
    "wrong_region_ignored": (11, 170, 250),  # amber
}


def _save_debug_frame(frame, detections: list[dict], target: dict, project_id: str, run_id: str,
                       frame_number: int, reason: str) -> str | None:
    """Draw the frame's detections + the current step's region (if a box)
    onto a copy of the frame and save it, so a run can actually be
    debugged by looking at what the model saw at the moment a step
    passed or reset — not just a text log. Returns the web-accessible
    URL, or None if saving failed (never blocks the run itself)."""
    try:
        img = frame.copy()
        h, w = img.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, p1, p2, (255, 255, 255), 1)
            label = f'{det["class_name"]} {det.get("conf", 0):.2f}'
            cv2.putText(img, label, (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if target.get("target_type", "region") == "region" and target.get("region_type") == "box":
            rx1, ry1, rx2, ry2 = target["region_coords"]
            color = _DEBUG_COLORS.get(reason, (0, 165, 255))
            cv2.rectangle(img, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)), color, 2)

        cv2.putText(img, f'Step: {target.get("label", "")}  ({reason})', (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DEBUG_COLORS.get(reason, (255, 255, 255)), 1, cv2.LINE_AA)

        out_dir = settings.upload_dir / project_id / "sequence_debug" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{frame_number}_{reason}.jpg"
        cv2.imwrite(str(out_dir / filename), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f"/uploads/{project_id}/sequence_debug/{run_id}/{filename}"
    except Exception:
        logger.exception("Failed to save debug frame for run {}", run_id)
        return None


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

    # Persistent log of event dicts (each optionally carrying a
    # "frame_url" debug snapshot) — the single source of truth written to
    # the DB both mid-run and at completion, so a debug image attached to
    # an earlier event is never lost by re-deriving from state.events.
    event_log: list[dict] = []

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

            # The step being evaluated THIS frame — captured before
            # process_frame() potentially advances current_step, so the
            # debug snapshot shows the region/step that was actually judged.
            evaluated_step = steps[state.current_step] if not state.is_complete else None

            event = state.process_frame(frame_number, detections)
            if event is not None:
                event_dict = dict(event.__dict__)
                if evaluated_step is not None:
                    frame_url = _save_debug_frame(
                        frame, detections, evaluated_step,
                        sequence["project_id"], run_id, frame_number, event.reason,
                    )
                    if frame_url:
                        event_dict["frame_url"] = frame_url
                event_log.append(event_dict)

                with db.get_session() as conn:
                    conn.execute(
                        text(
                            "UPDATE sequence_runs SET current_step = :cs, step_events = :events WHERE id = :id"
                        ),
                        {
                            "cs": state.current_step,
                            "events": json.dumps(event_log),
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
                "events": json.dumps(event_log),
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
