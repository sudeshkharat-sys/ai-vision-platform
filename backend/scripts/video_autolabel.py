"""
video_autolabel.py
~~~~~~~~~~~~~~~~~~
Stream a project's videos through ANY detector .pt on this PC's GPU and save
the useful frames, with their boxes, straight into the project for Review.

Normally started by the "Auto-label videos" button (api/autolabel.py), but it
also runs by hand from the backend/ folder:

    python scripts/video_autolabel.py --project-id <ID> [--weights file.pt]
        [--fps 3] [--conf 0.25] [--window] [--video-id <ID> ...]

For every video it samples ~--fps frames per second, runs the model, and keeps
a frame only when something was detected and the boxes differ from the last
kept frame (so near-identical frames don't flood Review). Kept frames are
saved under uploads/<project>/video_frames/<video_id>/ with source='auto'
boxes. Frames with no detection are never saved. --window shows the video
with the boxes drawn (q = stop, space = pause); the run also stops when the
stop file appears. Progress is written to the status file for the UI.
"""

import argparse
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.connectors.statedb_connector import StateDBConnector  # noqa: E402
from app.tasks.auto_annotate import _nms_filter  # noqa: E402
from app.tasks.video_processing import _resolve_video_path  # noqa: E402


def autolabel_dir(project_id: str) -> Path:
    return settings.model_dir.resolve() / project_id


def default_weights(project_id: str) -> Path:
    return autolabel_dir(project_id) / "autolabel.pt"


def status_file(project_id: str) -> Path:
    return autolabel_dir(project_id) / "autolabel_status.json"


def stop_file(project_id: str) -> Path:
    return autolabel_dir(project_id) / "autolabel_stop"


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _same_as_last(rows, last, iou_same):
    """True if every box matches a same-class box of the last kept frame (and vice versa)."""
    if last is None or len(rows) != len(last):
        return False
    pool = list(last)
    for r in rows:
        hit = next((p for p in pool if p["class_name"] == r["class_name"] and _iou(p["xyxy"], r["xyxy"]) >= iou_same), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--weights", default=None, help="default: the project's uploaded autolabel.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=float, default=3.0, help="frames per second to look at (low = fast)")
    ap.add_argument("--iou-same", type=float, default=0.9, help="boxes this similar to the last kept frame = duplicate")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--video-id", action="append", default=None)
    ap.add_argument("--window", action="store_true", help="show the video with boxes drawn")
    args = ap.parse_args()

    pid = args.project_id
    weights = Path(args.weights) if args.weights else default_weights(pid)
    sfile, stop = status_file(pid), stop_file(pid)
    sfile.parent.mkdir(parents=True, exist_ok=True)
    if stop.exists():
        stop.unlink()

    status = {"state": "running", "message": "", "video": None, "videos_done": 0, "videos_total": 0,
              "frames_seen": 0, "saved": 0, "boxes": 0, "per_video": {}, "started": time.time()}

    def write(**kw):
        status.update(kw)
        sfile.write_text(json.dumps(status))

    try:
        if not weights.exists():
            raise RuntimeError(f"Detector .pt not found: {weights} (upload it first)")
        from ultralytics import YOLO
        model = YOLO(str(weights))
        names = model.names
        db = StateDBConnector()

        with db.get_session() as conn:
            proj = db.execute_query(conn, "SELECT classes FROM projects WHERE id = :p", {"p": pid})
            if not proj:
                raise RuntimeError("Project not found")
            classes = list(proj[0]["classes"] or [])
            videos = db.execute_query(conn, "SELECT id, filepath, original_filename FROM videos WHERE project_id = :p ORDER BY created_at", {"p": pid})
        if args.video_id:
            videos = [v for v in videos if v["id"] in args.video_id]
        if not videos:
            raise RuntimeError("No videos in this project")
        write(videos_total=len(videos))

        show = args.window
        new_classes, stopped = [], False
        for vi, v in enumerate(videos):
            path = _resolve_video_path(v["filepath"])
            cap = cv2.VideoCapture(str(path)) if path else None
            if cap is None or not cap.isOpened():
                write(message=f"Could not open {v['original_filename']}")
                continue
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(1, round(vid_fps / max(args.fps, 0.1)))
            vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_dir = settings.upload_dir / pid / "video_frames" / v["id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(v["original_filename"]).stem
            counts, last, idx, paused = Counter(), None, -1, False
            write(video=v["original_filename"])

            while True:
                if stop.exists():
                    stopped = True
                    break
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    idx += 1
                    if idx % step:
                        continue
                    res = model.predict(frame, conf=args.conf, verbose=False)
                    rows = []
                    for r in res:
                        for box in r.boxes:
                            cname = names.get(int(box.cls[0]))
                            if cname:
                                rows.append({"ann_id": str(uuid.uuid4()), "class_name": cname, "conf": float(box.conf[0]),
                                             "xywhn": box.xywhn[0].cpu().numpy().tolist(), "xyxy": box.xyxy[0].tolist()})
                    byc = defaultdict(list)
                    for a in rows:
                        byc[a["class_name"]].append(a)
                    rows = [a for lst in byc.values() for a in _nms_filter([dict(x, conf=x["conf"], bbox=x["xywhn"]) for x in lst], 0.5)]
                    status["frames_seen"] += 1

                    if rows and (args.no_dedupe or not _same_as_last(rows, last, args.iou_same)):
                        fid = str(uuid.uuid4())
                        cv2.imwrite(str(out_dir / f"{fid}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        with db.get_session() as conn:
                            db.execute_insert(conn,
                                "INSERT INTO images (id, project_id, filename, filepath, width, height, status, created_at) "
                                "VALUES (:id, :p, :fn, :fp, :w, :h, 'annotated', NOW())",
                                {"id": fid, "p": pid, "fn": f"{stem}_frame_{idx:06d}.jpg",
                                 "fp": f"/uploads/{pid}/video_frames/{v['id']}/{fid}.jpg", "w": vw, "h": vh})
                            db.execute_many(conn,
                                "INSERT INTO annotations (id, image_id, class_name, bbox, annotation_type, points, source) "
                                "VALUES (:ann_id, :image_id, :class_name, CAST(:bbox_json AS JSONB), 'bbox', NULL, 'auto')",
                                [{"ann_id": a["ann_id"], "image_id": fid, "class_name": a["class_name"],
                                  "bbox_json": json.dumps(a["xywhn"])} for a in rows])
                        last = rows
                        status["saved"] += 1
                        status["boxes"] += len(rows)
                        for a in rows:
                            counts[a["class_name"]] += 1
                            if a["class_name"] not in classes and a["class_name"] not in new_classes:
                                new_classes.append(a["class_name"])
                    status["per_video"][v["original_filename"]] = dict(counts)
                    write()

                    if show:
                        view = frame.copy()
                        for a in rows:
                            x1, y1, x2, y2 = [int(t) for t in a["xyxy"]]
                            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)
                            cv2.putText(view, f"{a['class_name']} {a['conf']:.2f}", (x1, max(14, y1 - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                        cv2.putText(view, f"{v['original_filename']}  saved {status['saved']}  (q stop, space pause)",
                                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        try:
                            cv2.imshow("Auto-label", view)
                        except cv2.error:
                            show = False
                            write(message="No display available for the window; continuing without it")
                if show:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        stopped = True
                        break
                    if k == ord(" "):
                        paused = not paused
            cap.release()
            if stopped:
                break
            write(videos_done=vi + 1)

        if show:
            cv2.destroyAllWindows()
        if new_classes:  # training skips annotations whose class isn't registered
            with db.get_session() as conn:
                db.execute_update(conn, "UPDATE projects SET classes = CAST(:c AS JSON) WHERE id = :p",
                                  {"c": json.dumps(classes + new_classes), "p": pid})
        write(state="stopped" if stopped else "done", new_classes=new_classes)
    except Exception as e:  # noqa: BLE001
        write(state="failed", message=str(e))
        raise


if __name__ == "__main__":
    main()
