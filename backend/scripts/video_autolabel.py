"""
video_autolabel.py
~~~~~~~~~~~~~~~~~~
Label the frames of a project's videos with ANY detector .pt (e.g. your old
all-class detector) and drop the boxes straight into Review Annotation.

Workflow
--------
1. In the UI: create a project, upload the videos (name them by state, e.g.
   "locked_car1.mp4"), and click "Extract frames" on each one.
2. Run this script from the `backend/` folder:

       python scripts/video_autolabel.py --project-id <PROJECT_ID> \
           --weights /path/to/old_detector.pt --conf 0.25 \
           --preview-dir /path/to/previews

3. Open the project in the UI -> Review Annotation -> "Auto-annotated only".
   Each box is stored with source 'auto' (so Accept All / Reject All work).
   Frames the model found nothing in stay 'pending'.

Only frames that have NO annotations yet are labelled (so manual work is
never overwritten). Use --replace-auto to redo frames that only carry
machine boxes. Boxes are drawn on a copy of every labelled frame in
--preview-dir/<video name>/ so you can eyeball a video without the UI.
Frames are looked up under `video_frames/<video_id>/`, which also gives a
per-video summary (how often each class was found) at the end.
"""

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from app.connectors.statedb_connector import StateDBConnector  # noqa: E402
from app.tasks.auto_annotate import _nms_filter, _resolve_image_path  # noqa: E402


def _video_id_of(filepath: str) -> str:
    parts = filepath.replace("\\", "/").split("/")
    if "video_frames" in parts:
        i = parts.index("video_frames")
        if i + 1 < len(parts) - 1:
            return parts[i + 1]
    return "(no video)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--weights", required=True, help="detector .pt to label with")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--video-id", action="append", default=None,
                    help="only these videos (repeatable); default = every video frame in the project")
    ap.add_argument("--replace-auto", action="store_true",
                    help="also redo frames that only have machine (auto) boxes")
    ap.add_argument("--preview-dir", default=None, help="save frames with boxes drawn here")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names
    db = StateDBConnector()

    with db.get_session() as conn:
        proj = db.execute_query(conn, "SELECT id, classes FROM projects WHERE id = :p", {"p": args.project_id})
        if not proj:
            sys.exit("Project not found")
        classes = list(proj[0]["classes"] or [])
        videos = {
            v["id"]: v["original_filename"]
            for v in db.execute_query(conn, "SELECT id, original_filename FROM videos WHERE project_id = :p",
                                      {"p": args.project_id})
        }
        images = db.execute_query(
            conn,
            "SELECT i.id, i.filename, i.filepath, "
            "  (SELECT COUNT(*) FROM annotations a WHERE a.image_id = i.id) AS n_all, "
            "  (SELECT COUNT(*) FROM annotations a WHERE a.image_id = i.id AND a.source NOT IN ('auto')) AS n_manual "
            "FROM images i WHERE i.project_id = :p ORDER BY i.filepath",
            {"p": args.project_id},
        )

    todo = []
    for im in images:
        vid = _video_id_of(im["filepath"])
        if vid == "(no video)":
            continue
        if args.video_id and vid not in args.video_id:
            continue
        if im["n_manual"] > 0 or (im["n_all"] > 0 and not args.replace_auto):
            continue
        im["video_id"] = vid
        todo.append(im)
    if not todo:
        sys.exit("No video frames to label (extract frames first, or all are already annotated).")
    print(f"Labelling {len(todo)} frames from {len({t['video_id'] for t in todo})} video(s)...")

    per_video = defaultdict(Counter)
    frames_seen = Counter()
    new_classes = []

    with db.get_session() as conn:
        for n, im in enumerate(todo, 1):
            path = _resolve_image_path(im["filepath"])
            if path is None:
                print(f"  [missing file] {im['filepath']}")
                continue
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frames_seen[im["video_id"]] += 1
            res = model.predict(frame, conf=args.conf, verbose=False)
            rows = []
            for r in res:
                for box in r.boxes:
                    cname = names.get(int(box.cls[0]))
                    if not cname:
                        continue
                    xywhn = box.xywhn[0].cpu().numpy().tolist()
                    rows.append({
                        "ann_id": str(uuid.uuid4()), "image_id": im["id"], "class_name": cname,
                        "conf": float(box.conf[0]), "bbox_json": json.dumps(xywhn), "xyxy": box.xyxy[0].tolist(),
                    })
            by_class = defaultdict(list)
            for a in rows:
                by_class[a["class_name"]].append(a)
            rows = [a for lst in by_class.values() for a in _nms_filter(lst, iou_threshold=0.5)]

            if im["n_all"] > 0:  # --replace-auto: drop the old machine boxes first
                db.execute_update(conn, "DELETE FROM annotations WHERE image_id = :i AND source = 'auto'", {"i": im["id"]})
            if rows:
                db.execute_many(
                    conn,
                    "INSERT INTO annotations (id, image_id, class_name, bbox, annotation_type, points, source) "
                    "VALUES (:ann_id, :image_id, :class_name, CAST(:bbox_json AS JSONB), 'bbox', NULL, 'auto')",
                    rows,
                )
                db.execute_update(conn, "UPDATE images SET status = 'annotated' WHERE id = :i", {"i": im["id"]})
                for a in rows:
                    per_video[im["video_id"]][a["class_name"]] += 1
                    if a["class_name"] not in classes and a["class_name"] not in new_classes:
                        new_classes.append(a["class_name"])
            else:
                db.execute_update(conn, "UPDATE images SET status = 'pending' WHERE id = :i", {"i": im["id"]})

            if args.preview_dir and rows:
                out = Path(args.preview_dir) / (Path(videos.get(im["video_id"], im["video_id"])).stem)
                out.mkdir(parents=True, exist_ok=True)
                for a in rows:
                    x1, y1, x2, y2 = [int(v) for v in a["xyxy"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(frame, f"{a['class_name']} {a['conf']:.2f}", (x1, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
                cv2.imwrite(str(out / im["filename"]), frame)
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")

        if new_classes:  # training skips annotations whose class isn't registered
            db.execute_update(conn, "UPDATE projects SET classes = CAST(:c AS JSON) WHERE id = :p",
                              {"c": json.dumps(classes + new_classes), "p": args.project_id})
            print(f"Registered new classes in the project: {new_classes}")

    print("\nPer video (frames labelled -> boxes found per class):")
    for vid, total in frames_seen.items():
        print(f"  {videos.get(vid, vid)}: {total} frames  {dict(per_video[vid]) or 'NOTHING FOUND'}")


if __name__ == "__main__":
    main()
