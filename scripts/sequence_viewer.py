"""
sequence_viewer.py
~~~~~~~~~~~~~~~~~~~
Standalone, no-server tool to RUN a saved Sequence Detection against a
video or webcam and WATCH it live in a real OpenCV window — the thing the
web app's "Run" button can't do, because it runs on a headless backend
server with no display attached. This script is meant to be downloaded
and run on your own machine, not deployed as part of the platform.

It does NOT talk to the web app's database, Celery, or FastAPI at all —
it's fully self-contained (just opencv-python, numpy, ultralytics). You
export a sequence as JSON from the web app, and pass it here alongside
your .pt model(s) and a video file or webcam.

Setup (once)
------------
    pip install opencv-python numpy ultralytics

Getting a sequence.json
------------------------
In the Sequence Detection page, open a saved sequence and click
"Export JSON" — save the downloaded file anywhere, e.g. sequence.json.

Usage
-----
    # Run against a video file, using a detector model:
    python sequence_viewer.py --model main_best.pt --video clip.mp4 --sequence sequence.json

    # Also use a segmentation model for real mask-intersection matching
    # (recommended — matches exactly what the web app's matcher does):
    python sequence_viewer.py --model main_best.pt --seg-model seg_main_best.pt \
        --video clip.mp4 --sequence sequence.json

    # Live webcam instead of a file (0 = default camera):
    python sequence_viewer.py --model main_best.pt --seg-model seg_main_best.pt \
        --webcam 0 --sequence sequence.json

    # If the reference frame the regions were drawn on doesn't match the
    # camera/video's orientation, rotate incoming frames to match:
    python sequence_viewer.py ... --rotate cw   # cw | ccw | 180

Controls in the window
-----------------------
    q / Esc   — quit
    r         — reset the sequence progress back to step 1
    p         — pause / resume
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ── Matching logic — kept identical to backend/app/services/sequence_match.py
# and sequence_engine.py, so a sequence behaves the same way here as it did
# when you tested it in the web app. ──────────────────────────────────────

def bbox_to_polygon(xyxy):
    x1, y1, x2, y2 = xyxy
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def mask_intersection_percent(target_polygon, other_polygon, resolution: int = 256) -> float:
    def to_px(poly):
        return np.array(
            [[int(round(x * resolution)), int(round(y * resolution))] for x, y in poly],
            dtype=np.int32,
        )

    target_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    other_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(target_mask, [to_px(target_polygon)], 1)
    cv2.fillPoly(other_mask, [to_px(other_polygon)], 1)

    target_area = int(target_mask.sum())
    if target_area == 0:
        return 0.0
    intersection_area = int(np.logical_and(target_mask, other_mask).sum())
    return (intersection_area / target_area) * 100.0


def best_polygon_for_class(detections, class_name):
    best, best_conf = None, -1.0
    for det in detections:
        if det["class_name"] != class_name:
            continue
        conf = det.get("conf", 0.0)
        if conf > best_conf:
            best_conf = conf
            best = det.get("mask") or bbox_to_polygon(det["xyxy"])
    return best


def evaluate_step(step: dict, detections: list, threshold_pct: float) -> dict:
    target_type = step.get("target_type", "region")

    if target_type == "detection_class":
        target_class = step.get("target_class")
        target_polygon = best_polygon_for_class(detections, target_class) if target_class else None
        if target_polygon is None:
            return {"matched": False, "testable": True, "per_class": [],
                    "note": f'Target class "{target_class}" not detected this frame.'}
    else:
        if step.get("region_type") != "box":
            return {"matched": False, "testable": False, "per_class": [],
                    "note": "Line regions use motion-crossing, not evaluated here."}
        x1, y1, x2, y2 = step["region_coords"]
        target_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    needed = step.get("required_classes") or [step["required_class"]]
    per_class = []
    for cls in needed:
        other_polygon = best_polygon_for_class(detections, cls)
        pct = mask_intersection_percent(target_polygon, other_polygon) if other_polygon else 0.0
        per_class.append({"class_name": cls, "matched": pct >= threshold_pct, "percent": round(pct, 1)})

    return {"matched": bool(needed) and all(c["matched"] for c in per_class),
            "testable": True, "per_class": per_class, "note": None}


def segments_intersect(p1, p2, p3, p4) -> bool:
    def orientation(a, b, c) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c) -> bool:
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    o1, o2 = orientation(p1, p2, p3), orientation(p1, p2, p4)
    o3, o4 = orientation(p3, p4, p1), orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True
    return False


def _step_classes(step: dict) -> list:
    return list(step["required_classes"]) if step.get("required_classes") else [step["required_class"]]


def _is_line_region(step: dict) -> bool:
    return step.get("target_type", "region") == "region" and step.get("region_type") == "line"


@dataclass
class SequenceState:
    steps: list
    mode: str = "strict"
    overlap_threshold: float = 0.5
    current_step: int = 0
    prev_centers: dict = field(default_factory=dict)
    last_reason: str = "watching"

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def reset(self):
        self.current_step = 0
        self.prev_centers = {}
        self.last_reason = "watching"

    def process_frame(self, detections: list) -> str:
        """Returns a short status string for this frame: 'matched',
        'wrong_region_reset', 'wrong_region_ignored', or 'watching'."""
        if self.is_complete:
            return "complete"

        target = self.steps[self.current_step]
        needed_classes = _step_classes(target)
        threshold_pct = self.overlap_threshold * 100.0

        if _is_line_region(target):
            matched = self._check_line_region(target, needed_classes, detections)
        else:
            matched = evaluate_step(target, detections, threshold_pct)["matched"]

        if matched:
            self.current_step += 1
            for cls in needed_classes:
                self.prev_centers.pop(cls, None)
            self.last_reason = "matched"
            return "matched"

        wrong_region_hit = False
        for cls in needed_classes:
            for idx, step in enumerate(self.steps):
                if idx == self.current_step or _is_line_region(step):
                    continue
                if cls not in _step_classes(step):
                    continue
                if evaluate_step(step, detections, threshold_pct)["matched"]:
                    wrong_region_hit = True
                    break
            if wrong_region_hit:
                break

        if wrong_region_hit and self.mode == "strict":
            self.current_step = 0
            self.prev_centers = {}
            self.last_reason = "wrong_region_reset"
            return "wrong_region_reset"
        if wrong_region_hit:
            self.last_reason = "wrong_region_ignored"
            return "wrong_region_ignored"

        self.last_reason = "watching"
        return "watching"

    def _check_line_region(self, target, needed_classes, detections) -> bool:
        coords = target["region_coords"]
        p3, p4 = (coords[0], coords[1]), (coords[2], coords[3])
        all_crossed = True
        for cls in needed_classes:
            best_xyxy, best_conf = None, -1.0
            for det in detections:
                if det["class_name"] != cls:
                    continue
                if det.get("conf", 0.0) > best_conf:
                    best_conf, best_xyxy = det.get("conf", 0.0), det["xyxy"]
            crossed = False
            if best_xyxy is not None:
                dx1, dy1, dx2, dy2 = best_xyxy
                curr_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                prev_center = self.prev_centers.get(cls)
                if prev_center is not None:
                    crossed = segments_intersect(prev_center, curr_center, p3, p4)
                self.prev_centers[cls] = curr_center
            if not crossed:
                all_crossed = False
        return all_crossed


# ── Drawing (mirrors backend/app/tasks/sequence_run.py's _draw_overlay) ───

_STATUS_COLORS = {
    "matched": (74, 222, 128),
    "wrong_region_reset": (68, 68, 239),
    "wrong_region_ignored": (11, 170, 250),
    "watching": (250, 170, 11),
    "complete": (74, 222, 128),
}
_CLASS_PALETTE = [
    (80, 220, 255), (255, 140, 60), (120, 220, 120), (200, 120, 255),
    (60, 200, 255), (255, 200, 80), (150, 150, 255), (255, 100, 200),
]
_class_color_cache: dict = {}


def _color_for_class(class_name: str):
    if class_name not in _class_color_cache:
        _class_color_cache[class_name] = _CLASS_PALETTE[len(_class_color_cache) % len(_CLASS_PALETTE)]
    return _class_color_cache[class_name]


def draw_overlay(frame, detections, target, status):
    img = frame.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        p1 = (int(x1 * w), int(y1 * h))
        color = _color_for_class(det["class_name"])
        label_anchor = p1

        mask = det.get("mask")
        if mask and len(mask) >= 3:
            pts = np.array([[int(px * w), int(py * h)] for px, py in mask], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            label_anchor = (int(pts[:, 0].min()), int(pts[:, 1].min()))
        else:
            p2 = (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, p1, p2, color, 2)

        label = f'{det["class_name"]} {det.get("conf", 0):.2f}'
        cv2.putText(img, label, (label_anchor[0], max(14, label_anchor[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if target and target.get("target_type", "region") == "region" and target.get("region_type") == "box":
        rx1, ry1, rx2, ry2 = target["region_coords"]
        step_color = _STATUS_COLORS.get(status, (255, 255, 255))
        cv2.rectangle(img, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)), step_color, 2)
    elif target and target.get("region_type") == "line":
        x1, y1, x2, y2 = target["region_coords"]
        step_color = _STATUS_COLORS.get(status, (255, 255, 255))
        cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), step_color, 3)

    step_label = target.get("label", "") if target else "COMPLETE"
    banner_color = _STATUS_COLORS.get(status, (255, 255, 255))
    cv2.rectangle(img, (0, 0), (w, 34), (30, 30, 30), -1)
    cv2.putText(img, f'Step: {step_label}   [{status}]', (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2, cv2.LINE_AA)
    return img


# ── Detection ───────────────────────────────────────────────────────────

def detect_frame(detector, segmenter, frame) -> list:
    h, w = frame.shape[:2]
    detections = []

    if detector is not None:
        for r in detector.predict(frame, verbose=False, conf=0.25):
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
        for r in segmenter.predict(frame, verbose=False, conf=0.25):
            if r.boxes is None:
                continue
            names = r.names
            mask_polys = r.masks.xyn if r.masks is not None else [None] * len(r.boxes)
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


_ROTATE_CODES = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Path to a detector .pt (main_best.pt / seed_best.pt)")
    parser.add_argument("--seg-model", help="Path to a segmenter .pt (seg_main_best.pt) — recommended for real mask intersection")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--webcam", type=int, help="Webcam index (e.g. 0) instead of --video")
    parser.add_argument("--sequence", required=True, help="Path to a sequence.json exported from the web app")
    parser.add_argument("--rotate", choices=["cw", "ccw", "180"], help="Rotate incoming frames to match the reference orientation")
    args = parser.parse_args()

    if not args.model and not args.seg_model:
        sys.exit("Provide --model and/or --seg-model.")
    if not args.video and args.webcam is None:
        sys.exit("Provide --video PATH or --webcam INDEX.")

    from ultralytics import YOLO
    detector = YOLO(args.model) if args.model else None
    segmenter = YOLO(args.seg_model) if args.seg_model else None

    sequence = json.loads(Path(args.sequence).read_text())
    steps = sequence["steps"]
    state = SequenceState(steps=steps, mode=sequence.get("mode", "strict"),
                           overlap_threshold=sequence.get("overlap_threshold", 0.5))

    source = args.webcam if args.webcam is not None else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    rotate_code = _ROTATE_CODES.get(args.rotate) if args.rotate else None

    window = f'Sequence Viewer — {sequence.get("name", "sequence")}'
    paused = False
    print("Controls: q/Esc = quit, r = reset progress, p = pause/resume")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video (or camera disconnected).")
                break
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            detections = detect_frame(detector, segmenter, frame)
            target = steps[state.current_step] if not state.is_complete else None
            status = state.process_frame(detections)
            display = draw_overlay(frame, detections, target, status)

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            state.reset()
            print("Sequence progress reset.")
        elif key == ord("p"):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
