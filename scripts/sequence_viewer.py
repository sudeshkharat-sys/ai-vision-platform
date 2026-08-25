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

Understands all the step types the web builder can produce: box/line
(crossing or opt-in area-overlap trigger_mode)/frozen-polygon regions,
"detection_class" live class-vs-class steps, "multi_region" combo steps
(2+ independent sub-regions that must all match in the same frame), and
the complete_on modes "detect" (default, instant), "detect_hold" (must
stay matched for hold_seconds — a real press, not a pass-through), and
"undetect_hold" (must stay gone for hold_seconds — gesture released).

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


def mask_intersection_percent(target_polygon, other_polygon, resolution: int = 256,
                               basis: str = "region") -> float:
    """basis picks WHAT the intersection is measured as a fraction OF:

    * "region" (default): Intersection / DRAWN REGION area — "how much of
      the region is covered by the object". A region drawn much larger
      than the object can never score high on this no matter how squarely
      the object sits inside it.
    * "object": Intersection / DETECTED OBJECT area — "how much of the
      object is inside the region", which is what "my hand is in the box"
      intuitively means; fully inside scores 100%.
    """
    def to_px(poly):
        return np.array(
            [[int(round(x * resolution)), int(round(y * resolution))] for x, y in poly],
            dtype=np.int32,
        )

    target_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    other_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(target_mask, [to_px(target_polygon)], 1)
    cv2.fillPoly(other_mask, [to_px(other_polygon)], 1)

    denominator = int(other_mask.sum()) if basis == "object" else int(target_mask.sum())
    if denominator == 0:
        return 0.0
    intersection_area = int(np.logical_and(target_mask, other_mask).sum())
    return (intersection_area / denominator) * 100.0


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


# Half-width (normalized 0-1) a line region is buffered into for
# area-overlap matching (trigger_mode == "overlap") — a true zero-width
# line has no area, so "Intersection Area / Target Area" needs a strip
# to compare against, not a geometric line. Matches
# backend/app/services/sequence_match.py's LINE_OVERLAP_HALF_WIDTH.
LINE_OVERLAP_HALF_WIDTH = 0.02


def line_to_polygon(x1, y1, x2, y2, half_width=LINE_OVERLAP_HALF_WIDTH):
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return [[x1 - half_width, y1 - half_width], [x1 + half_width, y1 - half_width],
                [x1 + half_width, y1 + half_width], [x1 - half_width, y1 + half_width]]
    nx, ny = -dy / length * half_width, dx / length * half_width
    return [[x1 + nx, y1 + ny], [x2 + nx, y2 + ny], [x2 - nx, y2 - ny], [x1 - nx, y1 - ny]]


def _match_single(step: dict, detections: list, threshold_pct: float,
                  basis: str = "region") -> dict:
    target_type = step.get("target_type", "region")

    if target_type == "detection_class":
        target_class = step.get("target_class")
        target_polygon = best_polygon_for_class(detections, target_class) if target_class else None
        if target_polygon is None:
            return {"matched": False, "testable": True, "per_class": [],
                    "note": f'Target class "{target_class}" not detected this frame.'}
    else:
        region_type = step.get("region_type")
        if region_type == "box":
            x1, y1, x2, y2 = step["region_coords"]
            target_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif region_type == "polygon":
            # Frozen class boundary (from "Freeze boundary" in the web app) —
            # a static shape, so it works even if that class is occluded now.
            target_polygon = step["region_coords"]
        elif region_type == "line" and step.get("trigger_mode") == "overlap":
            x1, y1, x2, y2 = step["region_coords"]
            target_polygon = line_to_polygon(x1, y1, x2, y2)
        else:
            return {"matched": False, "testable": False, "per_class": [],
                    "note": "Line regions use motion-crossing, not evaluated here."}

    needed = step.get("required_classes") or [step["required_class"]]
    per_class = []
    for cls in needed:
        other_polygon = best_polygon_for_class(detections, cls)
        pct = mask_intersection_percent(target_polygon, other_polygon, basis=basis) if other_polygon else 0.0
        per_class.append({"class_name": cls, "matched": pct >= threshold_pct, "percent": round(pct, 1)})

    return {"matched": bool(needed) and all(c["matched"] for c in per_class),
            "testable": True, "per_class": per_class, "note": None}


def evaluate_step(step: dict, detections: list, threshold_pct: float) -> dict:
    """"multi_region" (a "Combo" step from the web builder) is 2+
    independent sub_targets that must ALL match in the SAME frame (e.g.
    left hand on region A AND right hand on region B at once) — evaluate
    each sub-target the same way a normal region step is, via
    _match_single, and AND every result together. Everything else goes
    straight to _match_single."""
    if step.get("target_type") == "multi_region":
        sub_targets = step.get("sub_targets") or []
        if not sub_targets:
            return {"matched": False, "testable": True, "per_class": [],
                    "note": "Combo step has no sub-regions defined."}
        # A combo step's own overlap_basis applies to every sub-region —
        # sub-targets carry only geometry + classes, not match semantics.
        basis = step.get("overlap_basis") or "region"
        results = [_match_single(sub, detections, threshold_pct, basis) for sub in sub_targets]
        testable = all(r["testable"] for r in results)
        return {"matched": testable and all(r["matched"] for r in results),
                "testable": testable,
                "per_class": [c for r in results for c in r["per_class"]],
                "note": "; ".join(r["note"] for r in results if r["note"]) or None}
    return _match_single(step, detections, threshold_pct, step.get("overlap_basis") or "region")


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
    if step.get("target_type") == "multi_region":
        classes = []
        for sub in step.get("sub_targets") or []:
            classes.extend(sub.get("required_classes") or [sub.get("required_class")])
        return [c for c in classes if c]
    return list(step["required_classes"]) if step.get("required_classes") else [step["required_class"]]


def _is_line_region(step: dict) -> bool:
    return step.get("target_type", "region") == "region" and step.get("region_type") == "line"


def _uses_crossing(step: dict) -> bool:
    """A line region defaults to motion-crossing. If trigger_mode ==
    "overlap" it's evaluated exactly like a box instead (see _match_single)."""
    return _is_line_region(step) and step.get("trigger_mode", "crossing") != "overlap"


# How many consecutive dropped-match frames a detect_hold step tolerates
# before treating it as a real release rather than hand jitter.
MISS_TOLERANCE = 2

# Minimum detection confidence required before a live class detection is
# trusted to re-sync a frozen region's position (see _resolve_target) —
# protects against a shaky/wrong guess (e.g. two adjacent keys like N
# under I both under the hand at once) corrupting a good synced position.
MIN_RESYNC_CONFIDENCE = 0.4


@dataclass
class SequenceState:
    steps: list
    mode: str = "strict"
    overlap_threshold: float = 0.5
    current_step: int = 0
    prev_centers: dict = field(default_factory=dict)
    last_reason: str = "watching"
    # Consecutive frames the current step's match condition has held true
    # (complete_on="detect_hold") or false (="undetect_hold").
    hold_counter: int = 0
    # Per-step-index drift-corrected polygon for a frozen "polygon" region
    # step that also carries target_class (see _resolve_target).
    synced_polygons: dict = field(default_factory=dict)
    # Consecutive dropped-match frames for the current detect_hold step —
    # see MISS_TOLERANCE.
    miss_counter: int = 0
    # Ordered log of {label, status} for every step, refreshed each frame —
    # drives the on-screen "Step 1: ... complete" checklist.
    step_status: list = field(default_factory=list)
    # Per-class {class_name, matched, percent} from the CURRENT step's most
    # recent match attempt — surfaced on-screen so "both hands look like
    # they're in the box but it won't pass" can actually be diagnosed
    # (e.g. only 42% of the drawn region is covered vs the 70% required)
    # instead of guessed at.
    last_detail: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def reset(self):
        self.current_step = 0
        self.prev_centers = {}
        self.hold_counter = 0
        self.miss_counter = 0
        self.synced_polygons = {}
        self.last_reason = "watching"

    def _resolve_target(self, step: dict, step_index: int, detections: list, allow_resync: bool = True) -> dict:
        """A frozen "polygon" region step with target_class set gets
        re-synced to that class's own live-detected mask whenever it's
        actually visible this frame — self-corrects for camera shake/
        drift instead of staying stuck at the exact spot it was frozen
        at. Falls back to the last known-good (or original frozen)
        polygon when the class is occluded/not detected this frame.

        allow_resync=False keeps the current position WITHOUT updating
        it — used mid-hold so the target can't wobble frame-to-frame
        from live-detection jitter while a press is already in progress."""
        if step.get("target_type", "region") != "region" or step.get("region_type") != "polygon":
            return step
        target_class = step.get("target_class")
        if not target_class:
            return step
        if allow_resync:
            best_conf = max(
                (det.get("conf", 0.0) for det in detections if det["class_name"] == target_class),
                default=-1.0,
            )
            if best_conf >= MIN_RESYNC_CONFIDENCE:
                live_polygon = best_polygon_for_class(detections, target_class)
                if live_polygon is not None:
                    self.synced_polygons[step_index] = live_polygon
        effective_coords = self.synced_polygons.get(step_index, step["region_coords"])
        if effective_coords is step["region_coords"]:
            return step
        return {**step, "region_coords": effective_coords}

    def process_frame(self, detections: list) -> str:
        """Returns a short status string for this frame: 'matched',
        'wrong_region_reset', 'wrong_region_ignored', or 'watching'."""
        if self.is_complete:
            return "complete"

        target = self.steps[self.current_step]
        needed_classes = _step_classes(target)
        threshold_pct = self.overlap_threshold * 100.0
        complete_on = target.get("complete_on")

        if complete_on == "undetect_hold":
            return self._check_undetect_hold(target, needed_classes, detections)
        if complete_on == "detect_hold":
            return self._check_detect_hold(target, needed_classes, detections, threshold_pct)

        if _uses_crossing(target):
            matched = self._check_line_region(target, needed_classes, detections)
            self.last_detail = []
        else:
            resolved = self._resolve_target(target, self.current_step, detections)
            result = evaluate_step(resolved, detections, threshold_pct)
            matched = result["matched"]
            self.last_detail = result["per_class"]

        if matched:
            self.current_step += 1
            for cls in needed_classes:
                self.prev_centers.pop(cls, None)
            self.last_reason = "matched"
            return "matched"

        # Only steps still AHEAD (idx > current_step) count as a "wrong
        # region hit" — a step already passed is skipped, since the hand
        # naturally lingers over the region it just completed for a
        # frame or two while moving toward the next one; that's not a
        # mistake, just physics.
        wrong_region_hit = False
        for cls in needed_classes:
            for idx, step in enumerate(self.steps):
                if idx <= self.current_step or _uses_crossing(step):
                    continue
                if cls not in _step_classes(step):
                    continue
                other_resolved = self._resolve_target(step, idx, detections)
                if evaluate_step(other_resolved, detections, threshold_pct)["matched"]:
                    wrong_region_hit = True
                    break
            if wrong_region_hit:
                break

        if wrong_region_hit and self.mode == "strict":
            self.current_step = 0
            self.prev_centers = {}
            self.synced_polygons = {}
            self.hold_counter = 0
            self.miss_counter = 0
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

    def _advance(self) -> str:
        target = self.steps[self.current_step]
        for cls in _step_classes(target):
            self.prev_centers.pop(cls, None)
        self.current_step += 1
        self.hold_counter = 0
        self.miss_counter = 0
        self.last_reason = "matched"
        return "matched"

    def _check_undetect_hold(self, target, needed_classes, detections) -> str:
        detected_now = any(det["class_name"] in needed_classes for det in detections)
        if detected_now:
            self.hold_counter = 0
            self.last_reason = "watching"
            return "watching"
        self.hold_counter += 1
        if self.hold_counter < target.get("hold_frames", 1):
            self.last_reason = "watching"
            return "watching"
        return self._advance()

    def _check_detect_hold(self, target, needed_classes, detections, threshold_pct) -> str:
        if _uses_crossing(target):
            matched_now = self._check_line_region(target, needed_classes, detections)
            self.last_detail = []
        else:
            resolved = self._resolve_target(target, self.current_step, detections, allow_resync=(self.hold_counter == 0))
            result = evaluate_step(resolved, detections, threshold_pct)
            matched_now = result["matched"]
            self.last_detail = result["per_class"]
        if not matched_now:
            # A hand naturally jitters slightly during a long hold — don't
            # wipe out the whole hold on a single dropped-match frame, only
            # after MISS_TOLERANCE in a row (a real release, not a blip).
            self.miss_counter += 1
            if self.miss_counter >= MISS_TOLERANCE:
                self.hold_counter = 0
                self.miss_counter = 0
            self.last_reason = "watching"
            return "watching"
        self.miss_counter = 0
        self.hold_counter += 1
        if self.hold_counter < target.get("hold_frames", 1):
            self.last_reason = "watching"
            return "watching"
        return self._advance()


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


def draw_overlay(frame, detections, target, status, detail=None, threshold_pct=0.0, basis="region"):
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

    step_color = _STATUS_COLORS.get(status, (255, 255, 255))

    def _draw_region_shape(rt, coords):
        if rt == "box":
            rx1, ry1, rx2, ry2 = coords
            cv2.rectangle(img, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)), step_color, 2)
        elif rt == "line":
            x1, y1, x2, y2 = coords
            cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), step_color, 3)
        elif rt == "polygon":
            # Class-target (frozen letter) regions specifically get a black
            # ring drawn UNDER the status-colored outline — none of your
            # trained classes' own segmentation masks use black/gray, so this
            # boundary always stays visible even against a screen full of
            # jumbled, similarly-colored mask fills from every other detected
            # class. Gate (box/line) regions don't need this — they aren't
            # drawn over other classes' masks the same way.
            pts = np.array([[int(px * w), int(py * h)] for px, py in coords], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=5)
            cv2.polylines(img, [pts], isClosed=True, color=step_color, thickness=3)

    if target and target.get("target_type") == "multi_region":
        # A combo step has no single region_type/region_coords of its own —
        # draw every sub-region so the whole combo's gate is visible.
        for sub in target.get("sub_targets", []):
            _draw_region_shape(sub.get("region_type"), sub.get("region_coords"))
    elif target and target.get("target_type", "region") == "region":
        _draw_region_shape(target.get("region_type"), target.get("region_coords"))

    step_label = target.get("label", "") if target else "COMPLETE"
    banner_color = _STATUS_COLORS.get(status, (255, 255, 255))

    # Second banner line: exact overlap %% per required class this frame,
    # vs the threshold — the actual number to look at when a hand LOOKS
    # like it's inside the box but the step still won't pass (e.g. only
    # 42%% of the drawn region is covered, but overlap_threshold needs 70%%
    # — the box is bigger than the hand, or the hand is only in one corner
    # of it, not truly "touching" by this formula's definition).
    banner_h = 34
    if detail:
        parts = []
        for c in detail:
            mark = "OK" if c["matched"] else "X"
            parts.append(f'{c["class_name"]}:{c["percent"]:.0f}%{mark}')
        detail_text = f'  [{basis}] need >= {threshold_pct:.0f}%   ' + '   '.join(parts)
        banner_h = 56
        cv2.rectangle(img, (0, 34), (w, banner_h), (30, 30, 30), -1)
        cv2.putText(img, detail_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w, 34), (30, 30, 30), -1)
    cv2.putText(img, f'Step: {step_label}   [{status}]', (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2, cv2.LINE_AA)
    return img


def draw_checklist(img, steps, current_step):
    """Left-side panel listing every step with its status — 'Step 1: gate1
    - COMPLETE', 'Step 2: m - CURRENT', 'Step 3: a - pending', etc."""
    h, w = img.shape[:2]
    n = len(steps)
    panel_h = 24 + n * 22
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (260, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    for i, step in enumerate(steps):
        if i < current_step:
            text, color = f'Step {i + 1}: {step.get("label", "")} - COMPLETE', (74, 222, 128)
        elif i == current_step:
            text, color = f'Step {i + 1}: {step.get("label", "")} - CURRENT', (11, 170, 250)
        else:
            text, color = f'Step {i + 1}: {step.get("label", "")} - pending', (140, 140, 140)
        y = h - panel_h + 22 + i * 22
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
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
    parser.add_argument("--overlap-basis", choices=["region", "object"],
                        help="What the overlap %% is measured as a fraction OF. "
                             "'region' (the saved default): how much of the DRAWN REGION the object covers — "
                             "a region drawn bigger than the object can never score high. "
                             "'object': how much of the DETECTED OBJECT is inside the region — "
                             "'my hand is in the box' in the intuitive sense (fully inside = 100%%). "
                             "Overrides whatever the sequence.json says, for all steps.")
    parser.add_argument("--overlap-threshold", type=float,
                        help="Override the sequence's overlap_threshold (0-1) without editing the JSON.")
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

    source = args.webcam if args.webcam is not None else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    # hold_seconds (a step's user-facing config for detect_hold/undetect_hold)
    # -> hold_frames, using this source's real FPS. This script evaluates
    # every frame (no FRAME_STRIDE sampling like the backend's Celery task),
    # so frames-per-second here is just the source's own FPS.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for step in steps:
        if step.get("complete_on") in ("undetect_hold", "detect_hold") and step.get("hold_frames") is None:
            step["hold_frames"] = max(1, round(step.get("hold_seconds", 1.0) * fps))
        if args.overlap_basis:
            step["overlap_basis"] = args.overlap_basis

    threshold = args.overlap_threshold if args.overlap_threshold is not None \
        else sequence.get("overlap_threshold", 0.5)
    state = SequenceState(steps=steps, mode=sequence.get("mode", "strict"),
                           overlap_threshold=threshold)
    basis_label = args.overlap_basis or (steps[0].get("overlap_basis") if steps else None) or "region"
    print(f'[sequence_viewer] overlap basis = "{basis_label}", threshold = {threshold:.2f} '
          f'({threshold * 100:.0f}%)')

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
            step_index = state.current_step
            target = steps[step_index] if not state.is_complete else None
            status = state.process_frame(detections)
            # Draw the RESYNCED position (if this step froze one and it's
            # been re-synced this run), not the original frozen coords —
            # otherwise the drawn box stays visually stuck at its first
            # capture forever even though matching itself already moved on.
            draw_target = target
            if target is not None and target.get("region_type") == "polygon" and target.get("target_class"):
                synced = state.synced_polygons.get(step_index)
                if synced is not None:
                    draw_target = {**target, "region_coords": synced}
            display = draw_overlay(frame, detections, draw_target, status,
                                    detail=state.last_detail, threshold_pct=state.overlap_threshold * 100.0,
                                    basis=basis_label)
            display = draw_checklist(display, steps, state.current_step)

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
