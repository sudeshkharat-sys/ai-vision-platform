"""
sequence_engine.py
~~~~~~~~~~~~~~~~~~~
The per-run state machine: given a saved sequence's ordered steps and a
stream of per-frame detections, tracks how far through the sequence a
single tracked object has progressed.

Box and "detection_class" target steps delegate their match check to
sequence_match.evaluate_step, which does real pixel-level mask
intersection (Intersection Area / Target Area x 100, per spec) shared
with the single-image tester. Line regions are handled here directly,
since a line crossing needs motion between frames — something that
single-frame mask intersection can't express.

This module does NOT track objects across frames (no ByteTrack) — for
each required class, the highest-confidence detection of that class this
frame is used. That keeps this class simple, testable with fake data,
and swappable later for a real multi-object tracker without changing its
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .sequence_geometry import segments_intersect
from .sequence_match import evaluate_step


@dataclass
class StepEvent:
    step_index: int
    label: str
    frame_number: int
    matched: bool
    reason: str  # "matched" | "wrong_region_reset" | "wrong_region_ignored"


def _step_classes(step: dict) -> list[str]:
    """A step's required classes — required_classes if set (2+ classes,
    ALL must be present at once), else the single legacy required_class."""
    classes = step.get("required_classes")
    if classes:
        return list(classes)
    return [step["required_class"]]


def _is_line_region(step: dict) -> bool:
    return step.get("target_type", "region") == "region" and step.get("region_type") == "line"


@dataclass
class SequenceRunState:
    steps: list[dict]
    mode: str = "strict"                # "strict" | "lenient"
    overlap_threshold: float = 0.5      # 0-1 fraction; converted to 0-100 for evaluate_step
    current_step: int = 0
    # Per-class last-seen center point, for line-region crossing checks.
    prev_centers: dict[str, tuple[float, float]] = field(default_factory=dict)
    events: list[StepEvent] = field(default_factory=list)
    # Consecutive sampled frames the current step's match condition has
    # held true (complete_on="detect_hold") or false (="undetect_hold").
    hold_counter: int = 0

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def current_target(self) -> dict | None:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def process_frame(self, frame_number: int, detections: list[dict]) -> StepEvent | None:
        """detections: [{"class_name": str, "xyxy": (x1,y1,x2,y2),
        "mask": [[x,y],...] (optional), "conf": float}, ...] for this
        frame, already normalized 0-1. Returns a StepEvent if something
        notable happened (a step matched, or a wrong-region hit in strict
        mode reset progress).
        """
        if self.is_complete:
            return None

        target = self.current_target()
        needed_classes = _step_classes(target)
        threshold_pct = self.overlap_threshold * 100.0

        complete_on = target.get("complete_on")
        if complete_on == "undetect_hold":
            return self._check_undetect_hold(target, needed_classes, frame_number, detections)
        if complete_on == "detect_hold":
            return self._check_detect_hold(target, needed_classes, frame_number, detections, threshold_pct)

        if _is_line_region(target):
            matched = self._check_line_region(target, needed_classes, detections)
        else:
            result = evaluate_step(target, detections, threshold_pct)
            matched = result["matched"]

        if matched:
            return self._advance(target, needed_classes, frame_number)

        # Wrong-region hit: any of this step's required classes has a
        # detection landing in a DIFFERENT step's box/detection_class
        # target instead. (Only checked for non-line steps — a line
        # region's "hit" is a crossing event, not a static landing.)
        wrong_region_hit = False
        for cls in needed_classes:
            for idx, step in enumerate(self.steps):
                if idx == self.current_step or _is_line_region(step):
                    continue
                other_needed = _step_classes(step)
                if cls not in other_needed:
                    continue
                other_result = evaluate_step(step, detections, threshold_pct)
                if other_result["matched"]:
                    wrong_region_hit = True
                    break
            if wrong_region_hit:
                break

        if wrong_region_hit and self.mode == "strict":
            event = StepEvent(
                step_index=self.current_step,
                label=target["label"],
                frame_number=frame_number,
                matched=False,
                reason="wrong_region_reset",
            )
            self.events.append(event)
            self.current_step = 0
            self.prev_centers = {}
            return event

        if wrong_region_hit:  # lenient
            return StepEvent(
                step_index=self.current_step,
                label=target["label"],
                frame_number=frame_number,
                matched=False,
                reason="wrong_region_ignored",
            )

        return None

    def _check_undetect_hold(
        self, target: dict, needed_classes: list[str], frame_number: int, detections: list[dict]
    ) -> StepEvent | None:
        """Step passes once ALL of needed_classes have been absent from
        the detections for hold_frames sampled frames in a row. Any frame
        where one reappears resets the counter to zero."""
        detected_now = any(det["class_name"] in needed_classes for det in detections)

        if detected_now:
            self.hold_counter = 0
            return None

        self.hold_counter += 1
        hold_frames = target.get("hold_frames", 1)
        if self.hold_counter < hold_frames:
            return None

        return self._advance(target, needed_classes, frame_number)

    def _check_detect_hold(
        self, target: dict, needed_classes: list[str], frame_number: int,
        detections: list[dict], threshold_pct: float,
    ) -> StepEvent | None:
        """Step passes once the step's normal match condition (region
        overlap / line crossing / class overlap) has held true for
        hold_frames sampled frames IN A ROW. A single frame where it
        drops out resets the counter — this is what tells apart a real
        'press and hold' from the hand just passing over the region on
        its way somewhere else."""
        if _is_line_region(target):
            matched_now = self._check_line_region(target, needed_classes, detections)
        else:
            result = evaluate_step(target, detections, threshold_pct)
            matched_now = result["matched"]

        if not matched_now:
            self.hold_counter = 0
            return None

        self.hold_counter += 1
        hold_frames = target.get("hold_frames", 1)
        if self.hold_counter < hold_frames:
            return None

        return self._advance(target, needed_classes, frame_number)

    def _advance(self, target: dict, needed_classes: list[str], frame_number: int) -> StepEvent:
        event = StepEvent(
            step_index=self.current_step,
            label=target["label"],
            frame_number=frame_number,
            matched=True,
            reason="matched",
        )
        self.events.append(event)
        self.current_step += 1
        self.hold_counter = 0
        for cls in needed_classes:
            self.prev_centers.pop(cls, None)
        return event

    def _check_line_region(self, target: dict, needed_classes: list[str], detections: list[dict]) -> bool:
        """All required classes must have crossed the line THIS frame
        (each tracked via its own prev_centers entry)."""
        coords = target["region_coords"]
        p3, p4 = (coords[0], coords[1]), (coords[2], coords[3])

        all_crossed = True
        for cls in needed_classes:
            best_xyxy, best_conf = None, -1.0
            for det in detections:
                if det["class_name"] != cls:
                    continue
                conf = det.get("conf", 0.0)
                if conf > best_conf:
                    best_conf, best_xyxy = conf, det["xyxy"]

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
