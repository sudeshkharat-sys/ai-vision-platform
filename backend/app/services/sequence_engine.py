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
from .sequence_match import evaluate_step, best_polygon_for_class


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


def _uses_crossing(step: dict) -> bool:
    """A line region defaults to motion-crossing (needs two frames, no
    overlap %). If trigger_mode == "overlap" it's evaluated exactly like a
    box instead — same Intersection Area / Target Area x 100 formula and
    the same shared overlap_threshold — via evaluate_step's line handling
    (sequence_match.py buffers the segment into a thin rectangle)."""
    return _is_line_region(step) and step.get("trigger_mode", "crossing") != "overlap"


# How many consecutive dropped-match sampled frames a detect_hold step
# tolerates before treating it as a real release rather than hand jitter.
MISS_TOLERANCE = 2

# Minimum detection confidence required before a live class detection is
# trusted to re-sync a frozen region's position (see _resolve_target).
MIN_RESYNC_CONFIDENCE = 0.4


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
    # Per-step-index drift-corrected polygon for a frozen "polygon"
    # region step that also carries target_class (see _resolve_target).
    synced_polygons: dict[int, list] = field(default_factory=dict)
    # Consecutive dropped-match sampled frames for the current detect_hold
    # step — see MISS_TOLERANCE.
    miss_counter: int = 0

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def current_target(self) -> dict | None:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def _resolve_target(self, step: dict, step_index: int, detections: list[dict], allow_resync: bool = True) -> dict:
        """A frozen "polygon" region step that also has target_class set
        (e.g. from "Freeze boundary") gets re-synced to that class's own
        live-detected mask whenever it's actually visible this frame —
        self-correcting for camera shake/drift instead of staying stuck
        at the exact spot it was frozen at. Falls back to the last
        known-good polygon (or the original frozen one) whenever the
        class is occluded/not detected this frame, same as before.

        Only resyncs on a detection with at least MIN_RESYNC_CONFIDENCE
        confidence — two adjacent keys (e.g. N sitting right under I) can
        make the model emit a shaky, low-confidence, wrongly-placed guess
        when a hand overlaps both; trusting that would corrupt an
        otherwise-good synced position instead of protecting it.

        allow_resync=False keeps using whatever position is already
        recorded (frozen or last-synced) WITHOUT updating it this frame —
        used by detect_hold mid-hold (hold_counter > 0) so the target
        can't wobble frame-to-frame from live-detection jitter while a
        press is already in progress, which would otherwise break a
        genuinely still hand out of a hold it should be completing."""
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

        if _uses_crossing(target):
            matched = self._check_line_region(target, needed_classes, detections)
        else:
            resolved = self._resolve_target(target, self.current_step, detections)
            result = evaluate_step(resolved, detections, threshold_pct)
            matched = result["matched"]

        if matched:
            return self._advance(target, needed_classes, frame_number)

        # Wrong-region hit: any of this step's required classes has a
        # detection landing in a step further AHEAD in the sequence
        # instead — a genuine mistake (touching a not-yet-reached
        # target early). Steps already PASSED (idx < current_step) are
        # never checked: the hand naturally lingers over the region it
        # just completed for a frame or two while moving toward the
        # next one, and that's not a mistake, just physics — flagging it
        # anyway sent progress back to step 1 the instant a step passed.
        # (Only checked for non-line steps — a line region's "hit" is a
        # crossing event, not a static landing.)
        wrong_region_hit = False
        for cls in needed_classes:
            for idx, step in enumerate(self.steps):
                if idx <= self.current_step or _uses_crossing(step):
                    continue
                other_needed = _step_classes(step)
                if cls not in other_needed:
                    continue
                other_resolved = self._resolve_target(step, idx, detections)
                other_result = evaluate_step(other_resolved, detections, threshold_pct)
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
            self.synced_polygons = {}
            self.hold_counter = 0
            self.miss_counter = 0
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
        hold_frames sampled frames IN A ROW — with a small tolerance for
        a hand naturally jittering during a long hold (MISS_TOLERANCE
        consecutive dropped-match frames are absorbed rather than wiping
        out the hold), so a real release still resets it but a brief
        blip doesn't. This is what tells apart a real 'press and hold'
        from the hand just passing over the region on its way
        somewhere else."""
        if _uses_crossing(target):
            matched_now = self._check_line_region(target, needed_classes, detections)
        else:
            resolved = self._resolve_target(target, self.current_step, detections, allow_resync=(self.hold_counter == 0))
            result = evaluate_step(resolved, detections, threshold_pct)
            matched_now = result["matched"]

        if not matched_now:
            self.miss_counter += 1
            if self.miss_counter >= MISS_TOLERANCE:
                self.hold_counter = 0
                self.miss_counter = 0
            return None

        self.miss_counter = 0
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
        self.miss_counter = 0
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
