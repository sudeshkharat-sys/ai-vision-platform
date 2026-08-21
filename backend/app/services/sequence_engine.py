"""
sequence_engine.py
~~~~~~~~~~~~~~~~~~~
The per-run state machine: given a saved sequence's ordered steps and a
stream of per-frame detections, tracks how far through the sequence a
single tracked object has progressed.

A step can require MORE THAN ONE class to be present in its region at
once — e.g. ["hand", "m"] to require a finger AND the letter "m" both
detected overlapping the same key region in the same frame, not just
either one alone. This is an AND condition across classes, not tracked
as separate objects with IDs (no ByteTrack yet) — for each required
class, the highest-overlap detection of that class this frame is used.
That keeps this class simple, testable with fake data, and swappable
later for a real multi-object tracker without changing its interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .sequence_geometry import step_matches, box_from_region_coords, overlap_ratio


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


@dataclass
class SequenceRunState:
    steps: list[dict]
    mode: str = "strict"                # "strict" | "lenient"
    overlap_threshold: float = 0.5
    current_step: int = 0
    # Per-class last-seen center point, for line-region crossing checks.
    prev_centers: dict[str, tuple[float, float]] = field(default_factory=dict)
    events: list[StepEvent] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    def current_target(self) -> dict | None:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def process_frame(self, frame_number: int, detections: list[dict]) -> StepEvent | None:
        """detections: [{"class_name": str, "xyxy": (x1,y1,x2,y2)}, ...] for this frame,
        already normalized 0-1. Returns a StepEvent if something notable happened
        (a step matched, or a wrong-region hit in strict mode reset progress).
        """
        if self.is_complete:
            return None

        target = self.current_target()
        needed_classes = _step_classes(target)
        region_xyxy = box_from_region_coords(target["region_coords"]) if target["region_type"] == "box" else None

        # Every required class must independently match this step's region
        # in this same frame — an AND condition, not "any one of them".
        per_class_matched = {}
        for cls in needed_classes:
            best_xyxy, best_score = None, 0.0
            for det in detections:
                if det["class_name"] != cls:
                    continue
                if region_xyxy is not None:
                    score = overlap_ratio(det["xyxy"], region_xyxy)
                    if score > best_score:
                        best_score, best_xyxy = score, det["xyxy"]
                elif best_xyxy is None:
                    best_xyxy = det["xyxy"]  # line region: track for crossing check below

            class_matched = False
            if best_xyxy is not None:
                class_matched = step_matches(target, best_xyxy, self.prev_centers.get(cls), self.overlap_threshold)
                dx1, dy1, dx2, dy2 = best_xyxy
                self.prev_centers[cls] = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
            per_class_matched[cls] = class_matched

        matched = bool(needed_classes) and all(per_class_matched.values())

        if matched:
            event = StepEvent(
                step_index=self.current_step,
                label=target["label"],
                frame_number=frame_number,
                matched=True,
                reason="matched",
            )
            self.events.append(event)
            self.current_step += 1
            # Reset motion history for the classes just used, so the next
            # step's line-region (if any) starts without stale history.
            for cls in needed_classes:
                self.prev_centers.pop(cls, None)
            return event

        # Wrong-region hit: any of this step's required classes has a
        # detection landing in a DIFFERENT step's box region instead.
        wrong_region_hit = False
        for cls in needed_classes:
            for det in detections:
                if det["class_name"] != cls:
                    continue
                for idx, step in enumerate(self.steps):
                    if idx == self.current_step or step["region_type"] != "box":
                        continue
                    other_region = box_from_region_coords(step["region_coords"])
                    if overlap_ratio(det["xyxy"], other_region) >= self.overlap_threshold:
                        wrong_region_hit = True
                        break
                if wrong_region_hit:
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
