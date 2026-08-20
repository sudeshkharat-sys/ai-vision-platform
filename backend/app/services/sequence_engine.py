"""
sequence_engine.py
~~~~~~~~~~~~~~~~~~~
The per-run state machine: given a saved sequence's ordered steps and a
stream of per-frame detections, tracks how far through the sequence a
single tracked object has progressed.

This module does NOT track objects across frames (no ByteTrack) — the
caller is expected to already know which detection is "the" object being
followed for this run (e.g. the highest-confidence detection of the
current step's required class). That keeps this class simple, testable
with fake data, and swappable later for a real multi-object tracker
without changing its interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .sequence_geometry import step_matches


@dataclass
class StepEvent:
    step_index: int
    label: str
    frame_number: int
    matched: bool
    reason: str  # "matched" | "wrong_region_reset" | "wrong_region_ignored"


@dataclass
class SequenceRunState:
    steps: list[dict]
    mode: str = "strict"                # "strict" | "lenient"
    overlap_threshold: float = 0.5
    current_step: int = 0
    prev_center: tuple[float, float] | None = None
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
        required_class = target["required_class"]

        # Best candidate: highest-overlap detection of the required class.
        best_xyxy, best_score = None, 0.0
        wrong_region_hit = False

        for det in detections:
            if det["class_name"] != required_class:
                continue
            from .sequence_geometry import box_from_region_coords, overlap_ratio
            region_xyxy = box_from_region_coords(target["region_coords"]) if target["region_type"] == "box" else None

            if region_xyxy is not None:
                score = overlap_ratio(det["xyxy"], region_xyxy)
                if score > best_score:
                    best_score, best_xyxy = score, det["xyxy"]

        matched = False
        if best_xyxy is not None:
            matched = step_matches(target, best_xyxy, self.prev_center, self.overlap_threshold)

        # Track motion center for line-region crossing checks next frame,
        # using whichever detection of the required class we saw (even if no match).
        for det in detections:
            if det["class_name"] == required_class:
                dx1, dy1, dx2, dy2 = det["xyxy"]
                self.prev_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                break

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
            self.prev_center = None  # reset motion history for the next step's region
            return event

        # Check if a required-class detection landed in a DIFFERENT step's region
        # (wrong-region hit) — only meaningful for box regions, where "landed in"
        # is well defined without motion history.
        for det in detections:
            if det["class_name"] != required_class:
                continue
            for idx, step in enumerate(self.steps):
                if idx == self.current_step or step["region_type"] != "box":
                    continue
                from .sequence_geometry import box_from_region_coords, overlap_ratio
                region_xyxy = box_from_region_coords(step["region_coords"])
                if overlap_ratio(det["xyxy"], region_xyxy) >= self.overlap_threshold:
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
            self.prev_center = None
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
