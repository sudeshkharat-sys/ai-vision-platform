import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, JSON, Float, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..database import Base


class RegionSequence(Base):
    """An ordered set of regions an object must pass through, in order, to
    complete a "sequence" (e.g. a bulb visiting socket 1 -> 2 -> 3, or a
    finger visiting key-zones S -> U -> D -> E -> S -> H).
    """
    __tablename__ = "region_sequences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # "strict" (wrong-region hit resets progress) or "lenient" (wrong-region hit is ignored)
    mode: Mapped[str] = mapped_column(String(20), default="strict")
    # Fraction of the detected object's box that must overlap a box region to count as "on it"
    overlap_threshold: Mapped[float] = mapped_column(Float, default=0.5)

    # Ordered list of steps:
    # [{ "order_index": 0, "label": "Middle Socket", "region_type": "box"|"line",
    #    "region_coords": [x1,y1,x2,y2] (normalized 0-1), "required_class": "bulb" }, ...]
    steps: Mapped[list[dict]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="region_sequences")


class SequenceRun(Base):
    """One attempt at running a saved RegionSequence against a video —
    tracks how far the state machine got and the per-step events it saw.
    """
    __tablename__ = "sequence_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sequence_id: Mapped[str] = mapped_column(String(36), ForeignKey("region_sequences.id", ondelete="CASCADE"), nullable=False)
    video_id: Mapped[str] = mapped_column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)

    # pending | running | complete | failed | error
    status: Mapped[str] = mapped_column(String(20), default="pending")
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)

    # [{ step_index, label, frame_number, matched, reason }, ...]
    step_events: Mapped[list[dict]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Continuously overwritten while the run is in progress, so the frontend
    # can poll and show a near-live "what the model is seeing right now" view.
    latest_frame_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    latest_frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
