import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, JSON
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

    # Ordered list of steps:
    # [{ "order_index": 0, "label": "Middle Socket", "region_type": "box"|"line",
    #    "region_coords": [x1,y1,x2,y2] (normalized 0-1), "required_class": "bulb" }, ...]
    steps: Mapped[list[dict]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="region_sequences")
