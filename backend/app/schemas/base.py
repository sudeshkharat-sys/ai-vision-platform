from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict
from datetime import datetime


# ── Auth schemas ───────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str = Field(..., min_length=8, max_length=128)

class UserLogin(BaseModel):
    email: str
    password: str = Field(..., min_length=8, max_length=128)

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    token: str
    user: UserResponse

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[str]
    project_type: str = "detection"  # detection | ocr | combined

class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    classes: Optional[List[str]] = None

class ClassRenameRequest(BaseModel):
    old_name: str
    new_name: str

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    classes: List[str]
    project_type: str = "detection"
    created_at: datetime

    class Config:
        from_attributes = True

class ImageResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    filepath: str
    width: int
    height: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class AnnotationCreate(BaseModel):
    image_id: str
    class_name: str
    bbox: Optional[List[float]] = None
    annotation_type: str = "bbox"  # "bbox" | "polygon"
    points: Optional[List[List[float]]] = None  # normalized [[x,y], ...] for polygons
    source: str = "manual"

class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    class_name: str
    bbox: Optional[List[float]]
    annotation_type: str = "bbox"
    points: Optional[List[List[float]]] = None
    source: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Video schemas ───────────────────────────────────────────────────
class VideoResponse(BaseModel):
    id: str
    project_id: str
    original_filename: str
    filepath: str
    file_size: int
    duration: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    total_frames: Optional[int]
    status: str
    frames_extracted: int
    task_id: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class VideoFrameExtractionRequest(BaseModel):
    # Extract 1 frame every `sample_every_n` frames (e.g. 30 = 1 fps at 30fps video)
    sample_every_n: int = Field(default=30, ge=1, le=3000)
    # Hard cap on total frames extracted (0 = no limit)
    max_frames: int = Field(default=300, ge=0, le=10000)


# ── Region sequence schemas ───────────────────────────────────────
class SequenceStep(BaseModel):
    order_index: int
    label: str

    # "region" (default): target = a user-drawn region on the reference
    # frame. "detection_class": target = another detected class's OWN
    # mask (e.g. use "M"'s predicted mask as the target, no drawn region).
    target_type: str = "region"

    # Required when target_type == "region".
    region_type: Optional[str] = None  # "box" | "line"
    region_coords: Optional[List[float]] = None  # normalized [x1, y1, x2, y2]

    # Required when target_type == "detection_class" — the class whose
    # own detected mask becomes this step's target area.
    target_class: Optional[str] = None

    # Kept for backward compatibility with sequences saved before
    # required_classes existed — treated as required_classes[0] if that
    # list is absent.
    required_class: str
    # If set (2+ entries), ALL of these classes must have a detection
    # overlapping the target in the SAME frame for the step to pass —
    # e.g. ["hand", "m"] to require a finger AND the letter "m" both
    # present at once. A single-entry list behaves the same as the old
    # required_class-only steps.
    required_classes: Optional[List[str]] = None


class RegionSequenceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    mode: str = Field(default="strict")  # "strict" | "lenient"
    overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    steps: List[SequenceStep] = Field(default_factory=list)
    # Dimensions of the reference frame regions were drawn against — used
    # at run time to detect a landscape/portrait mismatch vs the video.
    ref_image_width: Optional[int] = None
    ref_image_height: Optional[int] = None


class RegionSequenceUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    mode: Optional[str] = None
    overlap_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    steps: Optional[List[SequenceStep]] = None
    ref_image_width: Optional[int] = None
    ref_image_height: Optional[int] = None


class RegionSequenceResponse(BaseModel):
    id: str
    project_id: str
    name: str
    mode: str
    overlap_threshold: float
    steps: List[SequenceStep]
    ref_image_width: Optional[int] = None
    ref_image_height: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Single-image quick test (no save, no video needed) ────────────
class SequenceTestImageRequest(BaseModel):
    image_id: str
    steps: List[SequenceStep] = Field(..., min_length=1)
    overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SequenceTestClassResult(BaseModel):
    class_name: str
    matched: bool
    best_overlap: float


class SequenceTestStepResult(BaseModel):
    order_index: int
    label: str
    region_type: str
    testable: bool  # False for line regions — no motion history in a single still image
    passed: Optional[bool] = None
    per_class: List[SequenceTestClassResult] = Field(default_factory=list)
    note: Optional[str] = None


class SequenceTestImageResponse(BaseModel):
    image_id: str
    results: List[SequenceTestStepResult]


# ── Sequence run schemas ──────────────────────────────────────────
class SequenceRunStepEvent(BaseModel):
    step_index: int
    label: str
    frame_number: int
    matched: bool
    reason: str
    # Annotated snapshot of the frame at this event (detections + the
    # step's region drawn on it) — lets you SEE what the model saw,
    # not just read a pass/fail log. Absent if the snapshot failed to save.
    frame_url: Optional[str] = None


class SequenceRunResponse(BaseModel):
    id: str
    sequence_id: str
    video_id: str
    status: str
    current_step: int
    total_steps: int
    passed: bool
    step_events: List[SequenceRunStepEvent]
    error: Optional[str]
    # Overwritten continuously while status == "running" — poll this for a
    # near-live view of what the model is currently seeing in the video.
    latest_frame_url: Optional[str] = None
    latest_frame_number: Optional[int] = None
    created_at: datetime
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True
