from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete as sql_delete
from ..database import get_db
from ..models.project import Project
from ..models.image import Image
from ..models.annotation import Annotation
from ..models.user import User
from ..schemas.base import ImageResponse
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_image
from typing import List
from pydantic import BaseModel
import os
import re
import io
import json
import uuid
import zipfile
from PIL import Image as PILImage

router = APIRouter(prefix="/images", tags=["images"])

# Formats browsers decode natively. PIL reads many more (TIFF, PPM, ...),
# but a photo that cannot be shown in a browser cannot be annotated, so
# uploads are limited to these — see upload_images.
BROWSER_DISPLAYABLE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP"}


@router.post("/upload/{project_id}")
async def upload_images(
    project_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    project_dir = settings.upload_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded_images = []
    failed_files = []

    for file in files:
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = project_dir / unique_filename

        try:
            contents = await file.read()
            if not contents:
                raise ValueError("File is empty")

            with open(file_path, "wb") as buffer:
                buffer.write(contents)

            with PILImage.open(file_path) as img:
                # PILImage.open() only reads the header, so a truncated or
                # corrupt file gets accepted here and then renders blank on
                # the annotation canvas with nothing to explain why. load()
                # forces the actual decode, failing now instead of silently.
                img.load()

                # Likewise, PIL reads plenty of formats browsers cannot
                # display (TIFF, CMYK JPEG, ...). Annotating happens in a
                # browser, so accepting one just produces a blank photo.
                if img.format not in BROWSER_DISPLAYABLE_FORMATS:
                    raise ValueError(
                        f"{img.format} images can't be displayed by browsers — "
                        "convert to JPEG or PNG first"
                    )
                if img.mode == "CMYK":
                    raise ValueError(
                        "CMYK images render black in most browsers — "
                        "convert to RGB JPEG or PNG first"
                    )

                width, height = img.size
                try:
                    exif_data = img._getexif()
                    if exif_data:
                        orientation = exif_data.get(274, 1)
                        if orientation in (5, 6, 7, 8):
                            width, height = height, width
                except Exception:
                    pass

            db_image = Image(
                project_id=project_id,
                filename=file.filename,
                filepath=f"/uploads/{project_id}/{unique_filename}",
                width=width,
                height=height,
            )
            db.add(db_image)
            uploaded_images.append(db_image)

        except Exception as e:
            # Clean up the bad file from disk if it was written
            if file_path.exists():
                file_path.unlink()
            failed_files.append({"filename": file.filename, "reason": str(e)})

    if uploaded_images:
        await db.commit()
        for img in uploaded_images:
            await db.refresh(img)

    return {
        "uploaded": uploaded_images,
        "failed": failed_files,
    }


@router.get("/project/{project_id}", response_model=List[ImageResponse])
async def list_project_images(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(select(Image).where(Image.project_id == project_id))
    images = result.scalars().all()
    rows = (await db.execute(
        select(Annotation.image_id, Annotation.annotation_type, Annotation.bbox)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
    )).all()
    dims = {img.id: (img.width or 1, img.height or 1) for img in images}
    counts, shapes, longest = {}, {}, {}
    for image_id, atype, bbox in rows:
        atype = atype or "bbox"
        counts[image_id] = counts.get(image_id, 0) + 1
        sh = shapes.setdefault(image_id, {})
        sh[atype] = sh.get(atype, 0) + 1
        if atype == "bbox" and bbox and len(bbox) == 4:
            w, h = dims[image_id]
            bw, bh = bbox[2] * w, bbox[3] * h          # box size in pixels
            if min(bw, bh) > 0:
                longest[image_id] = max(longest.get(image_id, 0), max(bw, bh) / min(bw, bh))
    for img in images:
        img.ann_count = counts.get(img.id, 0)      # plain attributes, read by ImageResponse
        img.ann_shapes = shapes.get(img.id, {})
        img.ann_max_ratio = round(longest.get(img.id, 0), 2)
    return images


@router.patch("/{image_id}/mark-empty", response_model=ImageResponse)
async def mark_image_empty(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an image as annotated with no objects (negative/background frame).
    The image status is set to 'annotated' with zero annotation rows.
    YOLO training will generate an empty label file for it, which is valid
    and teaches the model to suppress false positives on blank frames.
    """
    image = await get_owned_image(image_id, current_user, db)
    image.status = "annotated"
    await db.commit()
    await db.refresh(image)
    return image


def _rotate_file_and_boxes(file_path, annotations, angle_deg_cw: float):
    """
    Rotate the image file on disk by an arbitrary angle (degrees, clockwise)
    with the canvas expanded so nothing is cropped, then remap every
    normalized [xc, yc, w, h] bbox into the new frame.

    For non-90° angles an axis-aligned box can only approximate the rotated
    region, so each box becomes the axis-aligned hull of its rotated corners.
    Returns (new_width, new_height).
    """
    import math
    from PIL import ImageOps

    with PILImage.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        old_w, old_h = img.size
        # PIL rotates counter-clockwise for positive angles
        rotated = img.rotate(
            -angle_deg_cw, expand=True, resample=PILImage.BICUBIC,
            fillcolor=(0, 0, 0) if img.mode == "RGB" else 0,
        )
        rotated.save(file_path)
        new_w, new_h = rotated.size

    # Visual CCW rotation of the image by theta maps a pixel (x, y) —
    # y-axis pointing down — to:
    #   x' =  (x-cx)·cosθ + (y-cy)·sinθ + cx'
    #   y' = -(x-cx)·sinθ + (y-cy)·cosθ + cy'
    theta = math.radians(-angle_deg_cw)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = old_w / 2.0, old_h / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0

    for ann in annotations:
        if not ann.bbox or len(ann.bbox) != 4:
            continue
        xc, yc, w, h = ann.bbox
        px, py, pw, ph = xc * old_w, yc * old_h, w * old_w, h * old_h
        corners = [
            (px - pw / 2, py - ph / 2), (px + pw / 2, py - ph / 2),
            (px - pw / 2, py + ph / 2), (px + pw / 2, py + ph / 2),
        ]
        xs, ys = [], []
        for x, y in corners:
            xs.append((x - cx) * cos_t + (y - cy) * sin_t + ncx)
            ys.append(-(x - cx) * sin_t + (y - cy) * cos_t + ncy)
        x0, x1 = max(0.0, min(xs)), min(float(new_w), max(xs))
        y0, y1 = max(0.0, min(ys)), min(float(new_h), max(ys))
        ann.bbox = [
            (x0 + x1) / 2 / new_w, (y0 + y1) / 2 / new_h,
            (x1 - x0) / new_w, (y1 - y0) / new_h,
        ]

    return new_w, new_h


def _detect_text_skew(file_path) -> float:
    """
    Estimate the dominant text/edge skew angle in degrees (clockwise-positive,
    i.e. the amount the content is tilted CW from horizontal). Returns 0.0
    when no confident estimate is found.

    Works on engraved/stamped plates: edge map -> probabilistic Hough lines ->
    length-weighted median of near-horizontal segment angles.
    """
    import math
    import cv2
    import numpy as np

    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    # Downscale big photos so Hough stays fast and noise-tolerant
    scale = 1200.0 / max(img.shape) if max(img.shape) > 1200 else 1.0
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150)
    min_len = max(20, int(min(img.shape) * 0.10))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=min_len, maxLineGap=8,
    )
    if lines is None:
        return 0.0

    weighted = []  # (angle_deg, length)
    for line in lines:
        x1, y1, x2, y2 = np.ravel(line)[:4]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        # Fold into [-45, 45): treat vertical strokes as their horizontal complement
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        weighted.append((angle, length))

    if not weighted:
        return 0.0

    # Length-weighted median angle
    weighted.sort(key=lambda t: t[0])
    total = sum(l for _, l in weighted)
    acc = 0.0
    skew = 0.0
    for angle, length in weighted:
        acc += length
        if acc >= total / 2:
            skew = angle
            break

    # Sub-degree tilts aren't worth resampling; huge estimates are unreliable
    if abs(skew) < 0.3 or abs(skew) > 44.0:
        return 0.0
    return skew


def _image_file(image):
    file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
    if not file_path.exists():
        file_path = settings.upload_dir / image.filepath.replace("/uploads/", "", 1)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return file_path


def _file_size(file_path):
    """(width, height) as a browser shows the file (EXIF orientation applied)."""
    with PILImage.open(file_path) as img:
        w, h = img.size
        if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):
            w, h = h, w
    return w, h


async def _rotate_image_obj(image, direction: str, db: AsyncSession, file: bool = True, labels: bool = True):
    """Rotate the image file 90° on disk and/or remap its boxes / polygon
    points (``file`` / ``labels`` pick which). Does not commit."""
    file_path = _image_file(image)

    if file:
        try:
            from PIL import ImageOps
            with PILImage.open(file_path) as img:
                # Bake in any EXIF orientation first so the pixels match what
                # the browser has been displaying, then rotate.
                img = ImageOps.exif_transpose(img)
                # PIL's rotate is counter-clockwise; expand=True swaps dimensions
                rotated = img.rotate(-90 if direction == "cw" else 90, expand=True)
                rotated.save(file_path)
                image.width, image.height = rotated.size
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not rotate image: {e}")
    else:
        image.width, image.height = _file_size(file_path)   # trust the real file

    if not labels:
        return
    # Remap normalized [xc, yc, w, h] boxes (and polygon / segment points) into the rotated frame
    result = await db.execute(select(Annotation).where(Annotation.image_id == image.id))
    for ann in result.scalars().all():
        if ann.bbox and len(ann.bbox) == 4:
            xc, yc, w, h = ann.bbox
            ann.bbox = [1 - yc, xc, h, w] if direction == "cw" else [yc, 1 - xc, h, w]
        if ann.points:
            ann.points = [[1 - y, x] if direction == "cw" else [y, 1 - x] for x, y in ann.points]


@router.get("/project/{project_id}/size-mismatches")
async def size_mismatches(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Images whose real file shape differs from the stored width/height.
    ``swapped`` = the file is landscape/portrait the other way round than the
    database says - what you get when labels were rotated but the picture was not."""
    await get_owned_project(project_id, current_user, db)
    imgs = (await db.execute(select(Image).where(Image.project_id == project_id))).scalars().all()
    swapped, other, missing = [], [], 0
    for im in imgs:
        try:
            fw, fh = _file_size(_image_file(im))
        except Exception:
            missing += 1
            continue
        if (fw, fh) == (im.width, im.height):
            continue
        row = {"id": im.id, "filename": im.filename, "db": [im.width, im.height], "file": [fw, fh]}
        (swapped if (fw, fh) == (im.height, im.width) else other).append(row)
    return {"checked": len(imgs), "swapped": swapped, "other": other, "missing_files": missing}


class IdsBody(BaseModel):
    image_ids: List[str]


@router.post("/project/{project_id}/reset-empty")
async def reset_empty_status(
    project_id: str,
    body: IdsBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Of the listed images, set those with NO annotations back to 'pending'
    (repairs images left 'annotated' after their boxes were deleted)."""
    await get_owned_project(project_id, current_user, db)
    imgs = (await db.execute(
        select(Image).where(Image.project_id == project_id, Image.id.in_(body.image_ids),
                            Image.status != "pending"))).scalars().all()
    has_ann = set()
    if imgs:
        rows = await db.execute(select(Annotation.image_id).where(Annotation.image_id.in_([i.id for i in imgs])).distinct())
        has_ann = {r[0] for r in rows.all()}
    reset = [i for i in imgs if i.id not in has_ann]
    for i in reset:
        i.status = "pending"
    await db.commit()
    return {"reset": [i.id for i in reset]}


class FixSizeBody(BaseModel):
    image_ids: List[str]
    mode: str       # "rotate_files": rotate the pictures clockwise to match the (already rotated) labels
                    # "revert_labels": rotate the labels back counter-clockwise to match the pictures


@router.post("/project/{project_id}/fix-size-mismatch")
async def fix_size_mismatch(
    project_id: str,
    body: FixSizeBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.mode not in ("rotate_files", "revert_labels"):
        raise HTTPException(status_code=400, detail="mode must be 'rotate_files' or 'revert_labels'")
    await get_owned_project(project_id, current_user, db)
    imgs = (await db.execute(
        select(Image).where(Image.project_id == project_id, Image.id.in_(body.image_ids)))).scalars().all()
    fixed, failed = [], []
    for im in imgs:
        try:
            fw, fh = _file_size(_image_file(im))
            if (fw, fh) != (im.height, im.width):      # only the clean "swapped" case is touched
                failed.append({"id": im.id, "filename": im.filename, "error": "not a simple rotated-labels mismatch"})
                continue
            if body.mode == "rotate_files":
                await _rotate_image_obj(im, "cw", db, file=True, labels=False)
            else:
                await _rotate_image_obj(im, "ccw", db, file=False, labels=True)
            await db.commit()
            fixed.append({"id": im.id, "width": im.width, "height": im.height})
        except HTTPException as e:
            await db.rollback()
            failed.append({"id": im.id, "filename": im.filename, "error": e.detail})
    return {"fixed": fixed, "failed": failed}


@router.post("/{image_id}/rotate")
async def rotate_image(
    image_id: str,
    direction: str = "cw",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Rotate the image file 90° on disk ('cw' or 'ccw') and remap every
    annotation bbox so labels stay on their characters. Useful when a
    plate was photographed in portrait but is easier to label landscape.
    """
    if direction not in ("cw", "ccw"):
        raise HTTPException(status_code=400, detail="direction must be 'cw' or 'ccw'")

    image = await get_owned_image(image_id, current_user, db)
    await _rotate_image_obj(image, direction, db)
    await db.commit()
    await db.refresh(image)
    return {
        "status": "rotated",
        "id": image_id,
        "direction": direction,
        "width": image.width,
        "height": image.height,
    }


class BulkRotateBody(BaseModel):
    image_ids: List[str]
    direction: str = "cw"


@router.post("/project/{project_id}/rotate-bulk")
async def rotate_images_bulk(
    project_id: str,
    body: BulkRotateBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rotate exactly the listed images of a project 90° (boxes and polygon
    points remapped). Each image is committed on its own, so one bad file
    does not undo the others; failures are reported."""
    if body.direction not in ("cw", "ccw"):
        raise HTTPException(status_code=400, detail="direction must be 'cw' or 'ccw'")
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(
        select(Image).where(Image.project_id == project_id, Image.id.in_(body.image_ids))
    )
    rotated, failed = [], []
    for image in result.scalars().all():
        try:
            await _rotate_image_obj(image, body.direction, db)
            await db.commit()
            rotated.append({"id": image.id, "width": image.width, "height": image.height})
        except HTTPException as e:
            await db.rollback()
            failed.append({"id": image.id, "filename": image.filename, "error": e.detail})
    return {"rotated": rotated, "failed": failed}


class CopyToProjectBody(BaseModel):
    image_ids: List[str]
    new_name: str = ""
    target_project_id: str = ""
    rotate: str = ""            # "", "cw" or "ccw": rotate the COPIES 90deg (boxes / polygons move with them)


@router.post("/project/{project_id}/copy-to-project")
async def copy_images_to_project(
    project_id: str,
    body: CopyToProjectBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Copy the listed images (file + all boxes/polygons) into a new project
    (``new_name``) or an existing one (``target_project_id``). The source is
    left untouched; classes are added to the target."""
    import shutil
    from ..services.image_files import resolve_image_file
    if body.rotate not in ("", "cw", "ccw"):
        raise HTTPException(status_code=422, detail="rotate must be '', 'cw' or 'ccw'")
    source = await get_owned_project(project_id, current_user, db)
    if body.target_project_id:
        if body.target_project_id == project_id:
            raise HTTPException(status_code=422, detail="Pick a different project")
        target = await get_owned_project(body.target_project_id, current_user, db)
    elif body.new_name.strip():
        target = Project(name=body.new_name.strip()[:250], description=source.description, classes=[],
                         project_type=source.project_type, segments=list(source.segments or []),
                         user_id=current_user.id)
        db.add(target)
        await db.flush()
    else:
        raise HTTPException(status_code=422, detail="Choose a target project or a new project name")

    imgs = (await db.execute(select(Image).where(Image.project_id == project_id, Image.id.in_(body.image_ids)))).scalars().all()
    anns = {}
    for a in (await db.execute(select(Annotation).where(Annotation.image_id.in_([i.id for i in imgs])))).scalars().all():
        anns.setdefault(a.image_id, []).append(a)
    dest_dir = settings.upload_dir / target.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    classes, copied, missing = list(target.classes or []), 0, 0
    for img in imgs:
        src = resolve_image_file(project_id, img.filepath)
        if src is None:
            missing += 1
            continue
        name = f"{uuid.uuid4()}{src.suffix}"
        shutil.copy2(src, dest_dir / name)
        new_img = Image(project_id=target.id, filename=img.filename, filepath=f"/uploads/{target.id}/{name}",
                        width=img.width, height=img.height, status=img.status)
        db.add(new_img)
        await db.flush()
        for a in anns.get(img.id, []):
            db.add(Annotation(image_id=new_img.id, class_name=a.class_name,
                              bbox=list(a.bbox) if a.bbox else None, source=a.source,
                              annotation_type=a.annotation_type, points=list(a.points) if a.points else None,
                              state=a.state))
            if a.class_name and a.class_name not in classes:
                classes.append(a.class_name)
        if body.rotate:
            await db.flush()
            await _rotate_image_obj(new_img, body.rotate, db)    # only the copy's file; the source stays as is
        copied += 1
    target.classes = classes
    await db.commit()
    return {"target_project_id": target.id, "target_name": target.name, "copied": copied, "missing_files": missing}


@router.post("/{image_id}/rotate-fine")
async def rotate_image_fine(
    image_id: str,
    angle: float,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Rotate the image by an arbitrary angle in degrees (positive = clockwise)
    so a tilted plate can be leveled before drawing character boxes.
    Existing bboxes are remapped to the axis-aligned hull of their rotated
    corners — best used before annotating.
    """
    if not (-180.0 <= angle <= 180.0) or angle == 0:
        raise HTTPException(status_code=400, detail="angle must be non-zero, within ±180 degrees")

    image = await get_owned_image(image_id, current_user, db)
    file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
    if not file_path.exists():
        file_path = settings.upload_dir / image.filepath.replace("/uploads/", "", 1)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    anns = result.scalars().all()
    try:
        new_w, new_h = _rotate_file_and_boxes(file_path, anns, angle)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not rotate image: {e}")

    image.width, image.height = new_w, new_h
    await db.commit()
    await db.refresh(image)
    return {
        "status": "rotated",
        "id": image_id,
        "angle": angle,
        "width": image.width,
        "height": image.height,
    }


@router.post("/{image_id}/auto-straighten")
async def auto_straighten_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Smart rotate: detect the dominant tilt of the engraved text/edges and
    level the image automatically. Returns the correction angle applied
    (0 when the image already looks straight or no confident estimate).
    """
    image = await get_owned_image(image_id, current_user, db)
    file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
    if not file_path.exists():
        file_path = settings.upload_dir / image.filepath.replace("/uploads/", "", 1)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        skew = _detect_text_skew(file_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Skew detection failed: {e}")

    if skew == 0.0:
        return {
            "status": "already_straight",
            "id": image_id,
            "angle": 0.0,
            "width": image.width,
            "height": image.height,
        }

    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    anns = result.scalars().all()
    try:
        # Content is tilted `skew` degrees CW -> correct by rotating CCW
        new_w, new_h = _rotate_file_and_boxes(file_path, anns, -skew)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not rotate image: {e}")

    image.width, image.height = new_w, new_h
    await db.commit()
    await db.refresh(image)
    return {
        "status": "rotated",
        "id": image_id,
        "angle": -skew,
        "width": image.width,
        "height": image.height,
    }


@router.post("/{image_id}/wipe")
async def wipe_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Clear all annotations from an image and reset its status to pending."""
    image = await get_owned_image(image_id, current_user, db)
    await db.execute(sql_delete(Annotation).where(Annotation.image_id == image_id))
    image.status = "pending"
    await db.commit()
    return {"status": "wiped", "id": image_id}


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an image, all its annotations, and the file from disk."""
    image = await get_owned_image(image_id, current_user, db)

    # Delete all annotations for this image
    await db.execute(sql_delete(Annotation).where(Annotation.image_id == image_id))

    # Remove the file from disk
    try:
        file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

    # Delete the image record
    await db.delete(image)
    await db.commit()
    return {"status": "deleted", "id": image_id}


# ── Dataset export / import ──────────────────────────────────────────
#
# Portable zip = images/ folder + manifest.json describing every image's
# annotations (all annotation_type values: bbox, polygon, segment). This
# lets a labeled dataset move between projects -- e.g. export a project's
# annotated plates and import them into a separate SAM-based project --
# without re-uploading images or redrawing any boxes/masks/polygons.

def _project_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_").lower()
    return slug or "project"


@router.get("/export/{project_id}")
async def export_dataset(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download this project's images + all annotations as a single zip."""
    project = await get_owned_project(project_id, current_user, db)

    result = await db.execute(select(Image).where(Image.project_id == project_id))
    images = result.scalars().all()
    if not images:
        raise HTTPException(status_code=404, detail="This project has no images to export.")

    image_ids = [img.id for img in images]
    ann_result = await db.execute(select(Annotation).where(Annotation.image_id.in_(image_ids)))
    anns_by_image: dict = {}
    for ann in ann_result.scalars().all():
        anns_by_image.setdefault(ann.image_id, []).append({
            "class_name": ann.class_name,
            "bbox": ann.bbox,
            "annotation_type": ann.annotation_type,
            "points": ann.points,
            "state": ann.state,
            "source": ann.source,
        })

    manifest = {
        "project_name": project.name,
        "project_type": project.project_type,
        "classes": project.classes or [],
        "images": [],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            src_candidates = [
                settings.upload_dir.parent / img.filepath.lstrip("/"),
                settings.upload_dir / img.filepath.replace("/uploads/", "", 1),
            ]
            src_path = next((p for p in src_candidates if p.exists()), None)
            if src_path is None:
                continue  # file missing on disk — skip but keep the rest of the export usable

            arc_filename = os.path.basename(img.filepath)
            zf.write(src_path, f"images/{arc_filename}")

            manifest["images"].append({
                "filename": img.filename,
                "stored_filename": arc_filename,
                "width": img.width,
                "height": img.height,
                "status": img.status,
                "annotations": anns_by_image.get(img.id, []),
            })

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    buffer.seek(0)
    slug = _project_slug(project.name)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug}_dataset.zip"'},
    )


@router.post("/import/{project_id}")
async def import_dataset(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Import a zip produced by /images/export/{project_id} into this project.

    Images are copied in under fresh filenames (no collision with anything
    already in this project) and every annotation -- bbox, precision
    polygon, or real segment mask -- is recreated as-is. New class names
    found in the manifest are merged into the project's class list.
    """
    project = await get_owned_project(project_id, current_user, db)

    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    contents = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(contents))
        manifest = json.loads(zf.read("manifest.json"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset zip: {e}")

    project_dir = settings.upload_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    existing_classes = list(project.classes or [])
    imported_images = []
    imported_annotations = 0
    skipped = []

    for entry in manifest.get("images", []):
        stored_name = entry.get("stored_filename")
        arcname = f"images/{stored_name}"
        if not stored_name or arcname not in zf.namelist():
            skipped.append(entry.get("filename", stored_name or "?"))
            continue

        file_ext = os.path.splitext(stored_name)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        dest_path = project_dir / unique_filename
        with zf.open(arcname) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())

        db_image = Image(
            project_id=project_id,
            filename=entry.get("filename", stored_name),
            filepath=f"/uploads/{project_id}/{unique_filename}",
            width=entry.get("width") or 0,
            height=entry.get("height") or 0,
            # Boxes make an image annotated, whatever status the exporting project
            # had stored; training and the stats only count "annotated" images.
            status="annotated" if entry.get("annotations") else entry.get("status", "pending"),
        )
        db.add(db_image)
        await db.flush()  # assigns db_image.id for its annotations below
        imported_images.append(db_image)

        for ann in entry.get("annotations", []):
            class_name = ann.get("class_name", "")
            if class_name and class_name not in existing_classes:
                existing_classes.append(class_name)
            db.add(Annotation(
                image_id=db_image.id,
                class_name=class_name,
                bbox=ann.get("bbox"),
                annotation_type=ann.get("annotation_type", "bbox"),
                points=ann.get("points"),
                state=ann.get("state"),
                source=ann.get("source", "manual"),
            ))
            imported_annotations += 1

    project.classes = existing_classes
    await db.commit()
    for img in imported_images:
        await db.refresh(img)

    return {
        "status": "imported",
        "images_imported": len(imported_images),
        "annotations_imported": imported_annotations,
        "skipped": skipped,
        "classes": existing_classes,
    }
