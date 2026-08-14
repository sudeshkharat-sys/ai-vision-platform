from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sql_delete
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
import os
import re
import io
import json
import uuid
import zipfile
from PIL import Image as PILImage

router = APIRouter(prefix="/images", tags=["images"])


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
    return result.scalars().all()


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
    file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
    if not file_path.exists():
        file_path = settings.upload_dir / image.filepath.replace("/uploads/", "", 1)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        from PIL import ImageOps
        with PILImage.open(file_path) as img:
            # Bake in any EXIF orientation first so the pixels match what
            # the browser has been displaying, then rotate.
            img = ImageOps.exif_transpose(img)
            # PIL's rotate is counter-clockwise; expand=True swaps dimensions
            rotated = img.rotate(-90 if direction == "cw" else 90, expand=True)
            rotated.save(file_path)
            new_w, new_h = rotated.size
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not rotate image: {e}")

    image.width, image.height = new_w, new_h

    # Remap normalized [xc, yc, w, h] boxes into the rotated frame
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    for ann in result.scalars().all():
        if not ann.bbox or len(ann.bbox) != 4:
            continue
        xc, yc, w, h = ann.bbox
        if direction == "cw":
            ann.bbox = [1 - yc, xc, h, w]
        else:
            ann.bbox = [yc, 1 - xc, h, w]

    await db.commit()
    await db.refresh(image)
    return {
        "status": "rotated",
        "id": image_id,
        "direction": direction,
        "width": image.width,
        "height": image.height,
    }


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
            status=entry.get("status", "pending"),
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
