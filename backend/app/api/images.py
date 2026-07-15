from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sql_delete
from ..database import get_db
from ..models.image import Image
from ..models.annotation import Annotation
from ..models.user import User
from ..schemas.base import ImageResponse
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_image
from typing import List
import os
import uuid
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
