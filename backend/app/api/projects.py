from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from ..database import get_db
from ..models.project import Project
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..schemas.base import ProjectCreate, ProjectResponse, ProjectUpdateRequest, ClassRenameRequest
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from typing import List, Dict
from pathlib import Path
import shutil
import uuid

router = APIRouter(prefix="/projects", tags=["projects"])




@router.post("", response_model=ProjectResponse)
async def create_project(
    data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if data.project_type not in ("detection", "ocr"):
        raise HTTPException(status_code=400, detail="project_type must be 'detection' or 'ocr'")

    classes = data.classes
    if data.project_type == "ocr" and not classes:
        # Pre-fill the full character set so labeling boxes is one click
        classes = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]

    project = Project(
        name=data.name,
        description=data.description,
        classes=classes,
        project_type=data.project_type,
        user_id=current_user.id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.user_id == current_user.id))
    return result.scalars().all()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_owned_project(project_id, current_user, db)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update project name, description, or classes list."""
    project = await get_owned_project(project_id, current_user, db)

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.classes is not None:
        project.classes = data.classes

    await db.commit()
    await db.refresh(project)
    return project


@router.post("/{project_id}/duplicate", response_model=ProjectResponse)
async def duplicate_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Copy a project's images, classes, and annotations into a brand new
    project, so labels can be edited/renamed/deleted there without
    touching the original while it's still in use. Trained model files
    and training-job history are deliberately NOT copied -- the new
    project starts untrained, same as any freshly created one, and needs
    its own training run once its labels are the way you want them.
    """
    source = await get_owned_project(project_id, current_user, db)

    existing_result = await db.execute(
        select(Project.name).where(
            Project.user_id == current_user.id,
            Project.name.like(f"{source.name} (Copy%"),
        )
    )
    existing_names = set(existing_result.scalars().all())

    new_name = f"{source.name} (Copy)"
    suffix = 2
    while new_name in existing_names:
        new_name = f"{source.name} (Copy {suffix})"
        suffix += 1

    new_project = Project(
        name=new_name,
        description=source.description,
        classes=list(source.classes or []),
        project_type=source.project_type,
        user_id=current_user.id,
    )
    db.add(new_project)
    await db.flush()  # assigns new_project.id for the images/annotations below

    result = await db.execute(select(Image).where(Image.project_id == project_id))
    source_images = result.scalars().all()

    if source_images:
        image_ids = [img.id for img in source_images]
        ann_result = await db.execute(select(Annotation).where(Annotation.image_id.in_(image_ids)))
        anns_by_image: Dict[str, List[Annotation]] = {}
        for ann in ann_result.scalars().all():
            anns_by_image.setdefault(ann.image_id, []).append(ann)

        dest_dir = settings.upload_dir / new_project.id
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img in source_images:
            # filepath is stored as "/uploads/{project_id}/{filename}", not
            # an absolute path -- resolve back to the real file on disk via
            # the upload_dir the same way upload_images() writes it.
            src_file = settings.upload_dir / project_id / Path(img.filepath).name
            new_filename = f"{uuid.uuid4()}{src_file.suffix}"
            dest_file = dest_dir / new_filename
            if src_file.exists():
                shutil.copy2(src_file, dest_file)

            new_image = Image(
                project_id=new_project.id,
                filename=img.filename,
                filepath=f"/uploads/{new_project.id}/{new_filename}",
                width=img.width,
                height=img.height,
                status=img.status,
            )
            db.add(new_image)
            await db.flush()  # assigns new_image.id for its annotations below

            for ann in anns_by_image.get(img.id, []):
                db.add(Annotation(
                    image_id=new_image.id,
                    class_name=ann.class_name,
                    bbox=list(ann.bbox) if ann.bbox else None,
                    source=ann.source,
                ))

    await db.commit()
    await db.refresh(new_project)
    return new_project


@router.patch("/{project_id}/rename-class")
async def rename_class(
    project_id: str,
    data: ClassRenameRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename a class in project.classes AND in all existing annotations."""
    project = await get_owned_project(project_id, current_user, db)

    # Replace in the classes list
    updated_classes = [
        data.new_name if c == data.old_name else c
        for c in project.classes
    ]
    project.classes = updated_classes

    # Bulk-update annotation class_name via image subquery
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    await db.execute(
        update(Annotation)
        .where(
            Annotation.image_id.in_(image_ids_subq),
            Annotation.class_name == data.old_name,
        )
        .values(class_name=data.new_name)
    )

    await db.commit()
    return {
        "updated_classes": updated_classes,
        "old_name": data.old_name,
        "new_name": data.new_name,
    }


@router.delete("/{project_id}/class-annotations/{class_name}")
async def delete_class_annotations(
    project_id: str,
    class_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all annotations that use a specific class_name in a project.

    Only annotations of this class are removed — other boxes on the same
    image are untouched. Any image left with zero annotations goes back to
    'pending', matching the single-annotation delete endpoint, so it
    reappears as needing attention instead of being mistaken for done.
    """
    await get_owned_project(project_id, current_user, db)
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    result = await db.execute(
        delete(Annotation)
        .where(
            Annotation.image_id.in_(image_ids_subq),
            Annotation.class_name == class_name,
        )
        .returning(Annotation.id, Annotation.image_id)
    )
    rows = result.fetchall()
    deleted_count = len(rows)
    affected_image_ids = {row.image_id for row in rows}

    if affected_image_ids:
        remaining = await db.execute(
            select(Annotation.image_id)
            .where(Annotation.image_id.in_(affected_image_ids))
            .distinct()
        )
        still_annotated = {row[0] for row in remaining.fetchall()}
        empty_image_ids = affected_image_ids - still_annotated
        if empty_image_ids:
            await db.execute(
                update(Image).where(Image.id.in_(empty_image_ids)).values(status="pending")
            )

    await db.commit()
    return {"deleted_count": deleted_count, "class_name": class_name}


@router.get("/{project_id}/class-images/{class_name}")
async def get_class_images(
    project_id: str,
    class_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict:
    """Image IDs that have at least one annotation with this class_name —
    powers 'edit mode': pick a class, jump straight to only the images
    that use it, to re-annotate or fix it after a detection failure."""
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(
        select(Annotation.image_id)
        .join(Image, Annotation.image_id == Image.id)
        .where(Image.project_id == project_id, Annotation.class_name == class_name)
        .distinct()
    )
    image_ids = [row[0] for row in result.fetchall()]
    return {"class_name": class_name, "image_ids": image_ids, "count": len(image_ids)}


@router.get("/{project_id}/class-stats")
async def get_class_stats(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, int]:
    """Return annotation count per class_name for a project."""
    await get_owned_project(project_id, current_user, db)
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    result = await db.execute(
        select(Annotation.class_name, func.count(Annotation.id).label("cnt"))
        .where(Annotation.image_id.in_(image_ids_subq))
        .group_by(Annotation.class_name)
    )
    rows = result.fetchall()
    return {row.class_name: row.cnt for row in rows}


@router.delete("/{project_id}/annotations")
async def delete_all_project_annotations(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete every annotation across all images in the project. Images are kept."""
    project = await get_owned_project(project_id, current_user, db)

    image_ids_subq = select(Image.id).where(Image.project_id == project.id).scalar_subquery()
    result = await db.execute(delete(Annotation).where(Annotation.image_id.in_(image_ids_subq)))

    # Reset all image statuses back to pending
    await db.execute(
        update(Image).where(Image.project_id == project.id).values(status='pending')
    )
    await db.commit()

    return {"deleted": result.rowcount}


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete a project and ALL of its data:
      - Annotations (via image cascade)
      - Images (SQLAlchemy cascade on project → images → annotations)
      - Training jobs (DB-level FK ON DELETE CASCADE)
      - Uploaded files on disk
    """
    project = await get_owned_project(project_id, current_user, db)

    # ORM delete — cascades to images → annotations automatically
    await db.delete(project)
    await db.commit()

    # Remove uploaded image files from disk
    project_upload_dir = settings.upload_dir / project_id
    if project_upload_dir.exists():
        shutil.rmtree(project_upload_dir, ignore_errors=True)

    return {"message": "Project deleted successfully", "id": project_id}


@router.post("/{project_id}/flush")
async def flush_project_data(
    project_id: str,
    keep_model: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all images and annotations for a project but keep the project itself.
    If keep_model=True (default), trained model weights (.pt files) are preserved.
    """
    project = await get_owned_project(project_id, current_user, db)

    # Count images before deletion for response
    count_result = await db.execute(
        select(func.count()).select_from(Image).where(Image.project_id == project_id)
    )
    image_count = count_result.scalar()

    # Delete all annotations for this project's images via subquery
    image_ids_subq = select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    await db.execute(delete(Annotation).where(Annotation.image_id.in_(image_ids_subq)))

    # Delete all image records
    await db.execute(delete(Image).where(Image.project_id == project_id))
    await db.commit()

    # Remove uploaded files from disk
    project_upload_dir = settings.upload_dir / project_id
    if project_upload_dir.exists():
        if keep_model:
            # Remove everything except model directories
            for item in project_upload_dir.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        else:
            shutil.rmtree(project_upload_dir, ignore_errors=True)

    # Preserve model files if requested
    model_dir = settings.model_dir / project_id
    model_kept = keep_model and model_dir.exists()

    return {
        "message": "Project data flushed successfully",
        "id": project_id,
        "images_deleted": image_count,
        "model_kept": model_kept,
    }
