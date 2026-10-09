"""Locate an image's file on disk from its stored ``filepath``."""

from __future__ import annotations

from pathlib import Path

from ..config import settings


def resolve_image_file(project_id: str, filepath: str) -> Path | None:
    """filepath looks like ``/uploads/<project>/<name>`` for uploads but
    ``/uploads/<project>/video_frames/<video>/<name>`` for extracted video
    frames, so keep the whole path under the project folder (not just the
    file name). Returns None when the file isn't there."""
    rel = filepath.lstrip("/")
    prefix = f"uploads/{project_id}/"
    candidates = []
    if rel.startswith(prefix):
        candidates.append(settings.upload_dir / project_id / rel[len(prefix):])
    candidates += [settings.upload_dir.parent / rel, Path(rel),
                   settings.upload_dir / project_id / Path(rel).name]
    for c in candidates:
        if c.is_file():
            return c
    return None
