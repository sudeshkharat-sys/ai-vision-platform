"""
seg_model.py
~~~~~~~~~~~~~
Resolves which trained segmentation-model weights file to use for a
project, preferring the newer seed/main split over the legacy single-file
name from before that split existed.

Preference order: seg_main_best.pt (fuller model, trained after more of
the project is labeled) -> seg_seed_best.pt (fast bootstrap model, trained
on a small starter batch) -> seg_best.pt (legacy name, pre seed/main split).
"""

from __future__ import annotations

from pathlib import Path

from ..config import settings


def resolve_seg_model_path(project_id: str) -> Path | None:
    project_model_dir = settings.model_dir.resolve() / project_id
    for name in ("seg_main_best.pt", "seg_seed_best.pt", "seg_best.pt"):
        path = project_model_dir / name
        if path.exists():
            return path
    return None
