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

import json
from pathlib import Path

from ..config import settings

# weights filename -> sidecar meta filename written by train_seg_model
_META_FOR = {
    "seg_main_best.pt": "seg_main_meta.json",
    "seg_seed_best.pt": "seg_seed_meta.json",
}


def resolve_seg_model_path(project_id: str) -> Path | None:
    project_model_dir = settings.model_dir.resolve() / project_id
    for name in ("seg_main_best.pt", "seg_seed_best.pt", "seg_best.pt"):
        path = project_model_dir / name
        if path.exists():
            return path
    return None


def seg_model_uses_preprocess(model_path: Path) -> bool:
    """Whether the model at ``model_path`` was trained on CLAHE+gamma+sharpen
    preprocessed images (see clahe_gamma_sharpen in tasks/training.py) --
    mirrors det_model.det_model_uses_preprocess for the segmentation models.

    Missing/unreadable meta (older models saved before this was tracked, or
    the legacy seg_best.pt filename) defaults to True, matching the
    training default of the time.
    """
    meta_name = _META_FOR.get(model_path.name)
    if meta_name is None:
        return True
    meta_path = model_path.parent / meta_name
    if not meta_path.exists():
        return True
    try:
        return bool(json.loads(meta_path.read_text()).get("preprocess", True))
    except Exception:
        return True
