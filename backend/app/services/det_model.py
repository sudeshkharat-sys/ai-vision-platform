"""
det_model.py
~~~~~~~~~~~~~
Resolves which trained detection-model weights file to use for a project,
mirroring :mod:`app.services.seg_model` for the box/character detector.

Preference order: main_best.pt (fuller model, trained on everything that
has been labeled so far) -> seed_best.pt (fast bootstrap model, trained on
a small starter batch).

Each trainer writes a sidecar meta file next to its weights recording
whether the model was trained on CLAHE+gamma+sharpen images, so inference
can apply the exact same preprocessing the model actually learned from.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import settings

# weights filename -> sidecar meta filename written by its trainer
_META_FOR = {
    "main_best.pt": "main_meta.json",
    "seed_best.pt": "seed_meta.json",
}


def resolve_det_model_path(project_id: str) -> Path | None:
    project_model_dir = settings.model_dir.resolve() / project_id
    for name in ("main_best.pt", "seed_best.pt"):
        path = project_model_dir / name
        if path.exists():
            return path
    return None


def det_model_uses_preprocess(model_path: Path) -> bool:
    """Whether the model at ``model_path`` was trained on preprocessed images.

    Missing/unreadable meta (older models saved before this was tracked)
    defaults to True, matching the training default of the time.
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
