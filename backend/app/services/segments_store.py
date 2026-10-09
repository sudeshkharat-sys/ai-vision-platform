"""Load a project's saved Data Segments for the training tasks (sync DB)."""

from __future__ import annotations

import json

from ..connectors.statedb_connector import StateDBConnector
from .data_segments import normalize_segments


def load_segments(project_id: str, names: list | None = None) -> list:
    """The project's saved segments, optionally only those called ``names``.

    Raises ValueError when names are requested that no longer exist, so a run
    never silently trains on a different set of images than the user picked.
    """
    db = StateDBConnector()
    with db.get_session() as conn:
        rows = db.execute_query(
            conn, "SELECT segments FROM projects WHERE id = :id", {"id": project_id})
    raw = rows[0].get("segments") if rows else None
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else []
    segments = normalize_segments(raw or [])
    if names is None:
        return segments
    wanted = list(dict.fromkeys(names))
    by_name = {s["name"]: s for s in segments}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise ValueError(f"Data segment(s) not found: {missing}")
    if not wanted:
        raise ValueError("Pick at least one data segment")
    return [by_name[n] for n in wanted]
