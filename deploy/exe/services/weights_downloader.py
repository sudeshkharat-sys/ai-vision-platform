"""
YOLO base-weight downloader for offline EXE deployment.

Called by launcher.py at startup to verify weights are present and,
when internet is available, to download any that are missing.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

# All models exposed in the UI — must stay in sync with
#   backend/app/api/pipeline.py  YOLO_MODELS  and
#   backend/scripts/download_yolo_weights.py  ALL_MODELS
ALL_MODELS: list[str] = [
    # YOLO26
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    # YOLO12
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLOv9
    "yolov9c.pt", "yolov9e.pt",
    # YOLOv8
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
]

_MIN_BYTES = 1024 * 1024   # 1 MB — anything smaller is a corrupt / partial file


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def check_weights(weights_dir: Path) -> dict[str, bool]:
    """Return {model_name: is_present} for every model in ALL_MODELS."""
    result: dict[str, bool] = {}
    for name in ALL_MODELS:
        p = weights_dir / name
        result[name] = p.exists() and p.stat().st_size >= _MIN_BYTES
    return result


def summary(weights_dir: Path) -> tuple[int, int, float]:
    """Return (n_present, n_total, total_mb)."""
    status = check_weights(weights_dir)
    n_ok = sum(1 for v in status.values() if v)
    total_mb = sum(
        (weights_dir / k).stat().st_size / 1_048_576
        for k, v in status.items()
        if v
    )
    return n_ok, len(ALL_MODELS), total_mb


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str], None]   # (done, total, current_model)


def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'=' * filled}{' ' * (width - filled)}]"


def download_missing(
    weights_dir: Path,
    progress_cb: ProgressCallback | None = None,
) -> list[str]:
    """
    Download any YOLO weights not already present in *weights_dir*.

    Uses the ultralytics YOLO class to trigger the download (it handles
    versioned GitHub URLs internally).  CWD is temporarily set to
    *weights_dir* so files land directly there.

    Returns a list of model names that failed to download.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(weights_dir)

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError:
        print("[weights] ultralytics not available — skipping weight download.")
        return []

    status = check_weights(weights_dir)
    missing = [m for m, ok in status.items() if not ok]
    if not missing:
        return []

    failed: list[str] = []
    for idx, model_name in enumerate(missing):
        if progress_cb:
            progress_cb(idx, len(missing), model_name)
        try:
            YOLO(model_name)
            dest = weights_dir / model_name
            if not (dest.exists() and dest.stat().st_size >= _MIN_BYTES):
                _try_copy_from_cache(model_name, dest)
        except Exception as exc:  # noqa: BLE001
            failed.append(model_name)
            print(f"  [weights] FAIL {model_name}: {exc}")

    if progress_cb:
        progress_cb(len(missing), len(missing), "")

    os.chdir(original_cwd)
    return failed


def _try_copy_from_cache(model_name: str, dest: Path) -> None:
    import shutil
    candidates: list[Path] = []
    try:
        from ultralytics.utils import SETTINGS
        w = SETTINGS.get("weights_dir", "")
        if w:
            candidates.append(Path(w) / model_name)
    except Exception:  # noqa: BLE001
        pass
    home = Path.home()
    candidates += [
        home / ".config" / "Ultralytics" / model_name,
        home / ".cache"  / "ultralytics" / model_name,
        Path("/tmp") / model_name,
    ]
    for src in candidates:
        if src.exists() and src.stat().st_size >= _MIN_BYTES:
            shutil.copy2(src, dest)
            return


# ---------------------------------------------------------------------------
# Console progress helper used by launcher.py
# ---------------------------------------------------------------------------

def print_download_progress(weights_dir: Path) -> list[str]:
    """
    Download missing weights while printing a live console progress bar.
    Returns a list of model names that failed.
    """
    status = check_weights(weights_dir)
    missing = [m for m, ok in status.items() if not ok]
    total = len(missing)

    if not missing:
        return []

    print(f"  Downloading {total} missing weight file{'s' if total > 1 else ''}…")
    failed: list[str] = []

    def _cb(done: int, n: int, name: str) -> None:
        if not name:
            return
        bar = _bar(done, n)
        line = f"  {bar} {done + 1:>2}/{n}  {name:<22}"
        print(f"\r{line}", end="", flush=True)

    failed = download_missing(weights_dir, progress_cb=_cb)
    print()   # newline after the in-place bar
    return failed
