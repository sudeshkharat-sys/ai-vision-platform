"""
check_uploaded_images.py
~~~~~~~~~~~~~~~~~~~~~~~~
Find uploaded photos that a browser cannot display.

Uploads are validated with a lazy ``PIL.Image.open()`` + ``.size``, which
only reads the header -- so a truncated file, or a format PIL can read but
browsers cannot (TIFF, BMP, CMYK JPEG), is accepted at upload and then
renders blank/black on the annotation and review canvases.

This script forces a FULL decode of every file and reports what it finds.
It only reads; nothing is modified or deleted.

Usage
-----
    # every project
    python -m scripts.check_uploaded_images

    # one project
    python -m scripts.check_uploaded_images <project_id>

Run it from the ``backend`` directory so ``./data/uploads`` resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image as PILImage

# Formats browsers decode natively. Anything else displays as blank even
# when the file itself is perfectly valid.
BROWSER_SAFE = {"JPEG", "PNG", "GIF", "WEBP", "BMP"}


def check_file(path: Path) -> tuple[str, str]:
    """Return (verdict, detail) for one file. verdict is OK / BROKEN / FORMAT."""
    try:
        size_bytes = path.stat().st_size
    except OSError as e:
        return "BROKEN", f"cannot stat: {e}"

    if size_bytes == 0:
        return "BROKEN", "zero-byte file"

    try:
        with PILImage.open(path) as img:
            fmt = img.format or "?"
            mode = img.mode
            # .load() is what actually decodes the pixels -- this is the
            # step the upload path skips, and where truncated files fail.
            img.load()
            w, h = img.size
    except Exception as e:
        return "BROKEN", f"decode failed: {type(e).__name__}: {e}"

    if fmt not in BROWSER_SAFE:
        return "FORMAT", f"{fmt} {mode} {w}x{h} — browsers cannot display {fmt}"

    if mode == "CMYK":
        return "FORMAT", f"{fmt} CMYK {w}x{h} — CMYK JPEG renders black in most browsers"

    return "OK", f"{fmt} {mode} {w}x{h} ({size_bytes // 1024} KB)"


def main() -> int:
    uploads = Path("./data/uploads")
    if not uploads.is_dir():
        print(f"No uploads directory at {uploads.resolve()}")
        print("Run this from the backend/ directory.")
        return 2

    only = sys.argv[1] if len(sys.argv) > 1 else None
    project_dirs = [d for d in sorted(uploads.iterdir()) if d.is_dir()]
    if only:
        project_dirs = [d for d in project_dirs if d.name == only]
        if not project_dirs:
            print(f"No such project directory: {uploads / only}")
            return 2

    total = 0
    problems: list[tuple[Path, str, str]] = []

    for pdir in project_dirs:
        files = [f for f in sorted(pdir.iterdir()) if f.is_file()]
        print(f"\n=== {pdir.name} — {len(files)} file(s) ===")
        for f in files:
            total += 1
            verdict, detail = check_file(f)
            if verdict != "OK":
                problems.append((f, verdict, detail))
                print(f"  {verdict:7} {f.name}  {detail}")

    print(f"\nChecked {total} file(s): {len(problems)} problem(s).")
    if problems:
        broken = sum(1 for _, v, _ in problems if v == "BROKEN")
        fmt = sum(1 for _, v, _ in problems if v == "FORMAT")
        print(f"  BROKEN (corrupt/truncated, re-upload these): {broken}")
        print(f"  FORMAT (valid but browsers can't show, convert to JPEG/PNG): {fmt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
