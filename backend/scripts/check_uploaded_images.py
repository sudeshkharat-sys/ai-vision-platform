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

It also cross-checks the database against the disk. An image row whose
file is missing is the case that looks strangest in the UI: the review
screen still draws that photo's annotation boxes (they come from the row)
while the photo itself never loads, because the request for it 404s.

Usage
-----
    # every project
    python -m scripts.check_uploaded_images

    # one project
    python -m scripts.check_uploaded_images <project_id>

    # skip the database cross-check (disk scan only)
    python -m scripts.check_uploaded_images --no-db

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


def check_db_rows(only: str | None) -> int:
    """Report image rows whose file is missing from disk.

    Returns the number of missing files, or -1 if the DB was unreachable.
    """
    try:
        # Imported lazily so --no-db works without app deps/DB configured.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from app.connectors.statedb_connector import StateDBConnector
        from app.config import settings
    except Exception as e:
        print(f"\n(Skipping DB cross-check — could not import app: {e})")
        return -1

    sql = "SELECT id, project_id, filename, filepath FROM images"
    params: dict = {}
    if only:
        sql += " WHERE project_id = :pid"
        params["pid"] = only
    sql += " ORDER BY project_id, filename"

    try:
        connector = StateDBConnector()
        with connector.get_session() as conn:
            rows = connector.execute_query(conn, sql, params)
    except Exception as e:
        print(f"\n(Skipping DB cross-check — could not query the database: {e})")
        return -1

    missing = []
    for r in rows:
        # Same resolution the API uses: filepath is stored as "/uploads/<pid>/<file>"
        path = settings.upload_dir.parent / str(r["filepath"]).lstrip("/")
        if not path.exists():
            missing.append(r)

    print(f"\n=== Database cross-check — {len(rows)} image row(s) ===")
    if not missing:
        print("  Every image row has its file on disk.")
    else:
        print(f"  {len(missing)} row(s) point at a file that is NOT on disk.")
        print("  These still show their annotation boxes in review, but the")
        print("  photo itself cannot load:")
        for r in missing:
            print(f"    project {r['project_id']}  {r['filename']}  ->  {r['filepath']}")
    return len(missing)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--no-db"]
    with_db = "--no-db" not in sys.argv
    uploads = Path("./data/uploads")
    if not uploads.is_dir():
        print(f"No uploads directory at {uploads.resolve()}")
        print("Run this from the backend/ directory.")
        return 2

    only = args[0] if args else None
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

    print(f"\nChecked {total} file(s) on disk: {len(problems)} problem(s).")
    if problems:
        broken = sum(1 for _, v, _ in problems if v == "BROKEN")
        fmt = sum(1 for _, v, _ in problems if v == "FORMAT")
        print(f"  BROKEN (corrupt/truncated, re-upload these): {broken}")
        print(f"  FORMAT (valid but browsers can't show, convert to JPEG/PNG): {fmt}")

    if with_db:
        check_db_rows(only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
