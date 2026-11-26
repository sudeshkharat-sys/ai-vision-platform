"""
Export a project's images + annotations to a YOLO dataset folder.

Why this script exists
  The "Download dataset" button builds one big zip inside the running backend.
  This script does NOT need the backend, Celery or Redis. It only needs
  Postgres to be running. It copies ONE image at a time, so it uses very
  little memory, and it can be RUN AGAIN: files that are already copied are
  skipped, so if it gets killed you just start it again and it continues.

Steps (run from an Anaconda Prompt with the ai-vision env active):
  1. Start Postgres only  (local_dev\\start-postgres.bat)
  2. List your projects:
         python scripts\\export_dataset.py --list
  3. Export one project:
         python scripts\\export_dataset.py --project "My Project" --out D:\\exports\\my_project

Make a zip you can upload with the app's Import button (Exe or normal app):
         python scripts\\export_dataset.py --project "My Project" --out D:\\exports\\my_project --zip
  This also creates my_project_dataset.zip next to the folder. For a very big project, use
  --zip-size 2000 to get several smaller zips (2000 images each) and import them one by one.

Result folder:
  images/            all images
  labels/            one .txt per image, YOLO format: class x_center y_center width height
  classes.txt        class names, one per line (line number = class id)
  data.yaml          ready for YOLO training
  manifest.json      same format idea as the app's zip (full annotation details, polygons too)
  missing_files.txt  images that are in the database but not found on disk (if any)

Database login: taken from --host/--port/--db/--user/--password, else from the .env file in
the repo root (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD),
else the app defaults (localhost:5432, ai_vision, postgres/password).
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

import psycopg2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- settings

def read_env_file(path):
    """Read KEY=VALUE lines from a .env file into a dictionary."""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_db_settings(args):
    env = read_env_file(os.path.join(REPO_ROOT, ".env"))

    def pick(arg_value, env_key, default):
        if arg_value is not None:
            return arg_value
        return env.get(env_key, default)

    return {
        "host": pick(args.host, "POSTGRES_HOST", "localhost"),
        "port": int(pick(args.port, "POSTGRES_PORT", "5432")),
        "dbname": pick(args.db, "POSTGRES_DB", "ai_vision"),
        "user": pick(args.user, "POSTGRES_USER", "postgres"),
        "password": pick(args.password, "POSTGRES_PASSWORD", "password"),
    }


# ---------------------------------------------------------------- database

def list_projects(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id, p.name, p.project_type, COUNT(i.id)
        FROM projects p
        LEFT JOIN images i ON i.project_id = p.id
        GROUP BY p.id, p.name, p.project_type, p.created_at
        ORDER BY p.created_at
        """
    )
    return cur.fetchall()


def find_project(conn, text):
    """Find a project by exact id, or by name (case-insensitive)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, classes FROM projects WHERE id = %s OR LOWER(name) = LOWER(%s)",
        (text, text),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        print("More than one project matches. Use the id instead:")
        for r in rows:
            print("  ", r[0], r[1])
        sys.exit(1)
    return rows[0]


def load_images(conn, project_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, filepath, width, height, status FROM images "
        "WHERE project_id = %s ORDER BY created_at, id",
        (project_id,),
    )
    return cur.fetchall()


def load_annotations(conn, project_id):
    """Return {image_id: [annotation dict, ...]} for the whole project."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.image_id, a.class_name, a.bbox, a.annotation_type, a.points, a.state, a.source
        FROM annotations a
        JOIN images i ON i.id = a.image_id
        WHERE i.project_id = %s
        """,
        (project_id,),
    )
    by_image = {}
    for image_id, class_name, bbox, ann_type, points, state, source in cur.fetchall():
        by_image.setdefault(image_id, []).append(
            {
                "class_name": class_name,
                "bbox": bbox,
                "annotation_type": ann_type,
                "points": points,
                "state": state,
                "source": source,
            }
        )
    return by_image


# ---------------------------------------------------------------- files

def find_image_file(filepath, uploads_dir):
    """Find the image on disk. Same two places the app itself looks."""
    candidates = [
        os.path.join(os.path.dirname(uploads_dir), filepath.lstrip("/").replace("/", os.sep)),
        os.path.join(uploads_dir, filepath.replace("/uploads/", "", 1).replace("/", os.sep)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def copy_file_safely(src, dst):
    """Copy to a .part file first, then rename, so a killed run never leaves a half file."""
    part = dst + ".part"
    shutil.copyfile(src, part)
    os.replace(part, dst)


def already_copied(src, dst):
    return os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)


def yolo_line(class_id, bbox):
    """bbox in the database is already normalized [x_center, y_center, width, height]."""
    x, y, w, h = bbox
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------- export

def export_project(conn, project, uploads_dir, out_dir):
    project_id, project_name, project_classes = project
    classes = list(project_classes or [])

    images = load_images(conn, project_id)
    annotations = load_annotations(conn, project_id)
    if not images:
        print("This project has no images.")
        return None

    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Any class used in annotations but missing from the project list is added at the end.
    for anns in annotations.values():
        for a in anns:
            if a["class_name"] not in classes:
                classes.append(a["class_name"])
    class_id_of = {name: i for i, name in enumerate(classes)}

    manifest = {"project_name": project_name, "classes": classes, "images": []}
    missing = []
    copied = skipped = 0
    total = len(images)
    start = time.time()

    for number, (image_id, filename, filepath, width, height, status) in enumerate(images, 1):
        src = find_image_file(filepath, uploads_dir)
        if src is None:
            missing.append(filepath)
            continue

        stored_name = os.path.basename(filepath)
        dst = os.path.join(images_dir, stored_name)

        if already_copied(src, dst):
            skipped += 1
        else:
            copy_file_safely(src, dst)
            copied += 1

        # label file (one line per box)
        lines = []
        for a in annotations.get(image_id, []):
            if a["bbox"]:
                lines.append(yolo_line(class_id_of[a["class_name"]], a["bbox"]))
        label_name = os.path.splitext(stored_name)[0] + ".txt"
        with open(os.path.join(labels_dir, label_name), "w") as f:
            f.write("\n".join(lines))

        manifest["images"].append(
            {
                "filename": filename,
                "stored_filename": stored_name,
                "width": width,
                "height": height,
                "status": status,
                "annotations": annotations.get(image_id, []),
            }
        )

        if number % 50 == 0 or number == total:
            seconds = int(time.time() - start)
            print(f"  {number}/{total} images done  (copied {copied}, already there {skipped})  {seconds}s")

    # small files at the end
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {json.dumps(name)}\n")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if missing:
        with open(os.path.join(out_dir, "missing_files.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing))

    print()
    print(f"Done. Project: {project_name}")
    print(f"  Images exported : {copied + skipped} of {total}")
    print(f"  Missing on disk : {len(missing)}" + ("  (see missing_files.txt)" if missing else ""))
    print(f"  Classes         : {len(classes)}")
    print(f"  Folder          : {out_dir}")
    return manifest


# ---------------------------------------------------------------- zip

def make_zips(out_dir, manifest, zip_size):
    """Zip the exported folder in the same format the app's Import button reads:
         manifest.json   (at the top of the zip)
         images/<stored_filename>
    The zip is written straight to disk, one file at a time (low memory).
    zip_size = images per zip file; 0 means one zip with everything.
    """
    entries = manifest["images"]
    size = zip_size if zip_size and zip_size > 0 else max(len(entries), 1)
    parts = [entries[i:i + size] for i in range(0, len(entries), size)] or [[]]

    base = os.path.abspath(out_dir).rstrip("\\/")
    zip_paths = []
    for number, part in enumerate(parts, 1):
        suffix = "" if len(parts) == 1 else f"_part{number}"
        zip_path = f"{base}{suffix}_dataset.zip"

        part_manifest = dict(manifest)
        part_manifest["images"] = part

        # images are already compressed (jpg/png), so store them without deflating: much faster
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for entry in part:
                name = entry["stored_filename"]
                zf.write(os.path.join(out_dir, "images", name), f"images/{name}")
            zf.writestr("manifest.json", json.dumps(part_manifest, indent=2))
        zip_paths.append(zip_path)
        print(f"  zip written: {zip_path}  ({len(part)} images)")
    return zip_paths


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Export a project dataset without the backend.")
    parser.add_argument("--list", action="store_true", help="list projects and exit")
    parser.add_argument("--project", help="project name or id to export")
    parser.add_argument("--out", help="folder to write the dataset into")
    parser.add_argument(
        "--uploads",
        default=os.path.join(REPO_ROOT, "backend", "data", "uploads"),
        help="the uploads folder (default: backend/data/uploads)",
    )
    parser.add_argument("--zip", action="store_true", help="also make a .zip for the app's Import button")
    parser.add_argument("--zip-size", type=int, default=0, help="images per zip (0 = one zip with everything)")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--db")
    parser.add_argument("--user")
    parser.add_argument("--password")
    args = parser.parse_args()

    settings = get_db_settings(args)
    try:
        conn = psycopg2.connect(connect_timeout=10, **settings)
    except Exception as error:
        print("Could not connect to Postgres:", error)
        print("Is Postgres running (local_dev\\start-postgres.bat)? Is the password right?")
        sys.exit(1)

    if args.list:
        print(f"{'ID':38} {'IMAGES':>7}  {'TYPE':10} NAME")
        for project_id, name, project_type, count in list_projects(conn):
            print(f"{project_id:38} {count:>7}  {project_type:10} {name}")
        return

    if not args.project or not args.out:
        parser.error("give --project and --out (or use --list)")

    uploads_dir = os.path.abspath(args.uploads)
    if not os.path.isdir(uploads_dir):
        print("Uploads folder not found:", uploads_dir)
        print("Use --uploads to point to it (the folder that holds your project images).")
        sys.exit(1)

    project = find_project(conn, args.project)
    if project is None:
        print("No project found for:", args.project, " (use --list to see names)")
        sys.exit(1)

    print(f"Exporting '{project[1]}' ...")
    out_dir = os.path.abspath(args.out)
    manifest = export_project(conn, project, uploads_dir, out_dir)
    if manifest is not None and (args.zip or args.zip_size):
        print("Making zip ...")
        make_zips(out_dir, manifest, args.zip_size)


if __name__ == "__main__":
    main()
