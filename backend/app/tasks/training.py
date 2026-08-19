from .celery_app import celery_app
from ultralytics import YOLO
import os
import shutil
import time
import random
import math
from pathlib import Path
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from collections import defaultdict
import yaml
import json
import cv2
import numpy as np
import redis as redis_lib


def _safe_float(v):
    """Convert a numeric value to a JSON-safe float (None for NaN/Inf)."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except Exception:
        return None


def clahe_gamma_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Three-stage preprocessing pipeline (CLAHE + gamma + unsharp mask),
    in-memory BGR -> BGR. This is what `preprocess=True` (the default)
    bakes into every YOLO training image via `_preprocess_for_inspection`
    below -- factored out so inference can apply the EXACT same transform
    before running the trained model. Training/inference images being on
    different distributions (enhanced vs raw) silently starves detection
    at test time even though the model looks fine on its own eval split.

    Stage 1 — Moderate CLAHE on the L channel
      clipLimit=3.0, tileGridSize=(8,8): enhances local contrast without
      flattening the whole image into grey.

    Stage 2 — Gamma correction (γ = 1.3)
      γ > 1 darkens the shadow/midtone range, widening the perceived gap
      between dark background and the lighter subject.

    Stage 3 — Unsharp mask sharpening
      Subtracts a Gaussian-blurred copy from the original (weighted sum)
      to crispen edges — the signal detectors key off of.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    lut = np.array([(i / 255.0) ** 1.3 * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(out, lut)

    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
    out = cv2.addWeighted(out, 1.4, blurred, -0.4, 0)
    return out


def _preprocess_for_inspection(src_path: Path, dst_path: Path) -> None:
    """Read src_path, apply clahe_gamma_sharpen, write to dst_path. Falls
    back to a plain file copy if OpenCV cannot read the image."""
    img = cv2.imread(str(src_path))
    if img is None:
        shutil.copy(src_path, dst_path)
        return
    cv2.imwrite(str(dst_path), clahe_gamma_sharpen(img))


# ── Shared helpers ────────────────────────────────────────────────

def _fetch_training_data(db, conn, project_id: str, status_filter: str = "annotated"):
    """Read project + images (filtered by status) + all annotations in one session."""
    proj_rows = db.execute_query(
        conn,
        "SELECT id, classes FROM projects WHERE id = :project_id",
        {"project_id": project_id},
    )
    if not proj_rows:
        return None, None, None, None

    raw_classes = proj_rows[0].get("classes")
    if isinstance(raw_classes, str):
        classes = json.loads(raw_classes) if raw_classes else []
    elif isinstance(raw_classes, list):
        classes = raw_classes
    else:
        classes = []

    img_rows = db.execute_query(
        conn,
        "SELECT id, filename, filepath FROM images "
        "WHERE project_id = :project_id AND status = :status",
        {"project_id": project_id, "status": status_filter},
    )
    if not img_rows:
        return proj_rows[0], classes, [], []

    image_ids = [img["id"] for img in img_rows]

    if not classes:
        placeholders = ", ".join(f":id_{i}" for i in range(len(image_ids)))
        params = {f"id_{i}": v for i, v in enumerate(image_ids)}
        class_rows = db.execute_query(
            conn,
            f"SELECT DISTINCT class_name FROM annotations "
            f"WHERE image_id IN ({placeholders})",
            params,
        )
        classes = [row["class_name"] for row in class_rows]

    placeholders = ", ".join(f":id_{i}" for i in range(len(image_ids)))
    params = {f"id_{i}": v for i, v in enumerate(image_ids)}
    ann_rows = db.execute_query(
        conn,
        f"SELECT image_id, class_name, bbox, source, points, annotation_type FROM annotations "
        f"WHERE image_id IN ({placeholders})",
        params,
    )
    return proj_rows[0], classes, img_rows, ann_rows


def _group_annotations(ann_rows):
    """Group raw annotation rows by image_id, normalising bbox/points type."""
    anns_by_image = defaultdict(list)
    for row in ann_rows:
        raw_bbox = row.get("bbox")
        bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
        raw_points = row.get("points")
        points = json.loads(raw_points) if isinstance(raw_points, str) else raw_points
        anns_by_image[row["image_id"]].append({
            "class_name": row["class_name"],
            "bbox": bbox,
            "source": row.get("source", "manual"),
            "points": points,
            "annotation_type": row.get("annotation_type") or "bbox",
        })
    return anns_by_image


def _classify_image_quality(anns: list) -> str:
    """
    Classify an image's annotation quality based on annotation sources.

    Returns 'manual', 'auto_high', or 'auto_review' — used to place images
    in the right training split (manual → always train, auto_review → val
    only or down-weighted).
    """
    sources = {a.get("source", "manual") for a in anns}
    if "manual" in sources:
        return "manual"
    if "auto_review" in sources:
        return "auto_review"
    return "auto_high"


def _split_images(img_rows, train_ratio=0.8, val_ratio=0.15, seed=42,
                   anns_by_image=None):
    """
    Shuffle and split images into train / val / test subsets.

    When *anns_by_image* is supplied the split is **quality-aware**:
    manual-annotated images are prioritised for training (highest quality),
    while ``auto_review`` images are pushed toward validation so the model
    is evaluated against potentially noisier labels rather than memorising
    them.  ``auto`` (high-confidence) images are treated like manual.

    Rules
    -----
    - < 5 images  → everything in train; val mirrors train; no test
    - 5–9 images  → 80 % train, 20 % val; no test
    - ≥ 10 images → train_ratio train, val_ratio val, remainder test
    """
    imgs = list(img_rows)
    rng = random.Random(seed)
    n = len(imgs)

    if n < 5:
        rng.shuffle(imgs)
        return imgs, imgs, []

    # Quality-aware ordering: manual first, then auto, then auto_review
    if anns_by_image:
        quality_order = {"manual": 0, "auto_high": 1, "auto_review": 2}
        imgs.sort(
            key=lambda im: quality_order.get(
                _classify_image_quality(anns_by_image.get(im["id"], [])), 1
            )
        )
        # Shuffle within each quality tier to avoid deterministic bias
        manual_end = 0
        for i, im in enumerate(imgs):
            q = _classify_image_quality(anns_by_image.get(im["id"], []))
            if q != "manual":
                manual_end = i
                break
        else:
            manual_end = n

        manual_imgs = imgs[:manual_end]
        rest_imgs = imgs[manual_end:]
        rng.shuffle(manual_imgs)
        rng.shuffle(rest_imgs)
        imgs = manual_imgs + rest_imgs
    else:
        rng.shuffle(imgs)

    n_train = max(1, round(n * train_ratio))

    if n < 10:
        n_val = n - n_train
        return imgs[:n_train], imgs[n_train:], []

    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train)

    return imgs[:n_train], imgs[n_train:n_train + n_val], imgs[n_train + n_val:]


def _write_split(dataset_path, split_name, split_imgs, anns_by_image, classes,
                 preprocess: bool = True,
                 task=None, progress_offset: int = 0, progress_total: int = 0):
    """Copy (and optionally CLAHE-enhance) images and write label files for one split."""
    (dataset_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(split_imgs):
        real_path = Path(".") / img["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img["filepath"].lstrip("/"))

        # Index-prefixed so a duplicated entry (rare-class oversampling repeats
        # the same source image several times in split_imgs) gets its own file
        # instead of silently overwriting the first copy at the same path.
        dest_name = f"{idx}_{os.path.basename(img['filepath'])}"
        dest_path = dataset_path / "images" / split_name / dest_name

        if preprocess:
            _preprocess_for_inspection(real_path, dest_path)
        else:
            shutil.copy(real_path, dest_path)

        # Push preprocessing progress every 5 images (throttled to avoid Redis flood)
        if task and preprocess and progress_total > 0 and (idx + 1) % 5 == 0:
            current = progress_offset + idx + 1
            try:
                task.update_state(
                    state="STARTED",
                    meta={
                        "phase": "preprocessing",
                        "current": current,
                        "total": progress_total,
                        "split": split_name,
                        "pct": round(current / progress_total * 100),
                    },
                )
            except Exception:
                pass

        label_file = (
            dataset_path / "labels" / split_name
            / (os.path.splitext(dest_name)[0] + ".txt")
        )
        with open(label_file, "w") as f:
            for ann in anns_by_image.get(img["id"], []):
                if ann["bbox"] and ann["class_name"] in classes:
                    cls_idx = classes.index(ann["class_name"])
                    bbox = ann["bbox"]
                    f.write(f"{cls_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def _oversample_rare_images(train_imgs, anns_by_image, classes, max_dup=6):
    """
    Duplicate whole training images that contain under-represented single-
    character classes, so YOLO sees rare characters (e.g. a class with only
    1-2 real boxes) far more often per epoch.

    Only kicks in for character/OCR-detector projects (>=60% single-char
    classes) — a general detection project's rare class is more likely a
    genuinely rare real-world case, not an artifact of which photos got
    uploaded, so it's left to YOLO's normal augmentation there.

    Train split only — never called on val/test, which must stay an
    untouched, representative sample for honest metrics.
    """
    single_char_classes = {c for c in classes if isinstance(c, str) and len(c.strip()) == 1}
    if not single_char_classes or len(single_char_classes) < 0.6 * len(classes):
        return train_imgs

    counts = {c: 0 for c in single_char_classes}
    for img in train_imgs:
        for ann in anns_by_image.get(img["id"], []):
            if ann["class_name"] in counts:
                counts[ann["class_name"]] += 1

    present_counts = [c for c in counts.values() if c > 0]
    if not present_counts:
        return train_imgs
    target = max(present_counts)

    out = []
    for img in train_imgs:
        img_classes = {
            ann["class_name"] for ann in anns_by_image.get(img["id"], [])
            if ann["class_name"] in counts
        }
        if not img_classes:
            out.append(img)
            continue
        rarest = min(counts[c] for c in img_classes)
        dup = min(max_dup, max(1, round(target / rarest)))
        out.extend([img] * dup)
    return out


def _emnist_detection_images(classes, anns_by_image, dataset_path, imgsz=640,
                             per_rare_class=15, rare_ratio=0.5, task=None):
    """
    Synthesize extra YOLO training images for under-represented character
    classes, using real EMNIST character crops (not just augmented copies
    of the one real photo you have) placed as a standalone box on a plain
    textured background — giving the detector genuine shape diversity for
    a class like a single-example 'V'.

    Character/OCR-detector projects only (>=60% single-char classes);
    silently returns 0 for other projects or when EMNIST can't be fetched
    (offline). Writes straight into dataset_path/images/train + labels/train.
    """
    single_char_classes = [c for c in classes if isinstance(c, str) and len(c.strip()) == 1]
    if not single_char_classes or len(single_char_classes) < 0.6 * len(classes):
        return 0

    counts = {c: 0 for c in single_char_classes}
    for anns in anns_by_image.values():
        for ann in anns:
            if ann["class_name"] in counts:
                counts[ann["class_name"]] += 1
    present = [n for n in counts.values() if n > 0]
    if not present:
        return 0
    target = max(present)
    rare_classes = [c for c, n in counts.items() if n < target * rare_ratio]
    if not rare_classes:
        return 0

    from .ocr_training import _load_emnist_chars
    from .crnn_training import CHARSET  # 0-9 then A-Z — same label order EMNIST uses

    rng = random.Random(1234)
    char_size = max(28, imgsz // 10)
    pool = _load_emnist_chars(char_size, per_class=per_rare_class * 2, rng=rng, task=task)
    if pool is None:
        return 0
    xs, ys = pool
    by_class = defaultdict(list)
    for x, y in zip(xs, ys):
        if y < len(CHARSET):
            by_class[CHARSET[y]].append(x)

    img_dir = dataset_path / "images" / "train"
    lbl_dir = dataset_path / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for cls in rare_classes:
        crops = by_class.get(cls)
        if not crops:
            continue  # EMNIST has no examples of this symbol at all
        cls_idx = classes.index(cls)
        for i in range(per_rare_class):
            crop = crops[i % len(crops)]
            bg = rng.randint(170, 230)
            noise = np.random.default_rng(rng.randrange(1 << 30)).integers(-15, 16, (imgsz, imgsz))
            canvas = np.clip(np.full((imgsz, imgsz), bg, dtype=np.int16) + noise, 0, 255).astype(np.uint8)

            ch_h = rng.randint(int(imgsz * 0.08), int(imgsz * 0.16))
            resized = cv2.resize(crop, (ch_h, ch_h), interpolation=cv2.INTER_AREA)
            x0 = rng.randint(0, imgsz - ch_h)
            y0 = rng.randint(0, imgsz - ch_h)
            canvas[y0:y0 + ch_h, x0:x0 + ch_h] = resized

            name = f"emnist_synth_{cls}_{i}"
            cv2.imwrite(str(img_dir / f"{name}.png"), canvas)
            xc, yc = (x0 + ch_h / 2) / imgsz, (y0 + ch_h / 2) / imgsz
            side = ch_h / imgsz
            with open(lbl_dir / f"{name}.txt", "w") as f:
                f.write(f"{cls_idx} {xc} {yc} {side} {side}\n")
            written += 1
    return written


def _build_yolo_dataset(img_rows, anns_by_image, classes, project_id,
                        train_ratio=0.8, val_ratio=0.15, preprocess=True,
                        imgsz=640, task=None):
    """
    Build a YOLO dataset directory with proper train / val / test splits.

    Directory layout
    ----------------
    temp_dataset_{project_id}/
        images/
            train/  val/  test/
        labels/
            train/  val/  test/
        data.yaml
    """
    dataset_path = Path(f"./temp_dataset_{project_id}")
    dataset_path.mkdir(exist_ok=True)

    train_imgs, val_imgs, test_imgs = _split_images(
        img_rows, train_ratio=train_ratio, val_ratio=val_ratio,
        anns_by_image=anns_by_image,
    )
    train_imgs = _oversample_rare_images(train_imgs, anns_by_image, classes)

    total = len(train_imgs) + len(val_imgs) + len(test_imgs)

    # Announce preprocessing start so the UI shows phase immediately
    if task and preprocess and total > 0:
        try:
            task.update_state(
                state="STARTED",
                meta={"phase": "preprocessing", "current": 0, "total": total, "split": "train", "pct": 0},
            )
        except Exception:
            pass

    _write_split(dataset_path, "train", train_imgs, anns_by_image, classes,
                 preprocess=preprocess, task=task,
                 progress_offset=0, progress_total=total)
    _write_split(dataset_path, "val",   val_imgs,   anns_by_image, classes,
                 preprocess=preprocess, task=task,
                 progress_offset=len(train_imgs), progress_total=total)
    if test_imgs:
        _write_split(dataset_path, "test", test_imgs, anns_by_image, classes,
                     preprocess=preprocess, task=task,
                     progress_offset=len(train_imgs) + len(val_imgs), progress_total=total)

    n_synthetic = _emnist_detection_images(classes, anns_by_image, dataset_path, imgsz=imgsz, task=task)

    data_yaml: dict = {
        "path":  str(dataset_path.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes,
    }
    if test_imgs:
        data_yaml["test"] = "images/test"

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return dataset_path, len(train_imgs) + n_synthetic, len(val_imgs), len(test_imgs)


# ── Segmentation dataset helpers ─────────────────────────────────────

def _write_seg_split(dataset_path, split_name, split_imgs, anns_by_image, classes,
                      preprocess: bool = True, task=None, progress_offset: int = 0,
                      progress_total: int = 0):
    """Copy images and write YOLO-seg label files (class + polygon points) for one split.

    Only annotations with annotation_type == 'segment' and real polygon
    points contribute a mask line — everything else (plain boxes, the
    bbox-precision 'polygon' type) is skipped, since a 4-number bbox isn't
    a valid YOLO-seg label. Images with no segment annotation still get an
    empty label file (valid negative sample).
    """
    (dataset_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(split_imgs):
        real_path = Path(".") / img["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img["filepath"].lstrip("/"))

        dest_name = f"{idx}_{os.path.basename(img['filepath'])}"
        dest_path = dataset_path / "images" / split_name / dest_name

        if preprocess:
            _preprocess_for_inspection(real_path, dest_path)
        else:
            shutil.copy(real_path, dest_path)

        if task and preprocess and progress_total > 0 and (idx + 1) % 5 == 0:
            current = progress_offset + idx + 1
            try:
                task.update_state(
                    state="STARTED",
                    meta={
                        "phase": "preprocessing",
                        "current": current,
                        "total": progress_total,
                        "split": split_name,
                        "pct": round(current / progress_total * 100),
                    },
                )
            except Exception:
                pass

        label_file = (
            dataset_path / "labels" / split_name
            / (os.path.splitext(dest_name)[0] + ".txt")
        )
        with open(label_file, "w") as f:
            for ann in anns_by_image.get(img["id"], []):
                if (ann.get("annotation_type") == "segment" and ann.get("points")
                        and len(ann["points"]) >= 3 and ann["class_name"] in classes):
                    cls_idx = classes.index(ann["class_name"])
                    coords = " ".join(f"{x} {y}" for x, y in ann["points"])
                    f.write(f"{cls_idx} {coords}\n")


def _build_yolo_seg_dataset(img_rows, anns_by_image, classes, project_id,
                             train_ratio=0.8, val_ratio=0.15, preprocess=True, task=None):
    """
    Build a YOLO-seg dataset directory (same images/labels/data.yaml layout
    as detection, but label files hold polygon masks instead of boxes).
    """
    dataset_path = Path(f"./temp_seg_dataset_{project_id}")
    dataset_path.mkdir(exist_ok=True)

    train_imgs, val_imgs, test_imgs = _split_images(
        img_rows, train_ratio=train_ratio, val_ratio=val_ratio,
        anns_by_image=anns_by_image,
    )

    total = len(train_imgs) + len(val_imgs) + len(test_imgs)
    if task and preprocess and total > 0:
        try:
            task.update_state(
                state="STARTED",
                meta={"phase": "preprocessing", "current": 0, "total": total, "split": "train", "pct": 0},
            )
        except Exception:
            pass

    _write_seg_split(dataset_path, "train", train_imgs, anns_by_image, classes,
                      preprocess=preprocess, task=task,
                      progress_offset=0, progress_total=total)
    _write_seg_split(dataset_path, "val", val_imgs, anns_by_image, classes,
                      preprocess=preprocess, task=task,
                      progress_offset=len(train_imgs), progress_total=total)
    if test_imgs:
        _write_seg_split(dataset_path, "test", test_imgs, anns_by_image, classes,
                          preprocess=preprocess, task=task,
                          progress_offset=len(train_imgs) + len(val_imgs), progress_total=total)

    data_yaml: dict = {
        "path":  str(dataset_path.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes,
    }
    if test_imgs:
        data_yaml["test"] = "images/test"

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return dataset_path, len(train_imgs), len(val_imgs), len(test_imgs)


def _make_epoch_callback(celery_task, total_epochs, epoch_history, epoch_start_times, stop_flag):
    """Return an on_fit_epoch_end callback that pushes live metrics to Celery."""
    def on_fit_epoch_end(trainer):
        # Check Redis stop flag — set by cancel/force-stop-all endpoints
        try:
            _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
            if _r.get(f"stop_training:{celery_task.request.id}"):
                _r.delete(f"stop_training:{celery_task.request.id}")
                stop_flag['value'] = True
                trainer.stop = True
                return
        except Exception:
            pass

        epoch = trainer.epoch + 1

        losses = {}
        try:
            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                vals = trainer.loss_items
                vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
                names = getattr(trainer, "loss_names", ["box_loss", "cls_loss", "dfl_loss"])
                for name, v in zip(names, vals):
                    losses[name] = _safe_float(v)
        except Exception:
            pass

        metrics = {}
        try:
            if hasattr(trainer, "metrics") and trainer.metrics:
                for k, v in trainer.metrics.items():
                    clean = k.replace("metrics/", "").replace("(B)", "")
                    metrics[clean] = _safe_float(v)
        except Exception:
            pass

        now = time.time()
        epoch_start_times.append(now)
        eta_seconds = None
        if len(epoch_start_times) >= 2:
            avg = (epoch_start_times[-1] - epoch_start_times[0]) / max(
                len(epoch_start_times) - 1, 1
            )
            eta_seconds = round(avg * (total_epochs - epoch))

        entry = {"epoch": epoch, **losses, **metrics}
        epoch_history.append(entry)

        try:
            celery_task.update_state(
                state="STARTED",
                meta={
                    "epoch":        epoch,
                    "total_epochs": total_epochs,
                    "eta_seconds":  eta_seconds,
                    "history":      epoch_history,
                },
            )
        except Exception:
            pass

    return on_fit_epoch_end


# ══════════════════════════════════════════════════════════════════
#  Seed Training Task
# ══════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.training.train_seed_model", bind=True)
def train_seed_model(
    self,
    project_id: str,
    model_name: str = "yolo11s.pt",
    epochs: int = 40,
    imgsz: int = 640,
    preprocess: bool = True,
    batch: int = -1,
    custom_weights: str = None,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_mosaic: float = 0.5,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    aug_mixup: float = 0.0,
    aug_copy_paste: float = 0.05,
):
    """
    Quick seed-training on manually annotated images.
    Synchronous — uses StateDBConnector (psycopg2), no asyncio event-loop conflict.

    Phases
    ------
    1. DB reads  — project + images (annotated) + annotations
    2. Dataset   — build YOLO directory on disk
    3. Training  — YOLO model.train()
    4. Cleanup   — copy seed_best.pt, remove temp dataset
    """
    db = StateDBConnector()

    # ── Phase 1: DB reads ────────────────────────────────────────
    with db.get_session() as conn:
        proj, classes, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )

    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    anns_by_image = _group_annotations(ann_rows)

    # ── Character/OCR projects: neutralise identity-changing augments ──
    # When most classes are single characters (letters/digits), this seed
    # model is a character detector for the OCR pipeline. The object-detection
    # augmentation defaults are actively harmful here: horizontal/vertical
    # flips MIRROR glyphs (S, 6, V, U…) and large rotations blur 6↔9, so the
    # detector is trained on wrong-identity labels. Auto-clamp them so a
    # character model can't be silently corrupted by the generic defaults.
    single_char = [c for c in classes if isinstance(c, str) and len(c.strip()) == 1]
    is_char_project = bool(classes) and len(single_char) >= 0.6 * len(classes)
    if is_char_project:
        aug_fliplr = 0.0
        aug_flipud = 0.0
        aug_degrees = min(aug_degrees, 3.0)

    # ── Phase 2: Build dataset ───────────────────────────────────
    dataset_path, n_train, n_val, n_test = _build_yolo_dataset(
        img_rows, anns_by_image, classes, project_id,
        preprocess=preprocess, imgsz=imgsz, task=self,
    )

    # ── Phase 3: Train ───────────────────────────────────────────
    total_epochs   = epochs
    epoch_history  = []
    epoch_start_times = []
    stop_flag      = {'value': False}

    if custom_weights:
        custom_path = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not custom_path.exists():
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        model = YOLO(str(custom_path))
    else:
        model = YOLO(model_name)
    model.add_callback(
        "on_fit_epoch_end",
        _make_epoch_callback(self, total_epochs, epoch_history, epoch_start_times, stop_flag),
    )

    self.update_state(
        state="STARTED",
        meta={"epoch": 0, "total_epochs": total_epochs, "eta_seconds": None,
              "history": [], "model_name": custom_weights or model_name,
              "split": {"train": n_train, "val": n_val, "test": n_test},
              "char_project": is_char_project},
    )

    # workers=0 is required for Celery daemonic processes; use cache=True so
    # all images are loaded into RAM once before training begins, eliminating
    # per-epoch disk I/O that would otherwise stall the GPU between batches.
    # batch=0.85 targets 85% VRAM (vs the 60% default of batch=-1).
    _batch = 0.90 if batch == -1 else batch

    results = model.train(
        data=str(dataset_path / "data.yaml"),
        epochs=total_epochs,
        imgsz=imgsz,
        batch=_batch,
        cache=True,          # preload dataset into RAM — eliminates disk I/O stall with workers=0
        amp=True,            # FP16 mixed precision — halves VRAM per tensor, faster tensor cores
        device=0,            # explicit CUDA device
        lr0=settings.seed_learning_rate,
        lrf=0.01,            # final lr = lr0 * lrf
        cos_lr=True,         # cosine LR schedule — smoother convergence on small datasets
        warmup_epochs=3,
        weight_decay=0.001,  # stronger L2 regularisation to reduce overfitting
        patience=20,         # early stopping — model converges fast on small datasets
        label_smoothing=0.1, # reduces overconfidence on small datasets
        # --- augmentation (tuned for bright-feature inspection) -----------
        # Key insight: the OK/NOT-OK signal is the *visibility of the white
        # plastic clip*.  Heavy brightness / saturation jitter destroys that
        # signal.  We intentionally keep HSV jitter low so the model learns
        # from the actual colour cue rather than fighting augmentation noise.
        hsv_h=aug_hsv_h,
        hsv_s=aug_hsv_s,
        hsv_v=aug_hsv_v,
        degrees=aug_degrees,
        translate=aug_translate,
        scale=aug_scale,
        fliplr=aug_fliplr,
        flipud=aug_flipud,
        mosaic=aug_mosaic,
        close_mosaic=15,     # disable mosaic for last 15 epochs to stabilise
        mixup=aug_mixup,
        copy_paste=aug_copy_paste,
        # ------------------------------------------------------------------
        project=str(settings.model_dir / project_id),
        name="seed_model",
        verbose=False,
        workers=0,           # Celery workers are daemonic — cannot spawn DataLoader subprocesses
    )

    # ── Discard if stopped by user ───────────────────────────────
    if stop_flag['value']:
        try:
            shutil.rmtree(results.save_dir, ignore_errors=True)
        except Exception:
            pass
        shutil.rmtree(dataset_path, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    # ── Phase 4: Persist + cleanup ───────────────────────────────
    best_model_path = results.save_dir / "weights" / "best.pt"
    target_path = settings.model_dir / project_id / "seed_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, target_path)
    # Remember whether THIS model was trained on CLAHE+gamma+sharpen images
    # so inference (ocr.py::_yolo_predict_raw) can apply the exact same
    # preprocessing this model actually learned from -- training toggling
    # preprocess on/off between runs must not leave inference guessing.
    (target_path.parent / "seed_meta.json").write_text(
        json.dumps({"preprocess": bool(preprocess)}))
    shutil.rmtree(dataset_path)

    final_metrics = epoch_history[-1] if epoch_history else {}

    return {
        "status":     "success",
        "model_path": str(target_path),
        "model_name": model_name,
        "metrics":    final_metrics,
        "history":    epoch_history,
        "split":      {"train": n_train, "val": n_val, "test": n_test},
        "char_project": is_char_project,
    }


# ══════════════════════════════════════════════════════════════════
#  Main Training Task
# ══════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.training.train_main_model", bind=True)
def train_main_model(
    self,
    project_id: str,
    model_name: str = "yolo11s.pt",
    epochs: int = 60,
    use_seed_weights: bool = True,
    imgsz: int = 640,
    preprocess: bool = True,
    batch: int = -1,
    custom_weights: str = None,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_mosaic: float = 0.5,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    aug_mixup: float = 0.0,
    aug_copy_paste: float = 0.1,
):
    """
    Full/main training on ALL annotated images (manual + auto-annotated).
    When use_seed_weights=True, fine-tunes from the existing seed_best.pt;
    otherwise trains from the selected YOLO architecture.

    Phases
    ------
    1. DB reads  — project + ALL annotated images + annotations
    2. Dataset   — build YOLO directory on disk
    3. Training  — YOLO model.train()
    4. Cleanup   — copy main_best.pt, remove temp dataset
    """
    db = StateDBConnector()

    # ── Phase 1: DB reads ────────────────────────────────────────
    with db.get_session() as conn:
        proj, classes, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )

    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    # Resolve pretrained weights — custom upload takes priority over seed/yolo
    if custom_weights:
        custom_path = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not custom_path.exists():
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        pretrained = str(custom_path)
    elif use_seed_weights:
        seed_path = settings.model_dir / project_id / "seed_best.pt"
        if not seed_path.exists():
            return {"error": "Seed model not found — train seed model first, or disable 'Use seed weights'."}
        pretrained = str(seed_path)
    else:
        pretrained = model_name

    anns_by_image = _group_annotations(ann_rows)

    # ── Phase 2: Build dataset ───────────────────────────────────
    dataset_path, n_train, n_val, n_test = _build_yolo_dataset(
        img_rows, anns_by_image, classes, f"{project_id}_main",
        preprocess=preprocess, imgsz=imgsz, task=self,
    )

    # ── Phase 3: Train ───────────────────────────────────────────
    total_epochs   = epochs
    epoch_history  = []
    epoch_start_times = []
    stop_flag      = {'value': False}

    # When fine-tuning from seed weights use a conservative LR to avoid
    # catastrophic forgetting / hallucination; when training from scratch
    # use the standard main LR.
    lr0 = (
        settings.main_learning_rate / 2
        if use_seed_weights
        else settings.main_learning_rate
    )

    model = YOLO(pretrained)
    model.add_callback(
        "on_fit_epoch_end",
        _make_epoch_callback(self, total_epochs, epoch_history, epoch_start_times, stop_flag),
    )

    self.update_state(
        state="STARTED",
        meta={"epoch": 0, "total_epochs": total_epochs, "eta_seconds": None,
              "history": [], "model_name": model_name,
              "use_seed_weights": use_seed_weights,
              "split": {"train": n_train, "val": n_val, "test": n_test}},
    )

    _batch = 0.90 if batch == -1 else batch

    results = model.train(
        data=str(dataset_path / "data.yaml"),
        epochs=total_epochs,
        imgsz=imgsz,
        batch=_batch,
        cache=True,          # preload dataset into RAM — eliminates disk I/O stall with workers=0
        amp=True,            # FP16 mixed precision — halves VRAM per tensor, faster tensor cores
        device=0,            # explicit CUDA device
        lr0=lr0,
        lrf=0.01,            # final lr = lr0 * lrf
        cos_lr=True,         # cosine LR schedule
        warmup_epochs=3,
        weight_decay=0.001,  # stronger L2 regularisation
        patience=20,         # early stopping — stop when mAP stops improving
        label_smoothing=0.05,
        # --- augmentation (same conservative tuning as seed) ---------------
        hsv_h=aug_hsv_h,
        hsv_s=aug_hsv_s,
        hsv_v=aug_hsv_v,
        degrees=aug_degrees,
        translate=aug_translate,
        scale=aug_scale,
        fliplr=aug_fliplr,
        flipud=aug_flipud,
        mosaic=aug_mosaic,
        close_mosaic=10,     # disable mosaic for last 10 epochs to stabilise
        mixup=aug_mixup,
        copy_paste=aug_copy_paste,
        # ------------------------------------------------------------------
        project=str(settings.model_dir / project_id),
        name="main_model",
        verbose=False,
        workers=0,           # Celery workers are daemonic — cannot spawn DataLoader subprocesses
    )

    # ── Discard if stopped by user ───────────────────────────────
    if stop_flag['value']:
        try:
            shutil.rmtree(results.save_dir, ignore_errors=True)
        except Exception:
            pass
        shutil.rmtree(dataset_path, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    # ── Phase 4: Persist + cleanup ───────────────────────────────
    best_model_path = results.save_dir / "weights" / "best.pt"
    target_path = settings.model_dir / project_id / "main_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, target_path)
    shutil.rmtree(dataset_path)

    final_metrics = epoch_history[-1] if epoch_history else {}

    return {
        "status":           "success",
        "model_path":       str(target_path),
        "model_name":       model_name,
        "use_seed_weights": use_seed_weights,
        "metrics":          final_metrics,
        "history":          epoch_history,
        "split":            {"train": n_train, "val": n_val, "test": n_test},
    }


# ══════════════════════════════════════════════════════════════════
#  Segmentation Training Task
# ══════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.training.train_seg_model", bind=True)
def train_seg_model(
    self,
    project_id: str,
    model_name: str = "yolo11n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    preprocess: bool = True,
    batch: int = -1,
    custom_weights: str = None,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_mosaic: float = 0.5,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    aug_mixup: float = 0.0,
    aug_copy_paste: float = 0.05,
):
    """
    Instance-segmentation training on annotations drawn with the 'segment'
    annotation_type (real polygon masks, not the bbox-precision 'polygon'
    type). Requires a '-seg' checkpoint (e.g. yolo11n-seg.pt) — Ultralytics
    infers task=segment from the checkpoint name.

    Phases mirror train_seed_model: DB reads -> dataset build -> train ->
    persist seg_best.pt. Augmentation is tuned the same way too — most seg
    projects here are dotted/engraved CHARACTER masks, and flips/large
    rotations mirror glyph identity (S<->2, 6<->9) just as badly for a
    mask as for a box, so the same char-project auto-clamp applies.
    """
    db = StateDBConnector()

    with db.get_session() as conn:
        proj, classes, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )

    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    anns_by_image = _group_annotations(ann_rows)

    has_seg_anns = any(
        a.get("annotation_type") == "segment" and a.get("points")
        for anns in anns_by_image.values() for a in anns
    )
    if not has_seg_anns:
        return {"error": "No segmentation (polygon mask) annotations found. Draw segment outlines before training."}

    single_char = [c for c in classes if isinstance(c, str) and len(c.strip()) == 1]
    is_char_project = bool(classes) and len(single_char) >= 0.6 * len(classes)
    if is_char_project:
        aug_fliplr = 0.0
        aug_flipud = 0.0
        aug_degrees = min(aug_degrees, 3.0)

    dataset_path, n_train, n_val, n_test = _build_yolo_seg_dataset(
        img_rows, anns_by_image, classes, project_id,
        preprocess=preprocess, task=self,
    )

    total_epochs = epochs
    epoch_history = []
    epoch_start_times = []
    stop_flag = {'value': False}

    if custom_weights:
        custom_path = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not custom_path.exists():
            shutil.rmtree(dataset_path, ignore_errors=True)
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        model = YOLO(str(custom_path))
    else:
        model = YOLO(model_name)
    model.add_callback(
        "on_fit_epoch_end",
        _make_epoch_callback(self, total_epochs, epoch_history, epoch_start_times, stop_flag),
    )

    self.update_state(
        state="STARTED",
        meta={"epoch": 0, "total_epochs": total_epochs, "eta_seconds": None,
              "history": [], "model_name": custom_weights or model_name,
              "split": {"train": n_train, "val": n_val, "test": n_test},
              "char_project": is_char_project},
    )

    _batch = 0.90 if batch == -1 else batch

    results = model.train(
        data=str(dataset_path / "data.yaml"),
        epochs=total_epochs,
        imgsz=imgsz,
        batch=_batch,
        cache=True,
        amp=True,
        device=0,
        lr0=settings.seed_learning_rate,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.001,
        patience=20,
        label_smoothing=0.1,
        hsv_h=aug_hsv_h,
        hsv_s=aug_hsv_s,
        hsv_v=aug_hsv_v,
        degrees=aug_degrees,
        translate=aug_translate,
        scale=aug_scale,
        fliplr=aug_fliplr,
        flipud=aug_flipud,
        mosaic=aug_mosaic,
        close_mosaic=15,
        mixup=aug_mixup,
        copy_paste=aug_copy_paste,
        project=str(settings.model_dir / project_id),
        name="seg_model",
        verbose=False,
        workers=0,
    )

    if stop_flag['value']:
        try:
            shutil.rmtree(results.save_dir, ignore_errors=True)
        except Exception:
            pass
        shutil.rmtree(dataset_path, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    best_model_path = results.save_dir / "weights" / "best.pt"
    target_path = settings.model_dir / project_id / "seg_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, target_path)
    shutil.rmtree(dataset_path)

    final_metrics = epoch_history[-1] if epoch_history else {}

    return {
        "status":     "success",
        "model_path": str(target_path),
        "model_name": custom_weights or model_name,
        "metrics":    final_metrics,
        "history":    epoch_history,
        "split":      {"train": n_train, "val": n_val, "test": n_test},
        "char_project": is_char_project,
    }
