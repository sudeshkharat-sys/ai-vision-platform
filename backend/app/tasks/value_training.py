"""
Whole-value classifier — the "new way" for closed-vocabulary plates.

For plates that can only say one of a few known values (badge plates
reading "4" / "6" / "10" / "11", trim-level tags, part codes), reading
character-by-character is solving a HARDER problem than the task needs:
a CTC reader must get every character right at every timestep, so one
glare frame invents or drops a digit ("4"->"44", "66"->"6"). This task
instead trains a small CNN to classify the WHOLE line crop directly
into one of the values that actually appear in the project's labels —
"which of these known values is it?" — which is dramatically more
robust with few labeled photos, because the model only has to separate
a handful of whole-image classes instead of localizing every stroke.

Reuses the CRNN's exact line-crop + normalization pipeline (de-skew,
pad, _normalize_line) so training and inference see identical geometry,
and augments each real crop many times (glare, blur, shift, contrast)
to multiply scarce real data.

Not a replacement for the CRNN on open text (the engine-block VIN can
say anything — that stays CRNN); this is the right tool when the value
set is closed.
"""
from .celery_app import celery_app
from .training import _fetch_training_data, _group_annotations, _safe_float
from .tesseract_training import _anns_to_chars, _group_chars_into_lines
from .crnn_training import (
    IMG_H, IMG_W, LINE_PAD_W_FRAC, LINE_PAD_H_PX,
    _normalize_line, _augment_line, _estimate_text_angle, _color_aware_gray,
)
from ..config import settings

import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import redis as redis_lib


_REGION_CLASS_NAMES = {"plate", "badge", "region", "serial", "serial_region"}


def _plate_region_bbox(anns, iw, ih):
    """Pixel (x1,y1,x2,y2) of this image's "plate"/region annotation (a
    multi-character label like "plate" or "badge"), or None. A box drawn
    once around the whole badge, unlike per-character boxes, doesn't move
    when detection misses or hallucinates one character -- exactly the
    stability a WHOLE-IMAGE classifier needs, since a crop built from
    character-box extents changes width whenever a box is missed or a
    stray glare/reflection box fires, and a classifier will happily learn
    "wider crop" as a proxy for "more digits" and get fooled by it."""
    for ann in anns:
        name = str(ann.get("class_name", "")).strip().lower()
        if name in _REGION_CLASS_NAMES and ann.get("bbox"):
            xc, yc, w, h = ann["bbox"]
            x1, y1 = (xc - w / 2) * iw, (yc - h / 2) * ih
            x2, y2 = (xc + w / 2) * iw, (yc + h / 2) * ih
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                return x1, y1, x2, y2
    return None


def _direct_value_bbox(anns):
    """A single box drawn around the WHOLE badge and labeled with the
    value itself ("10", "11", "4", "6" -- not a single character) --
    the simpler alternative to per-character boxes + a region box.
    _anns_to_chars silently drops any annotation whose label isn't
    exactly one character (it assumes multi-char labels are region boxes
    like "plate"), so without this, an image annotated only this way
    contributes ZERO training crops -- not an error, just silently
    skipped. Returns (bbox, text) for the first such annotation found, or
    None. Region-class names (plate/badge/region/...) are excluded even
    though they're also multi-character, since those mark an AREA, not a
    value to classify."""
    for ann in anns:
        if not ann.get("bbox"):
            continue
        label = str(ann["class_name"]).strip().upper()
        if len(label) <= 1 or label.lower() in _REGION_CLASS_NAMES:
            continue
        return ann["bbox"], label
    return None


def _line_crops_for_project(img_rows, anns_by_image, progress=None):
    """Cut every annotated text line. Prefers the image's whole "plate"
    region box (stable width, immune to per-character detection noise)
    over the character-box-extent crop; falls back to the extent crop
    only when no region box was annotated for that image.
    Returns list of (normalized uint8 image, value string, image_id)."""
    crops = []
    total = len(img_rows)
    for idx, img_row in enumerate(img_rows):
        real_path = Path(".") / img_row["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
        img = cv2.imread(str(real_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        gray_full = _color_aware_gray(img)
        anns = anns_by_image.get(img_row["id"], [])

        direct = _direct_value_bbox(anns)
        if direct is not None:
            (xc, yc, w, h), text = direct
            x1, y1 = max(0, int((xc - w / 2) * iw)), max(0, int((yc - h / 2) * ih))
            x2, y2 = min(iw, int((xc + w / 2) * iw)), min(ih, int((yc + h / 2) * ih))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                crops.append((_normalize_line(gray_full[y1:y2, x1:x2]), text, img_row["id"]))
            if progress and ((idx + 1) % 5 == 0 or idx + 1 == total):
                progress(idx + 1, total, len(crops))
            continue

        chars = _anns_to_chars(anns, iw, ih)
        if not chars:
            continue

        region = _plate_region_bbox(anns, iw, ih)
        if region is not None:
            rx1, ry1, rx2, ry2 = region
            text = "".join(str(c[0]).strip().upper()
                           for c in sorted(chars, key=lambda c: c[1]))
            if text:
                x1, y1 = max(0, int(rx1)), max(0, int(ry1))
                x2, y2 = min(iw, int(rx2)), min(ih, int(ry2))
                crops.append((_normalize_line(gray_full[y1:y2, x1:x2]), text, img_row["id"]))
            if progress and ((idx + 1) % 5 == 0 or idx + 1 == total):
                progress(idx + 1, total, len(crops))
            continue

        line_src = gray_full
        angle = _estimate_text_angle([(c[1], c[2], c[3] - c[1], c[4] - c[2]) for c in chars])
        if abs(angle) > 2.0:
            centers = np.array(
                [[(c[1] + c[3]) / 2, (c[2] + c[4]) / 2] for c in chars], dtype=np.float32)
            cx, cy = float(centers[:, 0].mean()), float(centers[:, 1].mean())
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            line_src = cv2.warpAffine(gray_full, M, (iw, ih),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((centers.shape[0], 1), dtype=np.float32)
            rotated = (M @ np.hstack([centers, ones]).T).T
            chars = [
                (c[0], ncx - (c[3] - c[1]) / 2, ncy - (c[4] - c[2]) / 2,
                 ncx + (c[3] - c[1]) / 2, ncy + (c[4] - c[2]) / 2, c[5])
                for c, (ncx, ncy) in zip(chars, rotated)
            ]

        med_w = float(np.median([c[3] - c[1] for c in chars]))
        pad_x = med_w * LINE_PAD_W_FRAC
        for line in _group_chars_into_lines(chars):
            text = "".join(str(c[0]).strip().upper() for c in line)
            if not text:
                continue
            x1 = max(0, int(min(c[1] for c in line) - pad_x))
            y1 = max(0, int(min(c[2] for c in line) - LINE_PAD_H_PX))
            x2 = min(iw, int(max(c[3] for c in line) + pad_x))
            y2 = min(ih, int(max(c[4] for c in line) + LINE_PAD_H_PX))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append((_normalize_line(line_src[y1:y2, x1:x2]), text, img_row["id"]))

        if progress and ((idx + 1) % 5 == 0 or idx + 1 == total):
            progress(idx + 1, total, len(crops))
    return crops


def _jitter_geometry(img_u8: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small random rotation/scale/translation of a real crop.

    A closed-vocabulary classifier trained on only a handful of real
    photos per value has nothing to learn from except those exact
    pixels -- without geometric variety it just as easily keys off the
    badge's fixed position/size/tilt in the frame (or the background
    behind it) as off the digit shape itself, and that spurious cue
    disappears on any new photo. `_augment_line` alone only varies
    brightness/contrast/blur/noise/horizontal shift, never the crop's
    scale or rotation, so this fills that gap."""
    h, w = img_u8.shape
    angle = rng.uniform(-6, 6)
    scale = rng.uniform(0.85, 1.15)
    tx = rng.uniform(-0.05, 0.05) * w
    ty = rng.uniform(-0.08, 0.08) * h
    bg = int(np.median(np.concatenate([img_u8[:, 0], img_u8[:, -1], img_u8[0, :], img_u8[-1, :]])))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img_u8, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


_VALUE_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_PLAIN,
]


def _render_synthetic_value(text: str, rng: random.Random) -> np.ndarray:
    """Render one of the project's own known values as printed/embossed
    text on a synthetic background, into the same IMG_H x IMG_W canvas
    real crops use.

    Real badge photos for a closed value set (a handful of cars, each
    photographed once or twice) are scarce enough that the classifier
    can memorize backgrounds instead of digit shapes -- this generator
    gives it many independent looks at the ACTUAL characters in varied
    fonts/sizes/contrast/background so it has something forcing it to
    key off the glyph. It never invents new values; text always comes
    from the project's own labeled classes, so it can't teach the model
    a class it doesn't have."""
    bg = rng.randint(60, 200)
    canvas = np.full((IMG_H * 2, IMG_W * 2), bg, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(2, 10), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    font = rng.choice(_VALUE_FONTS)
    thickness = rng.randint(1, 3)
    delta = rng.randint(50, 120) * (1 if rng.random() < 0.5 else -1)
    color = int(np.clip(bg + delta, 0, 255))
    fscale = rng.uniform(1.6, 2.6)
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thickness)
    org = (max(4, (IMG_W * 2 - tw) // 2 + rng.randint(-15, 15)),
           (IMG_H * 2 + th) // 2 + rng.randint(-10, 10))
    cv2.putText(canvas, text, org, font, fscale, color, thickness, cv2.LINE_AA)

    g = cv2.resize(canvas, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return _jitter_geometry(_normalize_line(g), rng)


def _build_value_cnn(tf, n_classes: int):
    """Small CNN over the CRNN's 32x256 line canvas -> value class."""
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])


@celery_app.task(name="app.tasks.value_training.train_value_model", bind=True)
def train_value_model(
    self,
    project_id: str,
    epochs: int = 40,
    augment_copies: int = 40,
    hard_image_ids: list = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.2,
    synthetic_per_class: int = 150,
):
    """Train the whole-value classifier and export ocr_value.tflite +
    value_labels.txt. Classes = the distinct line texts actually labeled
    in this project (e.g. "4", "6", "10", "11")."""
    import tensorflow as tf
    from ..connectors.statedb_connector import StateDBConnector

    db = StateDBConnector()
    rng = random.Random(42)

    try:
        _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
        _r.delete(f"stop_training:{self.request.id}")
    except Exception:
        pass

    with db.get_session() as conn:
        proj, _, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated")
    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}
    anns_by_image = _group_annotations(ann_rows)

    def progress(cur, total, n):
        try:
            self.update_state(state="STARTED", meta={
                "phase": "building_lines", "engine": "value",
                "current": cur, "total": total, "line_crops": n})
        except Exception:
            pass

    crops = _line_crops_for_project(img_rows, anns_by_image, progress)
    by_value = defaultdict(list)
    for im, text, img_id in crops:
        by_value[text].append((im, img_id))
    if len(by_value) < 2:
        return {"error": "Need at least 2 distinct labeled values (line texts). "
                         f"Found: {sorted(by_value.keys())}"}

    classes = sorted(by_value.keys())
    cls_index = {c: i for i, c in enumerate(classes)}
    raw_counts = {c: len(v) for c, v in by_value.items()}
    hard_ids = set(hard_image_ids or [])

    # Split per class FIRST (so no augmented copy of a val photo leaks into
    # train), then oversample the train side with independent augment draws.
    x_tr, y_tr, x_va, y_va = [], [], [], []
    for value, items in by_value.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) >= 2 else 0
        for im, _ in items[:n_val]:
            x_va.append(im.astype(np.float32) / 255.0)
            y_va.append(cls_index[value])
        for im, img_id in items[n_val:] or items[:1]:
            copies = augment_copies * (3 if img_id in hard_ids else 1)
            x_tr.append(im.astype(np.float32) / 255.0)
            y_tr.append(cls_index[value])
            for _ in range(copies):
                # Geometry jitter (rotate/scale/translate) BEFORE the
                # existing brightness/blur/glare/shift augmentation --
                # without it, every copy of a real photo shares the exact
                # same badge position/size/tilt, so the model can pass
                # training by keying on frame geometry alone instead of
                # the digit shape, then fails on any new photo where the
                # badge sits slightly differently in frame.
                jittered = _jitter_geometry(im, rng)
                x_tr.append(_augment_line(jittered, rng).astype(np.float32) / 255.0)
                y_tr.append(cls_index[value])

    # Synthetic rendered digits for each of the project's OWN known
    # values -- real photos per value are often just a handful of cars,
    # which is not enough variety for the CNN to learn "what a 4 looks
    # like" instead of "what THIS car's badge photo looks like". This
    # never introduces a class the project doesn't have (see
    # _render_synthetic_value), so it only ever reinforces the real
    # labels with cheap extra variety.
    made_synthetic = 0
    if synthetic_per_class > 0:
        for value in classes:
            for _ in range(synthetic_per_class):
                im = _render_synthetic_value(value, rng)
                x_tr.append(im.astype(np.float32) / 255.0)
                y_tr.append(cls_index[value])
                made_synthetic += 1

    x_train = np.stack(x_tr)[..., None]
    y_train = np.array(y_tr, dtype=np.int64)
    if x_va:
        x_val = np.stack(x_va)[..., None]
        y_val = np.array(y_va, dtype=np.int64)
    else:
        x_val, y_val = x_train, y_train
    perm = np.random.default_rng(42).permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    model = _build_value_cnn(tf, len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    epoch_hist, epoch_times = [], []
    celery_task = self

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            try:
                _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
                if _r.get(f"stop_training:{celery_task.request.id}"):
                    _r.delete(f"stop_training:{celery_task.request.id}")
                    self.model.stop_training = True
            except Exception:
                pass
            epoch_times.append(time.time())
            eta = None
            if len(epoch_times) >= 2:
                avg = (epoch_times[-1] - epoch_times[0]) / (len(epoch_times) - 1)
                eta = round(avg * (epochs - (epoch + 1)))
            epoch_hist.append({
                "epoch": epoch + 1, "loss": _safe_float(logs.get("loss")),
                "accuracy": _safe_float(logs.get("accuracy")),
                "val_loss": _safe_float(logs.get("val_loss")),
                "val_accuracy": _safe_float(logs.get("val_accuracy"))})
            try:
                celery_task.update_state(state="STARTED", meta={
                    "phase": "training", "engine": "value",
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "eta_seconds": eta, "history": epoch_hist})
            except Exception:
                pass

    self.update_state(state="STARTED", meta={
        "phase": "training", "engine": "value", "epoch": 0,
        "total_epochs": epochs, "classes": classes,
        "split": {"train": int(len(x_train)), "val": int(len(x_val))}})

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[Progress(),
                         tf.keras.callbacks.ReduceLROnPlateau(
                             monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
                         tf.keras.callbacks.EarlyStopping(
                             monitor="val_accuracy", patience=10,
                             restore_best_weights=True)])

    # Evaluate: per-value accuracy + confusions
    probs = model.predict(x_val, batch_size=batch_size, verbose=0)
    preds = probs.argmax(axis=1)
    per_class, confusions = {}, defaultdict(int)
    for t, p in zip(y_val, preds):
        st = per_class.setdefault(classes[t], {"total": 0, "correct": 0})
        st["total"] += 1
        if t == p:
            st["correct"] += 1
        else:
            confusions[f"{classes[t]}->{classes[p]}"] += 1
    per_class_acc = {c: round(s["correct"] / s["total"], 4) for c, s in per_class.items()}
    val_accuracy = float((preds == y_val).mean())

    # Real photos are the only ground truth for what THIS project's badges
    # actually look like; synthetic/augmented copies only multiply what's
    # already there. A class with very few real photos is still likely to
    # generalize poorly no matter how much augmentation runs on top of it,
    # so surface that plainly instead of a silent low accuracy number.
    low_data_classes = sorted(c for c, n in raw_counts.items() if n < 3)

    out_dir = settings.model_dir / project_id / "ocr_value"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "ocr_value.keras"))
    (out_dir / "value_labels.txt").write_text("\n".join(classes) + "\n")

    tflite_path = out_dir / "ocr_value.tflite"
    export_dir = out_dir / "_saved_model_tmp"
    try:
        model.export(str(export_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    except Exception:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path.write_bytes(converter.convert())
    shutil.rmtree(export_dir, ignore_errors=True)

    meta = {
        "engine": "value",
        "input": f"grayscale {IMG_H}x{IMG_W}, /255 (CRNN line canvas)",
        "classes": classes, "raw_counts": raw_counts,
        "val_accuracy": round(val_accuracy, 4),
        "per_class_accuracy": per_class_acc,
        "top_confusions": [{"pair": p, "count": c} for p, c in
                           sorted(confusions.items(), key=lambda kv: -kv[1])[:10]],
        "augment_copies": augment_copies,
        "synthetic_per_class": synthetic_per_class,
        "synthetic_lines_made": made_synthetic,
        "epochs_run": len(epoch_hist),
        "low_data_classes": low_data_classes,
    }
    (out_dir / "value_meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "status": "success", "engine": "value",
        "model_path": str(tflite_path), "classes": classes,
        "val_accuracy": round(val_accuracy, 4),
        "per_class_accuracy": per_class_acc,
        "top_confusions": meta["top_confusions"],
        "raw_counts": raw_counts, "history": epoch_hist,
        "split": {"train": int(len(x_train)), "val": int(len(x_val))},
        "tflite_size_kb": round(tflite_path.stat().st_size / 1024, 1),
        "low_data_classes": low_data_classes,
        "warning": (
            f"Classes with fewer than 3 real photos: {low_data_classes}. "
            "Label more photos for these values -- synthetic/augmented "
            "copies help but can't replace real examples of this "
            "project's actual badges."
        ) if low_data_classes else None,
    }
