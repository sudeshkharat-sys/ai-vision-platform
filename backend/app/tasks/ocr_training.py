"""
OCR character-classifier training.

Workflow
--------
1. Read every annotated image + its per-character box annotations
   (drawn in the normal annotation workspace, one box per character,
   class name = the character, e.g. "A", "V", "4").
2. Crop each box into a single-character image (grayscale, CLAHE,
   aspect-preserving resize onto a square canvas).
3. Balance classes offline: augment under-represented characters
   (rotation, scale, shift, brightness/contrast, blur, noise,
   erode/dilate) until every class reaches the same target count.
4. Train a small CNN classifier (Keras) with live per-epoch progress.
5. Export ocr_model.tflite + labels.txt (+ ocr_model.keras) so the
   mobile app can run it with tflite_flutter.
"""
from .celery_app import celery_app
from .training import _fetch_training_data, _group_annotations, _safe_float
from ..config import settings

import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import redis as redis_lib


# ── Character-crop extraction ─────────────────────────────────────

def _extract_char_crop(img, bbox, size: int, margin: float = 0.12):
    """
    Cut one character out of a full plate photo.

    bbox is the normalized YOLO-style [x_center, y_center, w, h].
    Returns a (size, size) uint8 grayscale image, or None if the box
    is degenerate.
    """
    ih, iw = img.shape[:2]
    xc, yc, w, h = bbox
    bw, bh = w * iw, h * ih
    if bw < 2 or bh < 2:
        return None

    # Add a small margin so strokes touching the box edge aren't cut off
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(round(xc * iw - bw / 2 - mx)))
    y1 = max(0, int(round(yc * ih - bh / 2 - my)))
    x2 = min(iw, int(round(xc * iw + bw / 2 + mx)))
    y2 = min(ih, int(round(yc * ih + bh / 2 + my)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Local contrast enhancement — engraved characters are low-contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    crop = clahe.apply(crop)

    # Aspect-preserving resize onto a square canvas (pad with edge median
    # so the padding looks like background, not a hard black border)
    ch, cw = crop.shape
    scale = size / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    border = np.concatenate([resized[0, :], resized[-1, :], resized[:, 0], resized[:, -1]])
    pad_val = int(np.median(border))
    canvas = np.full((size, size), pad_val, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ── Offline augmentation (class balancing) ────────────────────────

def _augment_char(img, rng: random.Random):
    """One random augmented variant of a character crop."""
    size = img.shape[0]
    out = img.copy()

    # Random affine: small rotation / scale / shift — characters must NOT
    # be flipped or heavily rotated (a flipped '4' is not a '4' anymore)
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.88, 1.12)
    tx = rng.uniform(-0.06, 0.06) * size
    ty = rng.uniform(-0.06, 0.06) * size
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    border_val = int(np.median(out))
    out = cv2.warpAffine(out, m, (size, size),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

    # Brightness / contrast jitter (lighting variation in the field)
    alpha = rng.uniform(0.75, 1.3)   # contrast
    beta = rng.uniform(-30, 30)      # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # Stroke-thickness variation — simulates deep vs shallow engraving
    r = rng.random()
    if r < 0.25:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif r < 0.5:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))

    # Blur (focus / motion) or sensor noise
    r = rng.random()
    if r < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.55:
        noise = rng.uniform(4, 12)
        gauss = np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, out.shape)
        out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    return out


def _balance_classes(train_by_class: dict, target_per_class: int, rng: random.Random):
    """
    Augment every class up to target_per_class samples.
    Returns (X_list, y_list, per-class real/augmented counts).
    """
    xs, ys, stats = [], [], {}
    for cls, crops in train_by_class.items():
        real = list(crops)
        xs.extend(real)
        ys.extend([cls] * len(real))
        n_aug = max(0, target_per_class - len(real))
        for i in range(n_aug):
            src = real[i % len(real)]
            xs.append(_augment_char(src, rng))
            ys.append(cls)
        stats[cls] = {"real": len(real), "augmented": n_aug}
    return xs, ys, stats


# ── Synthetic pretraining (prior knowledge of characters) ─────────

PRETRAIN_CHARS = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _render_synthetic_char(ch: str, size: int, rng: random.Random):
    """
    Render one synthetic 'engraved-looking' character: random font,
    stroke weight, contrast polarity (darker or lighter than the metal),
    background texture — then run it through the same augmentation used
    for real crops.
    """
    big = size * 2
    bg = rng.randint(70, 170)
    canvas = np.full((big, big), bg, dtype=np.uint8)

    # Background texture / noise (brushed metal-ish)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 14), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    font = rng.choice([
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ])
    thickness = rng.randint(2, 5)
    # Engraved chars can read darker (shadow) or lighter (highlight) than the metal
    delta = rng.randint(50, 110) * (1 if rng.random() < 0.4 else -1)
    color = int(np.clip(bg + delta, 0, 255))

    # Scale the glyph to roughly fill the canvas
    fscale = 1.0
    (tw, th), _ = cv2.getTextSize(ch, font, fscale, thickness)
    fscale = big * 0.62 / max(tw, th)
    (tw, th), baseline = cv2.getTextSize(ch, font, fscale, thickness)
    org = ((big - tw) // 2, (big + th) // 2)
    cv2.putText(canvas, ch, org, font, fscale, color, thickness, cv2.LINE_AA)

    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return _augment_char(canvas, rng)


EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def _download_emnist(cache_dir: Path, task=None):
    """Fetch the EMNIST balanced train split (idx files) into cache_dir."""
    import urllib.request
    import zipfile

    wanted = (
        "emnist-balanced-train-images-idx3-ubyte.gz",
        "emnist-balanced-train-labels-idx1-ubyte.gz",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "emnist_gzip.zip"
    if task:
        try:
            task.update_state(state="STARTED", meta={
                "phase": "pretraining",
                "detail": "downloading EMNIST from NIST (~0.5 GB, first time only)",
            })
        except Exception:
            pass
    urllib.request.urlretrieve(EMNIST_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = Path(name).name
            if base in wanted:
                with z.open(name) as src, open(cache_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    zip_path.unlink(missing_ok=True)  # free the ~0.5 GB archive


def _load_emnist_chars(img_size: int, per_class: int, rng: random.Random, task=None):
    """
    Load real labeled character images from the EMNIST balanced split
    (NIST). Labels 0-35 map to 0-9 then A-Z — the same order as
    PRETRAIN_CHARS. Downloads once into model_dir/emnist (no extra
    packages; plain idx parsing). Returns (images, label_indices), or
    None when the dataset can't be fetched (offline server) — callers
    fall back to synthetic-only.
    """
    import gzip

    cache_dir = settings.model_dir / "emnist"
    img_gz = cache_dir / "emnist-balanced-train-images-idx3-ubyte.gz"
    lbl_gz = cache_dir / "emnist-balanced-train-labels-idx1-ubyte.gz"
    try:
        if not (img_gz.exists() and lbl_gz.exists()):
            _download_emnist(cache_dir, task=task)
        with gzip.open(lbl_gz) as f:
            f.read(8)  # idx header
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        with gzip.open(img_gz) as f:
            f.read(16)  # idx header
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    except Exception:
        return None

    xs, ys = [], []
    counts = [0] * 36
    order = np.random.default_rng(rng.randrange(1 << 30)).permutation(len(labels))
    for i in order:
        li = int(labels[i])
        if li >= 36 or counts[li] >= per_class:
            continue
        # EMNIST idx images are stored transposed
        img = np.transpose(images[i])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        # EMNIST is white-on-black; engraved chars vary — randomize polarity
        if rng.random() < 0.5:
            img = 255 - img
        xs.append(_augment_char(img, rng))
        ys.append(li)
        counts[li] += 1
        if sum(counts) >= per_class * 36:
            break
    return (xs, ys) if xs else None


def _get_or_build_pretrained_base(tf, img_size: int, task=None,
                                  samples_per_char: int = 250, epochs: int = 10):
    """
    Return a base model with general knowledge of 0-9/A-Z, training and
    caching it on first use (one-time cost, shared by all projects).
    Trained on synthetic engraved-style characters plus, when available,
    real EMNIST character images (v2 cache).
    """
    cache_path = settings.model_dir / f"ocr_pretrained_base_{img_size}_v2.keras"
    if cache_path.exists():
        try:
            return tf.keras.models.load_model(str(cache_path))
        except Exception:
            cache_path.unlink(missing_ok=True)

    rng = random.Random(7)
    xs, ys = [], []
    n_chars = len(PRETRAIN_CHARS)
    for ci, ch in enumerate(PRETRAIN_CHARS):
        for _ in range(samples_per_char):
            xs.append(_render_synthetic_char(ch, img_size, rng))
            ys.append(ci)
        if task:
            try:
                task.update_state(state="STARTED", meta={
                    "phase": "pretraining",
                    "detail": "generating synthetic characters",
                    "current": ci + 1, "total": n_chars,
                })
            except Exception:
                pass

    # Mix in real EMNIST characters when the dataset is reachable —
    # real pen/print strokes generalize better than synthetic fonts alone.
    emnist = _load_emnist_chars(img_size, samples_per_char, rng, task=task)
    if emnist:
        exs, eys = emnist
        xs.extend(exs)
        ys.extend(eys)

    x = np.stack(xs).astype(np.float32)[..., None] / 255.0
    y = np.array(ys, dtype=np.int64)
    perm = np.random.default_rng(7).permutation(len(x))
    x, y = x[perm], y[perm]

    model = _build_cnn(tf, img_size, n_chars)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class PretrainProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if task:
                try:
                    task.update_state(state="STARTED", meta={
                        "phase": "pretraining",
                        "detail": "training base model (one-time, cached)",
                        "epoch": epoch + 1, "total_epochs": epochs,
                        "accuracy": _safe_float((logs or {}).get("accuracy")),
                    })
                except Exception:
                    pass

    model.fit(x, y, epochs=epochs, batch_size=128, verbose=0,
              validation_split=0.1, callbacks=[PretrainProgress()])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_path))
    return model


def _build_cnn(tf, img_size: int, n_classes: int):
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ])


# ── Celery task ───────────────────────────────────────────────────

@celery_app.task(name="app.tasks.ocr_training.train_ocr_model", bind=True)
def train_ocr_model(
    self,
    project_id: str,
    epochs: int = 50,
    img_size: int = 64,
    target_per_class: int = 300,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fine_tune: bool = False,
    focus_classes: list = None,
    use_pretrained: bool = True,
):
    """
    Train the single-character OCR classifier for a project.

    Phases: extract crops → balance/augment → train CNN → evaluate →
    export .tflite + labels.txt.

    fine_tune=True continues from the previously trained model instead of
    starting from scratch (lower LR, keeps existing knowledge — faster).
    focus_classes lists characters the model struggles with; their training
    samples are oversampled ~2.5x so the model sees them far more often,
    without forgetting the rest.
    """
    # TensorFlow is only needed by this task — import lazily so the
    # YOLO/torch workers don't pay the import cost (or conflict on GPU).
    import tensorflow as tf

    from ..connectors.statedb_connector import StateDBConnector

    db = StateDBConnector()
    rng = random.Random(42)

    # Clear any stale stop flag left over from a previous cancel so a
    # fresh run can't be killed by an old click of the Stop button.
    try:
        _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
        _r.delete(f"stop_training:{self.request.id}")
    except Exception:
        pass

    # ── Phase 1: DB reads ────────────────────────────────────────
    with db.get_session() as conn:
        proj, _, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )
    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    anns_by_image = _group_annotations(ann_rows)

    # ── Phase 2: crop every labeled character ────────────────────
    crops_by_class: dict[str, list] = defaultdict(list)
    total_imgs = len(img_rows)
    for idx, img_row in enumerate(img_rows):
        real_path = Path(".") / img_row["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
        img = cv2.imread(str(real_path))
        if img is None:
            continue
        for ann in anns_by_image.get(img_row["id"], []):
            if not ann["bbox"]:
                continue
            label = str(ann["class_name"]).strip().upper()
            if len(label) != 1:
                # Only single-character labels belong to the OCR classifier;
                # boxes like "plate" or "serial_region" are skipped.
                continue
            crop = _extract_char_crop(img, ann["bbox"], img_size)
            if crop is not None:
                crops_by_class[label].append(crop)

        if (idx + 1) % 5 == 0 or idx + 1 == total_imgs:
            try:
                self.update_state(state="STARTED", meta={
                    "phase": "extracting_characters",
                    "current": idx + 1, "total": total_imgs,
                    "pct": round((idx + 1) / total_imgs * 100),
                })
            except Exception:
                pass

    crops_by_class = {c: v for c, v in crops_by_class.items() if v}
    if len(crops_by_class) < 2:
        return {"error": "Need at least 2 different character classes with labeled boxes. "
                         "Draw one box per character and label it with that character (A-Z / 0-9)."}

    classes = sorted(crops_by_class.keys())
    cls_index = {c: i for i, c in enumerate(classes)}
    raw_counts = {c: len(v) for c, v in crops_by_class.items()}

    # ── Phase 3: split, then balance the train split ─────────────
    train_by_class, val_x, val_y = {}, [], []
    for cls, crops in crops_by_class.items():
        rng.shuffle(crops)
        n_val = max(1, round(len(crops) * val_ratio)) if len(crops) >= 2 else 0
        val_x.extend(crops[:n_val])
        val_y.extend([cls] * n_val)
        train_by_class[cls] = crops[n_val:] or crops[:1]

    try:
        self.update_state(state="STARTED", meta={
            "phase": "augmenting", "classes": len(classes),
            "target_per_class": target_per_class,
        })
    except Exception:
        pass

    focus = {str(c).strip().upper() for c in (focus_classes or []) if str(c).strip()}
    if focus:
        # Oversample weak characters: augment them to 2.5x the normal target
        xs, ys, aug_stats = [], [], {}
        for cls, crops in train_by_class.items():
            tgt = int(target_per_class * 2.5) if cls in focus else target_per_class
            cx, cy, cstats = _balance_classes({cls: crops}, tgt, rng)
            xs.extend(cx)
            ys.extend(cy)
            aug_stats.update(cstats)
        train_x, train_y = xs, ys
    else:
        train_x, train_y, aug_stats = _balance_classes(train_by_class, target_per_class, rng)

    def to_tensor(xs, ys):
        x = np.stack(xs).astype(np.float32)[..., None] / 255.0
        y = np.array([cls_index[c] for c in ys], dtype=np.int64)
        return x, y

    x_train, y_train = to_tensor(train_x, train_y)
    x_val, y_val = to_tensor(val_x, val_y) if val_x else (x_train, y_train)

    perm = np.random.default_rng(42).permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # ── Phase 4: build + train the CNN ───────────────────────────
    n_classes = len(classes)
    out_dir = settings.model_dir / project_id / "ocr"

    model = None
    fine_tuned_from = None
    if fine_tune:
        keras_path = out_dir / "ocr_model.keras"
        labels_path = out_dir / "labels.txt"
        if not keras_path.exists():
            return {"error": "No previously trained model to fine-tune — run a full training first."}
        prev_classes = (
            labels_path.read_text().split() if labels_path.exists() else []
        )
        if prev_classes != classes:
            return {"error": "Character set changed since the last training "
                             f"(was {len(prev_classes)} classes, now {len(classes)}). "
                             "Run a full training instead of fine-tune."}
        model = tf.keras.models.load_model(str(keras_path))
        if model.input_shape[1] != img_size:
            return {"error": f"Previous model was trained at {model.input_shape[1]}px, "
                             f"requested {img_size}px. Use the same size or run a full training."}
        # Gentler LR so existing knowledge is refined, not overwritten
        learning_rate = learning_rate / 5
        fine_tuned_from = str(keras_path)

    started_from = "scratch"
    if model is None and use_pretrained:
        # Start from a base model pretrained on synthetic engraved
        # characters — gives prior knowledge of what 0-9/A-Z look like,
        # which matters enormously when real data is scarce.
        try:
            base = _get_or_build_pretrained_base(tf, img_size, task=self)
            feat = tf.keras.Model(base.inputs, base.layers[-2].output)
            out = tf.keras.layers.Dense(n_classes, activation="softmax")(feat.outputs[0])
            model = tf.keras.Model(base.inputs, out)
            started_from = "pretrained_base"
        except Exception:
            model = None  # fall back to scratch below

    if model is None:
        model = _build_cnn(tf, img_size, n_classes)
    if fine_tuned_from:
        started_from = "fine_tune"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epoch_history = []
    epoch_times = []
    stop_flag = {"value": False}
    celery_task = self

    class ProgressCallback(tf.keras.callbacks.Callback):
        current_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            # Within-epoch progress so long epochs on CPU don't look frozen.
            # Throttled to every 10 batches to avoid flooding Redis.
            if batch % 10 != 0:
                return
            steps = self.params.get("steps")
            try:
                celery_task.update_state(state="STARTED", meta={
                    "phase": "training",
                    "epoch": self.current_epoch + 1, "total_epochs": epochs,
                    "batch": batch + 1, "total_batches": steps,
                    "history": epoch_history,
                })
            except Exception:
                pass

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Cooperative cancel — same Redis flag the YOLO trainer uses
            try:
                _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
                if _r.get(f"stop_training:{celery_task.request.id}"):
                    _r.delete(f"stop_training:{celery_task.request.id}")
                    stop_flag["value"] = True
                    self.model.stop_training = True
            except Exception:
                pass

            epoch_times.append(time.time())
            eta = None
            if len(epoch_times) >= 2:
                avg = (epoch_times[-1] - epoch_times[0]) / (len(epoch_times) - 1)
                eta = round(avg * (epochs - (epoch + 1)))

            entry = {
                "epoch": epoch + 1,
                "loss": _safe_float(logs.get("loss")),
                "accuracy": _safe_float(logs.get("accuracy")),
                "val_loss": _safe_float(logs.get("val_loss")),
                "val_accuracy": _safe_float(logs.get("val_accuracy")),
            }
            epoch_history.append(entry)
            try:
                celery_task.update_state(state="STARTED", meta={
                    "phase": "training",
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "eta_seconds": eta, "history": epoch_history,
                })
            except Exception:
                pass

    callbacks = [
        ProgressCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    self.update_state(state="STARTED", meta={
        "phase": "training", "epoch": 0, "total_epochs": epochs,
        "history": [], "classes": classes,
        "split": {"train": int(len(x_train)), "val": int(len(x_val))},
    })

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )

    if stop_flag["value"]:
        from celery.exceptions import Ignore
        raise Ignore()

    # ── Phase 5: evaluate — per-class accuracy + confusions ──────
    probs = model.predict(x_val, batch_size=batch_size, verbose=0)
    preds = probs.argmax(axis=1)
    per_class, confusions = {}, defaultdict(int)
    for true_i, pred_i in zip(y_val, preds):
        stats = per_class.setdefault(classes[true_i], {"total": 0, "correct": 0})
        stats["total"] += 1
        if true_i == pred_i:
            stats["correct"] += 1
        else:
            confusions[f"{classes[true_i]}->{classes[pred_i]}"] += 1
    per_class_acc = {
        c: round(s["correct"] / s["total"], 4) for c, s in per_class.items()
    }
    top_confusions = sorted(confusions.items(), key=lambda kv: -kv[1])[:10]
    val_accuracy = float((preds == y_val).mean())

    # ── Phase 6: export .tflite + labels + keras model ───────────
    out_dir.mkdir(parents=True, exist_ok=True)

    keras_path = out_dir / "ocr_model.keras"
    model.save(str(keras_path))

    (out_dir / "labels.txt").write_text("\n".join(classes) + "\n")

    tflite_path = out_dir / "ocr_model.tflite"
    export_dir = out_dir / "_saved_model_tmp"
    try:
        # Keras 3: export a SavedModel, then convert (most reliable route)
        model.export(str(export_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    except Exception:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path.write_bytes(converter.convert())
    shutil.rmtree(export_dir, ignore_errors=True)

    meta = {
        "classes": classes,
        "img_size": img_size,
        "input": "grayscale, CLAHE, aspect-pad to square, float32 /255",
        "val_accuracy": round(val_accuracy, 4),
        "per_class_accuracy": per_class_acc,
        "raw_counts": raw_counts,
        "augmentation": aug_stats,
        "epochs_run": len(epoch_history),
    }
    (out_dir / "ocr_meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "status": "success",
        "mode": "fine_tune" if fine_tuned_from else "full",
        "started_from": started_from,
        "focus_classes": sorted(focus),
        "model_path": str(tflite_path),
        "classes": classes,
        "val_accuracy": round(val_accuracy, 4),
        "per_class_accuracy": per_class_acc,
        "top_confusions": [{"pair": p, "count": c} for p, c in top_confusions],
        "raw_counts": raw_counts,
        "augmentation": aug_stats,
        "history": epoch_history,
        "split": {"train": int(len(x_train)), "val": int(len(x_val))},
        "tflite_size_kb": round(tflite_path.stat().st_size / 1024, 1),
    }
