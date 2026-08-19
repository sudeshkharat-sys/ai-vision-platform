"""
CRNN + CTC line-recognition OCR — a "real" OCR the way PaddleOCR / the
Tesseract-LSTM / ML Kit's reader work: it takes a cropped LINE of text
(located beforehand by YOLO or a box crop) and reads the whole string in
one shot, with character context, instead of classifying isolated boxes.

Design choices that matter for this project
-------------------------------------------
* Charset is the FULL 0-9 + A-Z (36) — so the model has knowledge of every
  character out of the box (learned from synthetic engraved-style text),
  and your labeled photos fine-tune the characters you actually have data
  for. This is the "OCR that already knows letters and gets better on our
  font" the manager asked for.
* Fully-convolutional backbone + CTC (NO LSTM). A conv-only sequence model
  converts to TFLite with builtin ops only, so it runs in Flutter with
  tflite_flutter and a trivial greedy CTC decode in Dart — no TF-Select
  delegate, no native code.
* Training data = real lines built from your per-character boxes, PLUS
  lines composited from your real character crops (huge in-domain data
  multiplier), PLUS synthetic rendered lines over the full charset (general
  character knowledge). Fixed input 32x256 grayscale.
"""
from .celery_app import celery_app
from .training import _fetch_training_data, _group_annotations, _safe_float
from .tesseract_training import _boxes_to_lines
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


# ── Fixed geometry / charset ──────────────────────────────────────
IMG_H = 32
IMG_W = 256
DOWNSAMPLE = 4                      # width stride of the backbone -> T = IMG_W/4
TIME_STEPS = IMG_W // DOWNSAMPLE    # 64
CHARSET = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
BLANK_INDEX = len(CHARSET)          # CTC blank is the last class
NUM_CLASSES = len(CHARSET) + 1
MAX_LABEL_LEN = 24
_CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}


# ── Line-image normalization ──────────────────────────────────────

def _normalize_line(gray: np.ndarray) -> np.ndarray:
    """Contrast-enhance and letterbox a line crop into a fixed IMG_H x IMG_W
    grayscale canvas (dark text on light background, aspect preserved).

    Dot-peen text lives or dies here: a single peen dot is often 1-2 px
    after the resize to IMG_H, so the dots must be fused into solid
    strokes at the crop's NATIVE resolution — a morphological closing
    sized to the dot pitch — before any downscale ever happens."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # Decide polarity from Otsu — engraved strokes can be lighter or darker
    thr, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if int(cv2.countNonZero(bw)) > bw.size / 2:
        # majority is bright -> strokes are the dark minority -> already fine
        pass
    else:
        g = 255 - g

    # Fuse dot-matrix dots into continuous strokes while the crop is still
    # large. Kernel ~1/12 of line height sits between typical dot pitch and
    # inter-character gap, so dots of one stroke merge without bridging into
    # the neighboring character. Solid engraved/printed strokes pass through
    # unchanged. Skipped for crops already at model scale (nothing to save).
    if g.shape[0] > IMG_H * 2:
        k = max(3, (g.shape[0] // 12) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        g = 255 - cv2.morphologyEx(255 - g, cv2.MORPH_CLOSE, kernel)

    h, w = g.shape
    scale = IMG_H / h
    nw = max(1, min(IMG_W, int(round(w * scale))))
    resized = cv2.resize(g, (nw, IMG_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    canvas[:, :nw] = resized
    return canvas


def _deskewed_char_crop(gray_full: np.ndarray, points_px):
    """
    Tight, de-skewed crop of a single character from its traced polygon
    outline (absolute pixel coords), instead of the character's own
    axis-aligned bounding box. On a rotated/tilted character (common on an
    angled engine-plate photo), the axis-aligned box is inflated by up to
    ~sqrt(2)x in the direction of the text line and commonly overlaps the
    neighboring character's box too -- so a plain box crop can contain a
    fragment of the character next to it, feeding that contamination
    straight into training as if it were a clean example of this one.
    Returns None (falls back to the axis-aligned crop) if the polygon is
    degenerate or too small to form a sane rectangle.
    """
    pts = np.array(points_px, dtype=np.float32)
    if pts.shape[0] < 3:
        return None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)
    rw_i, rh_i = int(round(rw)), int(round(rh))
    if rw_i < 2 or rh_i < 2:
        return None
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = gray_full.shape[:2]
    rotated = cv2.warpAffine(gray_full, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.getRectSubPix(rotated, (rw_i, rh_i), (cx, cy))
    # minAreaRect's (w, h) ordering isn't guaranteed to put the longer side
    # last for a near-square glyph -- rotate 90 degrees to portrait when the
    # crop reads noticeably wider than tall, since a single character strip
    # should never be much wider than it is tall.
    if crop.shape[1] > crop.shape[0] * 1.4:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _char_strip(gray_crop: np.ndarray) -> np.ndarray:
    """Height-normalize a single character crop to ~IMG_H tall, natural width,
    for compositing into synthetic lines."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    g = clahe.apply(gray_crop)
    h, w = g.shape
    target_h = int(IMG_H * 0.8)
    scale = target_h / h
    nw = max(2, int(round(w * scale)))
    return cv2.resize(g, (nw, target_h), interpolation=cv2.INTER_AREA)


# ── Synthetic data (general character knowledge) ──────────────────

def _render_synthetic_line(text: str, rng: random.Random) -> np.ndarray:
    """Render a string in a random engraved-looking style into IMG_H x IMG_W."""
    bg = rng.randint(70, 180)
    canvas = np.full((IMG_H * 2, IMG_W * 2), bg, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(3, 12), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    font = rng.choice([
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ])
    thickness = rng.randint(1, 3)
    delta = rng.randint(45, 110) * (1 if rng.random() < 0.4 else -1)
    color = int(np.clip(bg + delta, 0, 255))

    fscale = rng.uniform(1.4, 2.2)
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thickness)
    if tw < 4:
        tw = 4
    org = (rng.randint(4, 20), (IMG_H * 2 + th) // 2)
    cv2.putText(canvas, text, org, font, fscale, color, thickness, cv2.LINE_AA)

    g = cv2.resize(canvas, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return _normalize_line(g)


# 5x7 dot-matrix bitmaps for 0-9/A-Z — the layout a dot-peen marking head
# actually stamps, so synthetic lines match the real engraving style
# (isolated round dots on metal) instead of continuous Hershey strokes.
_DOT_FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "01010", "00100", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}

# Pairs that differ by one or two dots on a degraded stamp — the exact
# misreads seen in production (V read as U/T, 0 as 8, tilted 6 as 9, ...).
# Oversampled in the dot-peen synthetic set so the model learns to use the
# whole glyph shape, not just the dot that glare erased.
_CONFUSION_PAIRS = [
    ("V", "U"), ("V", "T"), ("0", "8"), ("6", "9"), ("1", "7"),
    ("1", "I"), ("1", "L"), ("5", "S"), ("B", "8"), ("2", "Z"),
    ("0", "O"), ("0", "D"), ("F", "R"), ("4", "A"), ("6", "G"),
    ("J", "V"), ("J", "T"), ("V", "1"), ("J", "1"),
]


def _confusable_string(rng: random.Random, min_len=6, max_len=12) -> str:
    """Random string biased toward the confusable characters."""
    n = rng.randint(min_len, max_len)
    out = []
    for _ in range(n):
        if rng.random() < 0.6:
            out.append(rng.choice(rng.choice(_CONFUSION_PAIRS)))
        else:
            out.append(rng.choice(CHARSET))
    return "".join(out)


def _render_dotpeen_line(text: str, rng: random.Random) -> np.ndarray:
    """Render a string as dot-peen engraving on synthetic metal: a 5x7 grid
    of round dots per character, with glare gradient, brushed-surface noise,
    randomly faint/missing dots, dot-position jitter, slight tilt and blur —
    the degradations that make real plates hard."""
    dot_r = rng.randint(2, 4)
    pitch = int(round(dot_r * rng.uniform(2.2, 3.0)))
    char_w, char_h = 5 * pitch, 7 * pitch
    gap = int(round(pitch * rng.uniform(0.9, 1.8)))
    margin = 3 * pitch
    W = len(text) * (char_w + gap) - gap + 2 * margin
    H = char_h + 2 * margin

    base = rng.uniform(90, 185)
    canvas = np.full((H, W), base, dtype=np.float32)
    # brushed-metal texture: noise smeared along the grind direction
    tex = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 14), (H, W))
    tex = cv2.blur(tex.astype(np.float32), (rng.randint(9, 25), 1))
    canvas += tex
    # glare: a broad bright (or dark shadow) blob drifting across the plate
    glare = np.zeros((H, W), dtype=np.float32)
    gx, gy = rng.randint(0, W - 1), rng.randint(0, H - 1)
    cv2.circle(glare, (gx, gy), max(H, W) // rng.randint(2, 4), rng.uniform(-60, 80), -1)
    glare = cv2.blur(glare, (W // 2 | 1, H | 1))
    canvas += glare

    bright_dots = rng.random() < 0.5
    for i, ch in enumerate(text):
        bitmap = _DOT_FONT.get(ch)
        if bitmap is None:
            continue
        x0 = margin + i * (char_w + gap)
        for r, row in enumerate(bitmap):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                if rng.random() < 0.06:      # dot the peen head skipped / glare ate
                    continue
                cx = int(round(x0 + c * pitch + rng.uniform(-0.18, 0.18) * pitch))
                cy = int(round(margin + r * pitch + rng.uniform(-0.18, 0.18) * pitch))
                strength = rng.uniform(0.35, 1.0)   # faint dots are common
                delta = rng.uniform(55, 115) * strength * (1 if bright_dots else -1)
                color = float(np.clip(canvas[min(cy, H - 1), min(cx, W - 1)] + delta, 0, 255))
                cv2.circle(canvas, (cx, cy), dot_r + (1 if rng.random() < 0.2 else 0),
                           color, -1, lineType=cv2.LINE_AA)

    g = np.clip(canvas, 0, 255).astype(np.uint8)
    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        g = cv2.GaussianBlur(g, (k, k), 0)
    return _normalize_line(g)


def _composite_line(char_crops: dict, text: str, rng: random.Random):
    """Build a line image from real character crops (in-domain). Returns None
    if any character in `text` has no available crop."""
    parts = []
    for ch in text:
        pool = char_crops.get(ch)
        if not pool:
            return None
        parts.append(rng.choice(pool))
    gap = rng.randint(1, 5)
    total_w = sum(p.shape[1] for p in parts) + gap * (len(parts) - 1)
    strip = np.full((int(IMG_H * 0.8), max(2, total_w)), 255, dtype=np.uint8)
    x = 0
    for i, p in enumerate(parts):
        strip[:, x:x + p.shape[1]] = p
        x += p.shape[1] + gap
    # pad vertically to IMG_H-ish then normalize
    pad = (IMG_H - strip.shape[0]) // 2 + 2
    strip = cv2.copyMakeBorder(strip, pad, pad, 4, 4, cv2.BORDER_CONSTANT, value=255)
    return _normalize_line(strip)


def _augment_line(img_u8: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light augmentation shared by real + synthetic lines."""
    out = img_u8
    if rng.random() < 0.5:
        alpha = rng.uniform(0.8, 1.25)
        beta = rng.uniform(-25, 25)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.25:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 10), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        # A local glare/shadow patch drifting over part of the line -- the
        # real cause seen in production of a specific character reading
        # wrong consistently on one photo (e.g. "8" -> "B"): a bright/dark
        # patch of reflection sitting over just that character. Real lines
        # only ever got ONE random augmentation draw per training run
        # before (train_crnn_model now oversamples them with several
        # independent draws), so this patch is what most of that repeated
        # variety should actually be teaching the model to see through.
        h, w = out.shape
        gx, gy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        glare = np.zeros((h, w), dtype=np.float32)
        cv2.circle(glare, (gx, gy), rng.randint(w // 6, w // 2), rng.uniform(-70, 70), -1)
        glare = cv2.blur(glare, (w // 4 | 1, h | 1))
        out = np.clip(out.astype(np.float32) + glare, 0, 255).astype(np.uint8)
    return out


# ── Label encoding ────────────────────────────────────────────────

def _encode(text: str):
    seq = [_CHAR_TO_IDX[c] for c in text if c in _CHAR_TO_IDX]
    return seq[:MAX_LABEL_LEN]


def _random_string(rng: random.Random, min_len=3, max_len=10) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(CHARSET) for _ in range(n))


def _emnist_char_pools(rng: random.Random, per_class: int = 80):
    """
    Build per-character pools of REAL character images from EMNIST (NIST),
    covering the full 0-9/A-Z. Used to composite lines grounded in real
    character shapes rather than only computer-drawn fonts. Reuses the same
    loader/cache as the per-char CNN engine (model_dir/emnist). Returns None
    when EMNIST isn't reachable/cached — caller just skips it.
    """
    try:
        from .ocr_training import _load_emnist_chars
    except Exception:
        return None
    data = _load_emnist_chars(int(IMG_H * 0.8), per_class, rng)
    if not data:
        return None
    xs, ys = data
    pools = defaultdict(list)
    for im, li in zip(xs, ys):
        if 0 <= li < len(CHARSET):
            pools[CHARSET[li]].append(im)
    return pools if pools else None


# ── Model ─────────────────────────────────────────────────────────

def _build_crnn(tf):
    """Fully-convolutional CRNN: CNN backbone that collapses height to 1 and
    keeps a width sequence, two 1D convs for local context, then per-timestep
    softmax over NUM_CLASSES. Converts to TFLite with builtin ops only."""
    layers = tf.keras.layers
    inp = layers.Input(shape=(IMG_H, IMG_W, 1), name="image")
    x = inp
    # Downsample width by 4 (two width-pools), height all the way to 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)                     # 16 x 128
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)                     # 8 x 64
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)                     # 4 x 64
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)                     # 2 x 64
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)                     # 1 x 64
    # (batch, 1, 64, 256) -> (batch, 64, 256)
    x = layers.Reshape((TIME_STEPS, 256))(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    y = layers.Dense(NUM_CLASSES, activation="softmax", name="logits")(x)
    return tf.keras.Model(inp, y, name="crnn")


def _greedy_decode(probs: np.ndarray) -> str:
    """CTC greedy decode of one (T, C) softmax matrix -> string."""
    best = probs.argmax(axis=1)
    out, prev = [], -1
    for idx in best:
        if idx != prev and idx != BLANK_INDEX:
            out.append(CHARSET[idx])
        prev = idx
    return "".join(out)


# ── Celery task ───────────────────────────────────────────────────

@celery_app.task(name="app.tasks.crnn_training.train_crnn_model", bind=True)
def train_crnn_model(
    self,
    project_id: str,
    epochs: int = 40,
    synthetic_lines: int = 3000,
    composite_lines: int = 4000,
    emnist_lines: int = 3000,
    dotpeen_lines: int = 4000,
    real_augment_copies: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.15,
):
    """Train the CRNN line recognizer and export ocr_crnn.tflite + charset."""
    import tensorflow as tf
    from ..connectors.statedb_connector import StateDBConnector

    db = StateDBConnector()
    rng = random.Random(42)

    try:
        _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
        _r.delete(f"stop_training:{self.request.id}")
    except Exception:
        pass

    # ── Phase 1: DB reads ────────────────────────────────────────
    with db.get_session() as conn:
        proj, _, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated")
    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}
    anns_by_image = _group_annotations(ann_rows)

    # ── Phase 2: real lines + per-char crops from the boxes ──────
    real_lines = []          # (uint8 image, text)
    char_crops = defaultdict(list)
    total = len(img_rows)
    for idx, img_row in enumerate(img_rows):
        real_path = Path(".") / img_row["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
        img = cv2.imread(str(real_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for line in _boxes_to_lines(anns_by_image.get(img_row["id"], []), iw, ih):
            text = "".join(c[0] for c in line if c[0] in _CHAR_TO_IDX)
            if not text:
                continue
            x1 = max(0, int(min(c[1] for c in line)))
            y1 = max(0, int(min(c[2] for c in line)))
            x2 = min(iw, int(max(c[3] for c in line)))
            y2 = min(ih, int(max(c[4] for c in line)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            real_lines.append((_normalize_line(gray_full[y1:y2, x1:x2]), text))
            # individual char crops for compositing
            for (label, cx1, cy1, cx2, cy2, points_px) in line:
                if label not in _CHAR_TO_IDX:
                    continue
                # A polygon-traced character gets a de-skewed crop of just
                # its own outline -- on a tilted plate, the axis-aligned
                # box below can extend well past the glyph and into a
                # neighboring character, silently feeding a crop that's
                # part-Z-part-whatever's-next-to-it into training as if it
                # were a clean "Z". Falls back to the plain box crop when
                # there's no polygon (old bbox-only annotations).
                crop = _deskewed_char_crop(gray_full, points_px) if points_px else None
                if crop is None:
                    a1, b1 = max(0, int(cx1)), max(0, int(cy1))
                    a2, b2 = min(iw, int(cx2)), min(ih, int(cy2))
                    if a2 - a1 >= 2 and b2 - b1 >= 2:
                        crop = gray_full[b1:b2, a1:a2]
                if crop is not None and crop.size > 0:
                    char_crops[label].append(_char_strip(crop))

        if (idx + 1) % 5 == 0 or idx + 1 == total:
            try:
                self.update_state(state="STARTED", meta={
                    "phase": "building_lines", "current": idx + 1, "total": total,
                    "real_lines": len(real_lines)})
            except Exception:
                pass

    have_chars = sorted(char_crops.keys())
    if not real_lines and not have_chars:
        return {"error": "No usable character boxes found. Label characters (one box each) first."}

    # ── Phase 3: assemble the training set ───────────────────────
    try:
        self.update_state(state="STARTED", meta={"phase": "synthesizing",
                                                 "detail": "compositing + rendering lines"})
    except Exception:
        pass

    # Real photos are the only data with this project's actual font,
    # lighting, and glare -- but until now each one was only ever trained
    # on ONCE per run, with a single random augmentation draw. Split them
    # into train/val FIRST, before any duplication, so an augmented copy
    # of a real photo can never end up in val while another copy of that
    # SAME photo sits in train (that would leak and make val accuracy look
    # better than it really is). Then oversample only the train side --
    # each copy gets its own independent random crop/glare/blur draw from
    # _augment_line, giving the model many more effective looks at the
    # hardest real cases without needing more labeled photos.
    real_shuffled = list(real_lines)
    rng.shuffle(real_shuffled)
    n_real_val = max(1, int(round(len(real_shuffled) * val_ratio))) if real_shuffled else 0
    real_val, real_train = real_shuffled[:n_real_val], real_shuffled[n_real_val:]

    train_set = []
    for im, txt in real_train:
        for _ in range(max(1, real_augment_copies)):
            train_set.append((im, txt))
    val_set = list(real_val)

    X, Y = [], []
    # composited-from-real lines (in-domain, only over characters we have)
    made = 0
    guard = 0
    while made < composite_lines and have_chars and guard < composite_lines * 5:
        guard += 1
        text = "".join(rng.choice(have_chars) for _ in range(rng.randint(3, 9)))
        im = _composite_line(char_crops, text, rng)
        if im is not None:
            X.append(im); Y.append(text); made += 1

    # EMNIST-composited lines: REAL character shapes over the full 0-9/A-Z,
    # so the model's general knowledge is grounded in real characters, not
    # only computer-drawn fonts. Skipped automatically if EMNIST isn't
    # available on this server.
    # emnist_lines == 0 means "skip EMNIST" -- don't even trigger the
    # (possibly ~0.5GB, first-run) dataset load/download for a result that
    # gets thrown away anyway by the made_em < emnist_lines check below.
    emnist_pools = _emnist_char_pools(rng) if emnist_lines > 0 else {}
    made_em = 0
    if emnist_pools:
        guard = 0
        pool_chars = [c for c in CHARSET if emnist_pools.get(c)]
        while made_em < emnist_lines and pool_chars and guard < emnist_lines * 5:
            guard += 1
            text = "".join(rng.choice(pool_chars) for _ in range(rng.randint(3, 9)))
            im = _composite_line(emnist_pools, text, rng)
            if im is not None:
                X.append(im); Y.append(text); made_em += 1

    # synthetic rendered lines over the FULL charset (font variety)
    for _ in range(synthetic_lines):
        text = _random_string(rng)
        X.append(_render_synthetic_line(text, rng)); Y.append(text)

    # dot-peen rendered lines — the actual engraving style of the engine
    # plates, half of them biased toward the one-dot-apart confusion pairs
    # (V/U, 0/8, 6/9, ...) so the model learns to disambiguate them from
    # overall glyph shape even when a dot is faint or missing
    for i in range(dotpeen_lines):
        text = _confusable_string(rng) if i % 2 == 0 else _random_string(rng, 6, 12)
        X.append(_render_dotpeen_line(text, rng)); Y.append(text)

    # ── Phase 4: split + tensors ─────────────────────────────────
    # X/Y here is only the synthetic/composite/emnist/dot-peen pool --
    # real lines were already split and oversampled above. Split this pool
    # normally and merge into the real-photo train/val sets.
    combined = list(zip(X, Y))
    rng.shuffle(combined)
    n_val = max(1, int(round(len(combined) * val_ratio))) if combined else 0
    val_set += combined[:n_val]
    train_set += combined[n_val:]
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    if not val_set:
        # No real lines and an unlucky split left val empty -- CTC training
        # needs a non-empty validation set for ReduceLROnPlateau/
        # EarlyStopping to have anything to monitor.
        n_val = max(1, int(round(len(train_set) * val_ratio)))
        val_set, train_set = train_set[:n_val], train_set[n_val:]

    def to_arrays(pairs, augment):
        imgs, labels, label_lens = [], [], []
        for im, txt in pairs:
            if augment:
                im = _augment_line(im, rng)
            imgs.append(im.astype(np.float32) / 255.0)
            enc = _encode(txt)
            if not enc:
                enc = [BLANK_INDEX]
            padded = enc + [BLANK_INDEX] * (MAX_LABEL_LEN - len(enc))
            labels.append(padded)
            label_lens.append(len(enc))
        x = np.stack(imgs)[..., None]
        return (x, np.array(labels, dtype=np.int32),
                np.array(label_lens, dtype=np.int32))

    x_tr, y_tr, ll_tr = to_arrays(train_set, augment=True)
    x_va, y_va, ll_va = to_arrays(val_set, augment=False)
    input_len_tr = np.full((len(x_tr), 1), TIME_STEPS, dtype=np.int32)
    input_len_va = np.full((len(x_va), 1), TIME_STEPS, dtype=np.int32)

    # ── Phase 5: CTC training model ──────────────────────────────
    base = _build_crnn(tf)
    labels_in = tf.keras.layers.Input((MAX_LABEL_LEN,), dtype="int32", name="labels")
    input_len_in = tf.keras.layers.Input((1,), dtype="int32", name="input_len")
    label_len_in = tf.keras.layers.Input((1,), dtype="int32", name="label_len")

    def ctc_lambda(args):
        y_pred, labels, in_len, lab_len = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, in_len, lab_len)

    loss_out = tf.keras.layers.Lambda(ctc_lambda, name="ctc")(
        [base.output, labels_in, input_len_in, label_len_in])
    train_model = tf.keras.Model(
        [base.input, labels_in, input_len_in, label_len_in], loss_out)
    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                        loss={"ctc": lambda y_true, y_pred: y_pred})

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
            entry = {"epoch": epoch + 1, "loss": _safe_float(logs.get("loss")),
                     "val_loss": _safe_float(logs.get("val_loss"))}
            epoch_hist.append(entry)
            try:
                celery_task.update_state(state="STARTED", meta={
                    "phase": "training", "engine": "crnn",
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "eta_seconds": eta, "history": epoch_hist})
            except Exception:
                pass

    self.update_state(state="STARTED", meta={
        "phase": "training", "engine": "crnn", "epoch": 0, "total_epochs": epochs,
        "split": {"train": int(len(x_tr)), "val": int(len(x_va))},
        "real_lines": len(real_lines)})

    dummy_tr = np.zeros((len(x_tr),), dtype=np.float32)
    dummy_va = np.zeros((len(x_va),), dtype=np.float32)
    train_model.fit(
        [x_tr, y_tr, input_len_tr, ll_tr], dummy_tr,
        validation_data=([x_va, y_va, input_len_va, ll_va], dummy_va),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[Progress(),
                   tf.keras.callbacks.ReduceLROnPlateau(
                       monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
                   tf.keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=8, restore_best_weights=True)])

    # ── Phase 6: evaluate (exact-line + char accuracy) ───────────
    self.update_state(state="STARTED", meta={"phase": "evaluating"})
    probs = base.predict(x_va, batch_size=batch_size, verbose=0)
    exact, total_chars, correct_chars = 0, 0, 0
    confusions = defaultdict(int)
    samples = []
    from difflib import SequenceMatcher
    for i in range(len(x_va)):
        pred = _greedy_decode(probs[i])
        truth = val_set[i][1]
        if pred == truth:
            exact += 1
        sm = SequenceMatcher(a=truth, b=pred, autojunk=False)
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op == "equal":
                correct_chars += a2 - a1
            elif op == "replace" and (a2 - a1) == (b2 - b1):
                for t, p in zip(truth[a1:a2], pred[b1:b2]):
                    confusions[f"{t}->{p}"] += 1
        total_chars += len(truth)
        if i < 40:
            samples.append({"truth": truth, "predicted": pred})
    line_acc = round(exact / len(x_va), 4) if len(x_va) else None
    char_acc = round(correct_chars / total_chars, 4) if total_chars else None
    top_conf = sorted(confusions.items(), key=lambda kv: -kv[1])[:10]

    # ── Phase 7: export TFLite + charset + keras ─────────────────
    out_dir = settings.model_dir / project_id / "ocr_crnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    base.save(str(out_dir / "ocr_crnn.keras"))
    (out_dir / "charset.txt").write_text("\n".join(CHARSET) + "\n")

    tflite_path = out_dir / "ocr_crnn.tflite"
    export_dir = out_dir / "_saved_model_tmp"
    try:
        base.export(str(export_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    except Exception:
        converter = tf.lite.TFLiteConverter.from_keras_model(base)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path.write_bytes(converter.convert())
    shutil.rmtree(export_dir, ignore_errors=True)

    meta = {
        "engine": "crnn",
        "input": f"grayscale {IMG_H}x{IMG_W}, /255, dark-on-light",
        "time_steps": TIME_STEPS, "charset": CHARSET, "blank_index": BLANK_INDEX,
        "line_accuracy": line_acc, "char_accuracy": char_acc,
        "labeled_chars": have_chars,
        "counts": {"real_lines": len(real_lines), "real_train_lines": len(real_train),
                   "real_val_lines": len(real_val), "real_augment_copies": real_augment_copies,
                   "composite_lines": made, "emnist_lines": made_em,
                   "synthetic_lines": synthetic_lines, "dotpeen_lines": dotpeen_lines},
        "emnist_used": bool(made_em),
        "top_confusions": [{"pair": p, "count": c} for p, c in top_conf],
        "samples": samples,
    }
    (out_dir / "crnn_meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "status": "success", "engine": "crnn",
        "model_path": str(tflite_path),
        "line_accuracy": line_acc, "char_accuracy": char_acc,
        "labeled_chars": have_chars,
        "top_confusions": meta["top_confusions"], "samples": samples[:20],
        "counts": meta["counts"],
        "history": epoch_hist,
        "tflite_size_kb": round(tflite_path.stat().st_size / 1024, 1),
    }
