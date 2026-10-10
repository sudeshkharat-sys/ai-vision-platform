"""
crop_cls_training.py
~~~~~~~~~~~~~~~~~~~~~
Trains the crop + classify pipeline (see services/cls_model.py):

  Region finding: the object-detection model the user already trained
      (Main, else Seed) is the detector -- nothing is trained for it here.
      Training crops come from the annotated crop-class boxes (e.g. engine).
  Classifier: YOLO-cls on those crops, one folder per state (mode="crop"),
      or on whole frames / imported class folders (mode="whole").

A box's state label is Annotation.state, falling back to class_name, so
existing "one class per state" annotations work unchanged.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2

from .celery_app import celery_app
from .training import (
    YOLO,
    _make_epoch_callback,
    _resolve_aug,
    _split_images,
    clahe_gamma_sharpen,
)
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from ..services.class_select import derive_dataset_labels
from ..services.data_segments import filter_by_segments, label_samples
from ..services.segments_store import load_segments
from ..services.cls_model import CLS_FILES, cls_dir, crop_region, resolve_detector



def _fetch(db, conn, project_id: str):
    """Annotated images + their box annotations (including state)."""
    img_rows = db.execute_query(
        conn,
        "SELECT id, filename, filepath FROM images "
        "WHERE project_id = :project_id AND status = 'annotated'",
        {"project_id": project_id},
    )
    if not img_rows:
        return [], []
    ids = [i["id"] for i in img_rows]
    ph = ", ".join(f":id_{k}" for k in range(len(ids)))
    ann_rows = db.execute_query(
        conn,
        f"SELECT image_id, class_name, bbox, source, points, annotation_type, state "
        f"FROM annotations WHERE image_id IN ({ph})",
        {f"id_{k}": v for k, v in enumerate(ids)},
    )
    return img_rows, ann_rows


def _group(ann_rows, region_classes):
    """image_id -> [{class_name, state, bbox, ...}] keeping only usable boxes
    (optionally only those whose class_name is in region_classes). state is
    Annotation.state, falling back to class_name for projects that encode
    the state as the class itself."""
    out: dict = {}
    for row in ann_rows:
        bbox = row.get("bbox")
        bbox = json.loads(bbox) if isinstance(bbox, str) else bbox
        if not bbox or len(bbox) != 4:
            continue
        if region_classes and row["class_name"] not in region_classes:
            continue
        points = row.get("points")
        points = json.loads(points) if isinstance(points, str) else points
        explicit = (row.get("state") or "").strip()
        out.setdefault(row["image_id"], []).append({
            "class_name": row["class_name"],
            "state": explicit or row["class_name"],
            "explicit_state": bool(explicit),
            "bbox": bbox,
            "points": points,
            "annotation_type": row.get("annotation_type") or "bbox",
            "source": row.get("source", "manual"),
        })
    # Once any box carries an explicit state, class_name is just "what the
    # box is" (e.g. engine) -- boxes lacking a state are unlabeled, not a
    # state of their own, so drop them instead of inventing a bogus class.
    if any(a["explicit_state"] for anns in out.values() for a in anns):
        out = {iid: kept for iid, anns in out.items()
               if (kept := [a for a in anns if a["explicit_state"]])}
    return out


def _bbox_from_points(points):
    """Normalized [xc, yc, w, h] around polygon / segment vertices, for rows
    saved without a bbox."""
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    if len(xs) < 3:
        return None
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def _group_plain(ann_rows):
    """image_id -> [{class_name, bbox, points, ...}] for every usable box,
    with no state logic (used by the crop-class / label-class flow, where
    the label comes from nested boxes rather than Annotation.state)."""
    out: dict = {}
    for row in ann_rows:
        bbox = row.get("bbox")
        bbox = json.loads(bbox) if isinstance(bbox, str) else bbox
        points = row.get("points")
        points = json.loads(points) if isinstance(points, str) else points
        if (not bbox or len(bbox) != 4) and points:
            bbox = _bbox_from_points(points)   # polygon / segment rows without a stored box
        if not bbox or len(bbox) != 4:
            continue
        out.setdefault(row["image_id"], []).append({
            "class_name": row["class_name"],
            "bbox": bbox,
            "points": points,
            "annotation_type": row.get("annotation_type") or "bbox",
            "source": row.get("source", "manual"),
        })
    return out


def _resolve(img_row) -> Path:
    p = Path(".") / img_row["filepath"].lstrip("/")
    if not p.exists():
        p = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
    return p


def _xywh_to_xyxy(bbox, w, h):
    xc, yc, bw, bh = bbox
    return ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)


def rotated_crop(img, bbox_xyxy, margin, angle, shift=(0.0, 0.0)):
    """Same window crop_region() would cut, but with the scene rotated by
    `angle` degrees about the region's centre first. YOLO-cls has no
    rotation augmentation of its own, so a rotation-invariant subject
    (e.g. a tyre) needs these baked into the train folder. Pixels that
    rotate in from outside the image are mirrored from the edge."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(round(x1 - mx))), max(0, int(round(y1 - my)))
    x2, y2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    cw, ch = x2 - x1, y2 - y1
    if cw < 2 or ch < 2:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # `shift` moves the window by that fraction of its own size (translate aug)
    m[0, 2] += cw / 2 - cx - shift[0] * cw
    m[1, 2] += ch / 2 - cy - shift[1] * ch
    return cv2.warpAffine(img, m, (cw, ch), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _write_jpg(path: Path, img) -> bool:
    """Encode + write a JPEG and never leave a 0-byte file behind.

    OpenCV 5 can return (False, empty buffer) from imencode instead of raising,
    and ultralytics' patched imwrite then happily writes an EMPTY file that
    later crashes the size probe / the YOLO dataset loader. So check the
    buffer, fall back to PIL, and report failure to the caller."""
    import numpy as np
    # Classifier input is ~224-640px; huge crops (2400+px) only waste disk/RAM and
    # can fail to encode, so cap the long side.
    long_side = max(img.shape[:2])
    if long_side > 1280:
        s = 1280.0 / long_side
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(img)
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok and buf is not None and buf.size > 0:
            buf.tofile(str(path))
            return True
    except Exception:
        pass
    try:
        from PIL import Image as PILImage
        arr = img[:, :, ::-1] if img.ndim == 3 and img.shape[2] == 3 else img
        PILImage.fromarray(np.ascontiguousarray(arr)).save(str(path), quality=95)
        if path.stat().st_size > 0:
            return True
        path.unlink()
    except Exception:
        pass
    return False


def _process_image(root: Path, split: str, n: int, img_row, anns, mode, margin, preprocess,
                   rotate_copies, degrees, translate):
    """Cut + write every crop (and train-only augmented copies) of ONE image.
    Returns ([state, ...] written, number failed, first failure note)."""
    import random
    rng = random.Random(str(img_row["id"]))      # deterministic per image, thread-safe
    img = cv2.imread(str(_resolve(img_row)))
    if img is None:
        return [], 0, f"could not read {img_row.get('filename')}"
    if preprocess:
        img = clahe_gamma_sharpen(img)
    h, w = img.shape[:2]

    def n_copies(state):
        # rotate_copies is an int, or {state: n} when classes are balanced
        return rotate_copies.get(state, 0) if isinstance(rotate_copies, dict) else rotate_copies

    def aug_args():
        return (rng.uniform(-degrees, degrees),
                (rng.uniform(-translate, translate), rng.uniform(-translate, translate)))

    items = []
    if mode == "whole":
        # One label per frame: the state of the largest box.
        big = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        items.append((big["state"], img))
        if split == "train":
            for _ in range(n_copies(big["state"])):
                ang, sh = aug_args()
                rot = rotated_crop(img, (0, 0, w, h), 0.0, ang, sh)
                if rot is not None:
                    items.append((big["state"], rot))
    else:
        for a in anns:
            xyxy = _xywh_to_xyxy(a["bbox"], w, h)
            crop = crop_region(img, xyxy, margin)
            if crop is None:
                continue
            items.append((a["state"], crop))
            if split == "train":
                for _ in range(n_copies(a["state"])):
                    ang, sh = aug_args()
                    rot = rotated_crop(img, xyxy, margin, ang, sh)
                    if rot is not None:
                        items.append((a["state"], rot))

    written, failed, note = [], 0, None
    for k, (state, crop) in enumerate(items):
        d = root / split / state
        d.mkdir(parents=True, exist_ok=True)
        if _write_jpg(d / f"{img_row['id']}_{n}_{k}.jpg", crop):
            written.append(state)
        else:
            failed += 1
            note = note or f"could not encode a crop (shape {getattr(crop, 'shape', None)}, dtype {getattr(crop, 'dtype', None)})"
    return written, failed, note


def _balanced_copies(train_imgs, anns_by_image, mode, base: int, cap: int = 10) -> dict:
    """Extra augmented copies per class so every class ends up with about as
    many TRAIN samples as the biggest one (YOLO-cls has no class weights, so
    oversampling the rare classes with rotated/shifted copies does the job).
    A class never gets fewer than `base` copies."""
    cnt = Counter()
    for row in train_imgs:
        anns = anns_by_image.get(row["id"]) or []
        if mode == "whole" and anns:
            cnt[max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])["state"]] += 1
        else:
            cnt.update(a["state"] for a in anns)
    if not cnt:
        return {}
    top = max(cnt.values())
    limit = max(base, cap)
    return {s: min(limit, max(base, round((1 + base) * top / c) - 1)) for s, c in cnt.items()}


def _recommended_epochs(n_train: int, batch: int, target_steps: int = 4000,
                        lo: int = 25, hi: int = 100) -> int:
    """Epochs so training makes ~target_steps optimizer steps: big (augmented)
    datasets need few epochs, small ones many. Clamped; early-stopping
    (patience) ends it sooner if the validation accuracy stops improving."""
    steps_per_epoch = max(1, -(-n_train // max(1, batch)))
    return int(min(hi, max(lo, round(target_steps / steps_per_epoch))))


def _build_cls_dataset(split_map, anns_by_image, root: Path, mode, margin, preprocess,
                       rotate_copies: int = 0, degrees: float = 0.0, translate: float = 0.0,
                       progress=None, stats: dict | None = None):
    """Write root/<split>/<state>/*.jpg. Returns {split: Counter(state)}.

    Images are processed on a thread pool (OpenCV releases the GIL), which is
    several times faster than one at a time -- the CLAHE/gamma/sharpen
    preprocessing of full-size photos dominates.

    rotate_copies > 0 adds that many copies of every TRAIN sample, each
    rotated by a random angle in [-degrees, +degrees] and shifted by up to
    `translate` of its size (YOLO-cls has no rotation/translation of its
    own). degrees=180 is the full 360-degree spin. val/test stay untouched
    so the accuracy still measures real, un-augmented images.

    progress(split, done, total) is called as images finish; `stats` (if
    given) receives {"failed": n, "notes": [...]}."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    counts = {s: Counter() for s in split_map}
    jobs = [(split, n, row) for split, imgs in split_map.items()
            for n, row in enumerate(imgs) if anns_by_image.get(row["id"])]
    failed_total, notes = 0, []
    workers = max(1, min(8, os.cpu_count() or 2))
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_process_image, root, split, n, row, anns_by_image[row["id"]], mode,
                          margin, preprocess, rotate_copies, degrees, translate): split
                for split, n, row in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            states, failed, note = fut.result()
            counts[futs[fut]].update(states)
            failed_total += failed
            if note and len(notes) < 3:
                notes.append(note)
            if progress:
                progress("all", done, len(jobs))      # may raise _Stopped
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)  # drop queued images right away
        raise
    ex.shutdown(wait=True)
    if stats is not None:
        stats.update({"failed": failed_total, "notes": notes})
    return counts


def _confusion(model, val_dir: Path, classes):
    """Confusion matrix {true: {pred: n}} over the val folder."""
    matrix = {c: {p: 0 for p in classes} for c in classes}
    for cdir in val_dir.iterdir() if val_dir.exists() else []:
        if not cdir.is_dir() or cdir.name not in matrix:
            continue
        files = [str(f) for f in cdir.glob("*.jpg")]
        if not files:
            continue
        for res in model.predict(files, verbose=False):
            matrix[cdir.name][res.names[int(res.probs.top1)]] += 1
    return matrix


class _Stopped(Exception):
    """The user pressed Stop; unwinds the task from any stage."""


def _stop_requested(task_id) -> bool:
    """The Stop button sets this Redis flag (see POST /crop-cls/stop). The
    epoch callback only looks at it once per epoch, so every other stage
    (building crops, loading weights) checks it here."""
    try:
        import redis as redis_lib
        return bool(redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
                    .get(f"stop_training:{task_id}"))
    except Exception:
        return False


def _clear_stop(task_id):
    try:
        import redis as redis_lib
        redis_lib.from_url(settings.redis_url, socket_connect_timeout=1).delete(f"stop_training:{task_id}")
    except Exception:
        pass


class _TaskLog:
    """Stands in for the Celery task so EVERY progress update carries the
    current stage and a running log (shown in the panel's Jobs tab), and each
    log line is also printed on the worker console. The epoch callback from
    training.py calls update_state(state=..., meta=...) and reads request.id,
    both of which are forwarded."""

    def __init__(self, task):
        self._task = task
        self.logs: list = []
        self.stage = "starting"
        self._last_epoch = 0

    @property
    def request(self):
        return self._task.request

    def check_stop(self):
        if _stop_requested(self.request.id):
            raise _Stopped()

    def log(self, msg: str):
        print(f"[crop-cls] {msg}", flush=True)
        self.logs.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del self.logs[:-300]

    def set(self, stage: str, **meta):
        self.stage = stage
        self.update_state(state="STARTED", meta=meta)

    def update_state(self, state="STARTED", meta=None):
        meta = dict(meta or {})
        epoch, hist = meta.get("epoch") or 0, meta.get("history") or []
        if epoch and epoch != self._last_epoch and hist:
            self._last_epoch = epoch
            last = hist[-1]
            acc = last.get("accuracy_top1")
            self.log(f"epoch {epoch}/{meta.get('total_epochs')}"
                     + (f"  loss {last['loss']:.4f}" if last.get("loss") is not None else "")
                     + (f"  val acc {acc:.3f}" if acc is not None else ""))
        meta.setdefault("stage", self.stage)
        meta["logs"] = list(self.logs)
        self._task.update_state(state=state, meta=meta)


def _auto_imgsz(root: Path, limit: int = 200) -> int:
    """Image size for the classifier when the user chose Auto: the median
    longest side of the training crops, rounded to a multiple of 32 and kept
    within 160-448 (small crops gain nothing from upscaling; huge ones are
    slow for no accuracy gain on a classifier). Unreadable files are skipped."""
    sizes = []
    for f in list((root / "train").rglob("*.jpg"))[:limit]:
        try:
            im = cv2.imread(str(f))
        except Exception:
            continue
        if im is not None:
            sizes.append(max(im.shape[:2]))
    if not sizes:
        return 224
    sizes.sort()
    med = sizes[len(sizes) // 2]
    return int(min(448, max(160, round(med / 32) * 32)))


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@celery_app.task(name="app.tasks.crop_cls_training.train_crop_cls_model", bind=True)
def train_crop_cls_model(
    self,
    project_id: str,
    mode: str = "crop",                    # "crop" | "whole"
    region_classes: list | None = None,    # whole mode: only these class_names are classes
    detector: str | None = None,           # "main" | "seed" | None = Main if trained, else Seed
    cls_model_name: str = "yolo11s-cls.pt",
    custom_weights: str | None = None,
    cls_epochs: int = 0,                   # 0 = Auto: sized from the number of training crops
    balance_classes: bool = True,          # oversample rare classes with augmented copies
    cls_imgsz: int = 0,                    # 0 = Auto: sized from the crops
    margin: float = 0.12,
    preprocess: bool = True,
    batch: int = -1,                       # -1 = Auto: fit the GPU (16 on CPU)
    crop_class: str | None = None,         # the detector class to cut out, e.g. "engine"
    label_classes: list | None = None,     # classes inside it that name its state (None = all others)
    empty_label: str | None = "no_cover",  # label for a region with nothing inside (None = skip)
    min_overlap: float = 0.5,              # share of a label box that must lie inside the region
    # Same augmentation knobs as the detection / segmentation panels. YOLO-cls
    # takes flips, colour and scale natively; rotation and translation are
    # baked into extra train crops (rotate_copies per sample).
    augment: bool = True,
    aug_rotate_360: bool = False,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    rotate_copies: int = 4,
    segment_names: list | None = None,     # Data Segments: train only on these, labelled by segment
    **_ignored,                            # mosaic / mixup / copy_paste: no meaning for classification
):
    if mode not in ("crop", "whole"):
        return {"error": "mode must be 'crop' or 'whole'"}

    tl = _TaskLog(self)
    tl.set("reading", epoch=0, total_epochs=cls_epochs, history=[])
    tl.log("Reading annotations from the database")
    db = StateDBConnector()
    with db.get_session() as conn:
        img_rows, ann_rows = _fetch(db, conn, project_id)
    if not img_rows:
        return {"error": "No annotated images found"}
    tl.log(f"{len(img_rows)} annotated images, {len(ann_rows)} annotations")

    plain = _group_plain(ann_rows)
    derive_summary = None
    det_name = det_path = None
    detector_note = None
    seg_labels = None
    if segment_names is not None:
        # Data Segments: only images matching a chosen segment are used, and the
        # segment's label (locked / unlocked / partial ...) is the class.
        try:
            segs = load_segments(project_id, segment_names)
            img_rows, plain, seg_labels, seg_summary = filter_by_segments(img_rows, plain, segs)
        except ValueError as e:
            return {"error": str(e)}
        tl.log("Data segments: " + ", ".join(
            f"{s['name']} -> '{s['label']}': {s['images']}" for s in seg_summary["segments"])
            + f"  (unmatched images skipped: {seg_summary['unmatched']})")
        if not img_rows:
            return {"error": "No annotated images match the selected data segments. Check the segment rules: class names are case-sensitive and must match the project classes exactly (open Data Segments > Preview counts)."}
    if seg_labels is not None:
        if mode == "crop" and not crop_class:
            return {"error": "Choose the detector class to crop (crop_class)"}
        if mode == "crop":
            det_name, det_path = resolve_detector(project_id, detector)
        anns_by_image = label_samples(plain, seg_labels, crop_class if mode == "crop" else None)
        if mode == "crop":
            per = Counter(a["state"] for v in anns_by_image.values() for a in v)
            derive_summary = {"per_class": dict(per), "conflict": 0, "empty": 0,
                              "images": len(anns_by_image)}
            tl.log(f"Crops per class: {dict(per)}")
    elif mode == "crop":
        if not crop_class:
            return {"error": "Choose the detector class to crop (crop_class)"}
        # The detector is NOT needed to train: crops are cut from the boxes you
        # drew. It is only used later, to find the region when testing/predicting,
        # so a missing or mismatched one is just a note, never an error.
        det_name, det_path = resolve_detector(project_id, detector)
        if det_path is not None:
            det_names = list(YOLO(str(det_path)).names.values())
            if crop_class not in det_names:
                detector_note = (f"The {det_name} detector doesn't know the class '{crop_class}' "
                                 f"(it has: {det_names}) -- retrain it with that class before testing.")
        if not label_classes:
            label_classes = sorted({a["class_name"] for anns in plain.values() for a in anns}
                                   - {crop_class})
        anns_by_image, derive_summary = derive_dataset_labels(
            plain, crop_class, label_classes, empty_label, min_overlap)
        tl.log(f"Crops per class: {derive_summary['per_class']}  "
               f"(conflicts skipped: {derive_summary['conflict']}, empty: {derive_summary['empty']})")
        if derive_summary["conflict"]:
            tl.log(f"WARNING: {derive_summary['conflict']} crops hold more than one class at once "
                   f"{derive_summary['conflict_combos']} and were skipped. A crop gets one label - untick "
                   "one of those classes (e.g. a sub-box type) in the class list to keep these crops.")
    else:
        anns_by_image = _group(ann_rows, region_classes)
    img_rows = [i for i in img_rows if anns_by_image.get(i["id"])]
    if not img_rows:
        return {"error": "No usable box annotations found"}

    states = sorted({a["state"] for anns in anns_by_image.values() for a in anns})
    if len(states) < 2:
        return {"error": f"Need at least 2 classes to classify, found: {states}"}

    # Resolve the starting weights before doing any work.
    start_weights = cls_model_name
    if custom_weights:
        cw = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not cw.exists():
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        start_weights = str(cw)

    out_dir = cls_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    result: dict = {"status": "success", "mode": mode, "classes": states}
    if derive_summary:
        result["label_summary"] = derive_summary
    if seg_labels is not None:
        result["segments"] = segment_names
    if det_name:
        result["detector"] = det_name
    if detector_note:
        result["detector_note"] = detector_note

    (fl, fu, _mo, hv, hh, hs, deg, tr, sc, _mx, _cp) = _resolve_aug(
        augment, aug_rotate_360, aug_fliplr, aug_flipud, 0.0, aug_hsv_v, aug_hsv_h,
        aug_hsv_s, aug_degrees, aug_translate, aug_scale, 0.0, 0.0)
    copies = rotate_copies if (augment and (deg > 0 or tr > 0)) else 0

    cls_root = Path(f"./temp_cls_{project_id}")
    try:
        train_i, val_i, test_i = _split_images(img_rows, anns_by_image=anns_by_image)
        split_map = {"train": train_i, "val": val_i}
        if test_i:
            split_map["test"] = test_i
        copies_map = copies
        if balance_classes:
            copies_map = _balanced_copies(train_i, anns_by_image, mode, copies)
        tl.log(f"Images per split: train {len(train_i)}, val {len(val_i)}, test {len(test_i)}"
               + (f"; each train crop gets {copies} rotated/shifted copies" if copies else ""))
        if balance_classes and len(set(copies_map.values())) > 1:
            tl.log(f"Class balancing: extra copies per class {copies_map} "
                   "(rare classes get more so every class has a similar number of training samples)")
        tl.check_stop()
        tl.set("dataset", epoch=0, total_epochs=cls_epochs, history=[])
        shutil.rmtree(cls_root, ignore_errors=True)
        _tick = {"t": 0.0}

        def _progress(split, done, total):
            if time.time() - _tick["t"] >= 1.0:     # don't flood Redis
                _tick["t"] = time.time()
                tl.check_stop()
                tl.set("dataset", epoch=0, total_epochs=cls_epochs, history=[],
                       dataset_progress={"split": split, "done": done, "total": total})

        build_stats: dict = {}
        counts = _build_cls_dataset(
            split_map, anns_by_image, cls_root, mode, margin, preprocess,
            rotate_copies=copies_map, degrees=deg, translate=tr, progress=_progress, stats=build_stats)
        tl.check_stop()
        tl.log("Dataset built: " + "; ".join(f"{k} {dict(v)}" for k, v in counts.items()))
        if build_stats.get("failed"):
            tl.log(f"WARNING: {build_stats['failed']} crops could not be saved and were skipped "
                   f"({'; '.join(build_stats['notes'])})")
        train_states = [s for s in states if counts["train"][s] > 0]
        if len(train_states) < 2:
            return {"error": f"Need training samples for at least 2 classes, got {dict(counts['train'])}"}
        if sum(counts["val"].values()) == 0:   # tiny dataset: mirror train
            shutil.copytree(cls_root / "train", cls_root / "val", dirs_exist_ok=True)
            counts["val"] = counts["train"]

        if augment:
            cls_aug = dict(fliplr=fl, flipud=fu, hsv_h=hh, hsv_s=hs, hsv_v=hv, scale=sc)
        else:
            cls_aug = dict(fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                           scale=0.0, erasing=0.0, auto_augment=None)
        if not cls_imgsz:
            cls_imgsz = _auto_imgsz(cls_root)
        # Auto batch = 90% of GPU memory (same as the detection trainers); CPU can't auto-size.
        run_batch = batch if batch != -1 else (0.9 if device == 0 else 16)
        if not cls_epochs or cls_epochs < 1:
            n_train = sum(counts["train"].values())
            cls_epochs = _recommended_epochs(n_train, batch if batch > 0 else 32)
            tl.log(f"Epochs: Auto -> {cls_epochs} ({n_train} training samples; "
                   "stops earlier if validation accuracy stops improving)")
        history, starts, stop = [], [], {"value": False}
        tl.log(f"Image size {cls_imgsz}, batch {run_batch}, device {'GPU' if device == 0 else 'CPU'}")
        tl.check_stop()
        tl.set("weights", epoch=0, total_epochs=cls_epochs, history=[])
        tl.log(f"Loading weights {start_weights} (downloaded on first use - needs internet)")
        try:
            model = YOLO(start_weights)
        except Exception as e:
            return {"error": f"Could not load '{start_weights}': {e}. The first run downloads these "
                             "weights - check the internet/proxy connection, or copy the file next to the app."}
        model.add_callback(
            "on_fit_epoch_end",
            _make_epoch_callback(tl, cls_epochs, history, starts, stop),
        )
        tl.check_stop()
        tl.log("Training started")
        tl.set("classifier", epoch=0, total_epochs=cls_epochs, history=[],
               samples={s: dict(c) for s, c in counts.items()})
        res = model.train(
            data=str(cls_root), epochs=cls_epochs, imgsz=cls_imgsz, batch=run_batch,
            device=device, amp=device == 0, lr0=settings.seed_learning_rate,
            cos_lr=True, patience=15, **cls_aug,
            project=str(settings.model_dir / project_id), name="crop_cls_classifier",
            verbose=True, workers=0,
        )
        if stop["value"]:
            shutil.rmtree(res.save_dir, ignore_errors=True)
            raise _Stopped()
        best = res.save_dir / "weights" / "best.pt"
        shutil.copy(best, out_dir / CLS_FILES["classifier"])

        confusion = _confusion(YOLO(str(best)), cls_root / "val", train_states)
        total = sum(sum(r.values()) for r in confusion.values())
        correct = sum(confusion[c][c] for c in confusion)
        tl.log(f"Done. Validation accuracy {correct}/{total}")

        meta = {
            "mode": mode, "preprocess": bool(preprocess), "margin": margin,
            "classes": train_states, "region_classes": region_classes,
            "detector": det_name, "crop_class": crop_class, "label_classes": label_classes,
            "empty_label": empty_label,
            "augment": bool(augment), "degrees": deg, "translate": tr,
            "cls_imgsz": cls_imgsz, "batch": batch,
            "val_accuracy": round(correct / total, 4) if total else None,
            "confusion": confusion,
        }
        (out_dir / CLS_FILES["meta"]).write_text(json.dumps(meta))
        result.update({
            "classes": train_states,
            "val_accuracy": meta["val_accuracy"],
            "confusion": confusion,
            "samples": {s: dict(c) for s, c in counts.items()},
            "classifier_metrics": history[-1] if history else {},
            "history": history,
            "model_path": str(out_dir / CLS_FILES["classifier"]),
            "logs": tl.logs,
        })
        return result
    except _Stopped:
        _clear_stop(self.request.id)
        tl.log("Stopped by user")
        return {"status": "stopped", "stopped": True, "logs": tl.logs}
    finally:
        shutil.rmtree(cls_root, ignore_errors=True)


def build_label_preview(
    project_id: str,
    crop_class: str,
    label_classes: list | None = None,
    empty_label: str | None = "no_cover",
    min_overlap: float = 0.5,
    margin: float = 0.12,
    samples_per_class: int = 4,
    thumb: int = 160,
) -> dict:
    """Dry run of the crop-class / label-class flow: how many crops each
    label would get, how many were skipped (conflicts / empty), and a few
    thumbnail crops per label so the user can sanity-check before training.
    Nothing is written to disk."""
    import base64

    db = StateDBConnector()
    with db.get_session() as conn:
        img_rows, ann_rows = _fetch(db, conn, project_id)
    if not img_rows:
        return {"error": "No annotated images found"}

    plain = _group_plain(ann_rows)
    if not label_classes:
        label_classes = sorted({a["class_name"] for anns in plain.values() for a in anns} - {crop_class})
    samples_by_image, summary = derive_dataset_labels(
        plain, crop_class, label_classes, empty_label, min_overlap)
    summary["label_classes"] = label_classes
    summary["annotated_images"] = len(img_rows)

    by_id = {i["id"]: i for i in img_rows}
    shown: dict = {}
    for iid, samples in samples_by_image.items():
        for a in samples:
            label = a["state"]
            if len(shown.setdefault(label, [])) >= samples_per_class:
                continue
            img = cv2.imread(str(_resolve(by_id[iid])))
            if img is None:
                continue
            h, w = img.shape[:2]
            crop = crop_region(img, _xywh_to_xyxy(a["bbox"], w, h), margin)
            if crop is None:
                continue
            scale = thumb / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                shown[label].append(
                    "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    summary["samples"] = shown
    # A big no_cover share usually means cover boxes were never drawn.
    total = sum(summary["per_class"].values())
    if empty_label and total and summary["per_class"].get(empty_label, 0) / total > 0.6:
        summary["warning"] = (
            f"'{empty_label}' is over 60% of the samples -- check that the "
            "other label boxes were actually annotated.")
    if summary["conflict"] and summary["conflict"] >= 0.2 * (total + summary["conflict"]):
        top = ", ".join(f"{k} (x{v})" for k, v in summary["conflict_combos"].items())
        summary["conflict_warning"] = (
            f"{summary['conflict']} crops hold more than one class at once - {top} - so they get no label "
            "and are skipped. Untick one of those classes (for example a sub-box type like a 'cut' mark "
            "drawn inside a Cut_Cover) to keep them.")
    return summary
