"""
crop_cls_training.py
~~~~~~~~~~~~~~~~~~~~~
Trains the crop + classify pipeline (see services/cls_model.py):

  Stage 1 (mode="crop" only): YOLO detector on a single class
      (REGION_CLASS) -- every annotated box is relabeled to it, so the
      detector only learns WHERE the region is, never which state it is in.
  Stage 2: YOLO-cls on the cropped regions, one folder per state
      (mode="crop"), or on whole frames (mode="whole", a baseline to
      compare the crop approach against).

A box's state label is Annotation.state, falling back to class_name, so
existing "one class per state" annotations work unchanged.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

from .celery_app import celery_app
from .training import (
    YOLO,
    _build_yolo_dataset,
    _make_epoch_callback,
    _split_images,
    clahe_gamma_sharpen,
)
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from ..services.class_select import derive_dataset_labels
from ..services.cls_model import CLS_FILES, REGION_CLASS, cls_dir, crop_region



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


def _group_plain(ann_rows):
    """image_id -> [{class_name, bbox, points, ...}] for every usable box,
    with no state logic (used by the crop-class / label-class flow, where
    the label comes from nested boxes rather than Annotation.state)."""
    out: dict = {}
    for row in ann_rows:
        bbox = row.get("bbox")
        bbox = json.loads(bbox) if isinstance(bbox, str) else bbox
        if not bbox or len(bbox) != 4:
            continue
        points = row.get("points")
        points = json.loads(points) if isinstance(points, str) else points
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


def _build_cls_dataset(split_map, anns_by_image, root: Path, mode, margin, preprocess):
    """Write root/<split>/<state>/*.jpg. Returns {split: Counter(state)}."""
    counts = {s: Counter() for s in split_map}
    for split, imgs in split_map.items():
        for n, img_row in enumerate(imgs):
            anns = anns_by_image.get(img_row["id"], [])
            if not anns:
                continue
            img = cv2.imread(str(_resolve(img_row)))
            if img is None:
                continue
            if preprocess:
                img = clahe_gamma_sharpen(img)
            h, w = img.shape[:2]
            if mode == "whole":
                # One label per frame: the state of the largest box.
                big = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
                items = [(big["state"], img)]
            else:
                items = []
                for a in anns:
                    crop = crop_region(img, _xywh_to_xyxy(a["bbox"], w, h), margin)
                    if crop is not None:
                        items.append((a["state"], crop))
            for k, (state, crop) in enumerate(items):
                d = root / split / state
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{img_row['id']}_{n}_{k}.jpg"), crop)
                counts[split][state] += 1
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
    region_classes: list | None = None,    # only these class_names count as the region box
    det_model_name: str = "yolo11s.pt",
    cls_model_name: str = "yolo11s-cls.pt",
    det_epochs: int = 60,
    cls_epochs: int = 40,
    det_imgsz: int = 640,
    cls_imgsz: int = 224,
    margin: float = 0.12,
    preprocess: bool = True,
    batch: int = 32,
    aug_fliplr: float = 0.5,
    crop_class: str | None = None,         # big region box, e.g. "engine"
    label_classes: list | None = None,     # boxes inside it that name its state
    empty_label: str | None = "no_cover",  # label for a region with nothing inside (None = skip)
    min_overlap: float = 0.5,              # share of a label box that must lie inside the region
):
    if mode not in ("crop", "whole"):
        return {"error": "mode must be 'crop' or 'whole'"}

    db = StateDBConnector()
    with db.get_session() as conn:
        img_rows, ann_rows = _fetch(db, conn, project_id)
    if not img_rows:
        return {"error": "No annotated images found"}

    derive_summary = None
    if crop_class:
        if not label_classes:
            return {"error": "label_classes is required when crop_class is set"}
        anns_by_image, derive_summary = derive_dataset_labels(
            _group_plain(ann_rows), crop_class, label_classes, empty_label, min_overlap)
    else:
        anns_by_image = _group(ann_rows, region_classes)
    img_rows = [i for i in img_rows if anns_by_image.get(i["id"])]
    if not img_rows:
        return {"error": "No usable box annotations found"}

    states = sorted({a["state"] for anns in anns_by_image.values() for a in anns})
    if len(states) < 2:
        return {"error": f"Need at least 2 states to classify, found: {states}"}

    out_dir = cls_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    result: dict = {"status": "success", "mode": mode, "classes": states}
    if derive_summary:
        result["label_summary"] = derive_summary

    cls_root = Path(f"./temp_cls_{project_id}")
    det_dataset = None
    try:
        # ── Stage 1: region detector ─────────────────────────────
        if mode == "crop":
            relabeled = {
                iid: [{**a, "class_name": REGION_CLASS} for a in anns]
                for iid, anns in anns_by_image.items()
            }
            det_dataset, n_tr, n_va, n_te = _build_yolo_dataset(
                img_rows, relabeled, [REGION_CLASS], f"{project_id}_cropcls",
                preprocess=preprocess, imgsz=det_imgsz, task=self,
            )
            history, starts, stop = [], [], {"value": False}
            model = YOLO(det_model_name)
            model.add_callback(
                "on_fit_epoch_end",
                _make_epoch_callback(self, det_epochs, history, starts, stop),
            )
            self.update_state(state="STARTED", meta={
                "stage": "detector", "epoch": 0, "total_epochs": det_epochs,
                "history": [], "split": {"train": n_tr, "val": n_va, "test": n_te},
            })
            res = model.train(
                data=str(det_dataset / "data.yaml"), epochs=det_epochs, imgsz=det_imgsz,
                batch=0.9 if device == 0 else 8, cache=True, amp=device == 0, device=device,
                lr0=settings.seed_learning_rate, lrf=0.01, cos_lr=True, warmup_epochs=3,
                weight_decay=0.001, patience=20, fliplr=aug_fliplr, flipud=0.0,
                project=str(settings.model_dir / project_id), name="crop_cls_region",
                verbose=False, workers=0,
            )
            if stop["value"]:
                shutil.rmtree(res.save_dir, ignore_errors=True)
                from celery.exceptions import Ignore
                raise Ignore()
            shutil.copy(res.save_dir / "weights" / "best.pt", out_dir / CLS_FILES["detector"])
            result["detector_metrics"] = history[-1] if history else {}

        # ── Stage 2: classifier ──────────────────────────────────
        train_i, val_i, test_i = _split_images(img_rows, anns_by_image=anns_by_image)
        split_map = {"train": train_i, "val": val_i}
        if test_i:
            split_map["test"] = test_i
        self.update_state(state="STARTED", meta={
            "stage": "dataset", "epoch": 0, "total_epochs": cls_epochs, "history": []})
        shutil.rmtree(cls_root, ignore_errors=True)
        counts = _build_cls_dataset(split_map, anns_by_image, cls_root, mode, margin, preprocess)
        train_states = [s for s in states if counts["train"][s] > 0]
        if len(train_states) < 2:
            return {"error": f"Need training samples for at least 2 states, got {dict(counts['train'])}"}
        if sum(counts["val"].values()) == 0:   # tiny dataset: mirror train
            shutil.copytree(cls_root / "train", cls_root / "val")
            counts["val"] = counts["train"]

        history, starts, stop = [], [], {"value": False}
        model = YOLO(cls_model_name)
        model.add_callback(
            "on_fit_epoch_end",
            _make_epoch_callback(self, cls_epochs, history, starts, stop),
        )
        self.update_state(state="STARTED", meta={
            "stage": "classifier", "epoch": 0, "total_epochs": cls_epochs, "history": [],
            "samples": {s: dict(c) for s, c in counts.items()},
        })
        res = model.train(
            data=str(cls_root), epochs=cls_epochs, imgsz=cls_imgsz, batch=batch,
            device=device, amp=device == 0, lr0=settings.seed_learning_rate,
            cos_lr=True, patience=15, fliplr=aug_fliplr, flipud=0.0,
            project=str(settings.model_dir / project_id), name="crop_cls_classifier",
            verbose=False, workers=0,
        )
        if stop["value"]:
            shutil.rmtree(res.save_dir, ignore_errors=True)
            from celery.exceptions import Ignore
            raise Ignore()
        best = res.save_dir / "weights" / "best.pt"
        shutil.copy(best, out_dir / CLS_FILES["classifier"])

        confusion = _confusion(YOLO(str(best)), cls_root / "val", train_states)
        total = sum(sum(r.values()) for r in confusion.values())
        correct = sum(confusion[c][c] for c in confusion)

        meta = {
            "mode": mode, "preprocess": bool(preprocess), "margin": margin,
            "classes": train_states, "region_classes": region_classes,
            "crop_class": crop_class, "label_classes": label_classes,
            "empty_label": empty_label,
            "cls_imgsz": cls_imgsz,
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
        })
        return result
    finally:
        shutil.rmtree(cls_root, ignore_errors=True)
        if det_dataset is not None:
            shutil.rmtree(det_dataset, ignore_errors=True)


def build_label_preview(
    project_id: str,
    crop_class: str,
    label_classes: list,
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

    samples_by_image, summary = derive_dataset_labels(
        _group_plain(ann_rows), crop_class, label_classes, empty_label, min_overlap)
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
    return summary
