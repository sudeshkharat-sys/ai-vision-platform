"""
Tesseract LSTM fine-tuning (the "tesstrain" path).

Workflow
--------
1. Read the same per-character box annotations used by the CNN trainer.
2. Group the character boxes of each photo into text LINES (Tesseract's
   LSTM reads whole lines, not single characters); crop each line and
   write it as a ground-truth pair:  line_xxx.png + line_xxx.gt.txt.
3. Download Google's pretrained eng.traineddata (tessdata_best — the
   float model, the only variant that can be fine-tuned) on first use.
4. Fine-tune with the official training binaries (same flow as the
   tesstrain Makefile): tesseract lstm.train → .lstmf files,
   combine_tessdata -e → eng.lstm, lstmtraining --continue_from …
5. Evaluate character accuracy on held-out lines and export
   <lang>.traineddata — a drop-in file for flutter_tesseract_ocr.
"""
from .celery_app import celery_app
from .training import _fetch_training_data, _group_annotations, _safe_float
from ..config import settings

import json
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import redis as redis_lib


TESSDATA_BEST_ENG_URL = (
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"
)
CUSTOM_LANG = "engraved"

REQUIRED_BINARIES = ("tesseract", "combine_tessdata", "lstmtraining")


def _missing_binaries():
    return [b for b in REQUIRED_BINARIES if shutil.which(b) is None]


def _get_tessdata_best(task=None) -> Path:
    """Download (once) and return the dir holding the float eng.traineddata."""
    tessdata = settings.model_dir / "tessdata_best"
    target = tessdata / "eng.traineddata"
    if target.exists() and target.stat().st_size > 10_000_000:
        return tessdata
    tessdata.mkdir(parents=True, exist_ok=True)
    if task:
        try:
            task.update_state(state="STARTED", meta={
                "phase": "preparing",
                "detail": "downloading pretrained eng.traineddata (tessdata_best, ~15 MB, first time only)",
            })
        except Exception:
            pass
    import urllib.request
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(TESSDATA_BEST_ENG_URL, tmp)
    tmp.rename(target)
    return tessdata


# ── Line building from per-character boxes ────────────────────────

def _boxes_to_lines(anns, iw, ih):
    """
    Group single-character annotations into text lines.

    Returns a list of lines; each line is a list of
    (label, x1, y1, x2, y2, points_px) sorted left-to-right, and lines are
    sorted top-to-bottom. points_px is the annotation's polygon vertices in
    absolute pixel coordinates (a list of (x, y) tuples) when the source
    annotation is a polygon, else None -- x1..y2 stay the axis-aligned
    envelope either way (used for line grouping/sorting, which tolerates
    the extra slack fine), while points_px lets a consumer that needs a
    single clean character crop -- not just an axis-aligned box that can
    overlap a neighboring character on a tilted plate -- de-skew to the
    actual traced outline instead.
    """
    chars = []
    for ann in anns:
        if not ann["bbox"]:
            continue
        label = str(ann["class_name"]).strip().upper()
        if len(label) != 1:
            continue  # region boxes like "plate" don't belong to OCR
        xc, yc, w, h = ann["bbox"]
        x1 = (xc - w / 2) * iw
        y1 = (yc - h / 2) * ih
        x2 = (xc + w / 2) * iw
        y2 = (yc + h / 2) * ih
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        points_px = None
        if ann.get("annotation_type") == "polygon" and ann.get("points") and len(ann["points"]) >= 3:
            points_px = [(px * iw, py * ih) for px, py in ann["points"]]
        chars.append((label, x1, y1, x2, y2, points_px))
    if not chars:
        return []

    med_h = float(np.median([c[4] - c[2] for c in chars]))
    chars.sort(key=lambda c: (c[2] + c[4]) / 2)

    lines = []
    for c in chars:
        cy = (c[2] + c[4]) / 2
        placed = False
        for line in lines:
            ly = np.mean([(x[2] + x[4]) / 2 for x in line])
            if abs(cy - ly) < med_h * 0.6:
                line.append(c)
                placed = True
                break
        if not placed:
            lines.append([c])

    for line in lines:
        line.sort(key=lambda c: c[1])
    lines.sort(key=lambda l: np.mean([(c[2] + c[4]) / 2 for c in l]))
    return lines


def _normalize_line_image(gray):
    """
    Make a line crop look the way Tesseract expects: contrast-enhanced,
    DARK text on a LIGHT background, fixed 48-px height.

    Engraved characters can read lighter or darker than the metal, so the
    polarity is decided per line from the Otsu split: the minority pixels
    are assumed to be the strokes, and the image is inverted when those
    strokes are lighter than the background. The exact same normalization
    runs at prediction time, so train and inference always match.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    thr, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_white = int(cv2.countNonZero(bw))
    n_total = bw.size
    stroke_is_light = n_white < n_total - n_white  # minority == strokes
    if stroke_is_light:
        g = 255 - g

    h, w = g.shape
    scale = 48.0 / h
    g = cv2.resize(g, (max(8, int(round(w * scale))), 48),
                   interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    # Small white border helps Tesseract's line finder
    g = cv2.copyMakeBorder(g, 8, 8, 12, 12, cv2.BORDER_CONSTANT,
                           value=int(np.percentile(g, 90)))
    return g


def _crop_line(img, line, margin_y=0.35, margin_x=0.15):
    """Crop the union box of a line's characters (with margin) as grayscale."""
    ih, iw = img.shape[:2]
    x1 = min(c[1] for c in line)
    y1 = min(c[2] for c in line)
    x2 = max(c[3] for c in line)
    y2 = max(c[4] for c in line)
    h = y2 - y1
    mx, my = h * margin_x, h * margin_y
    x1 = max(0, int(round(x1 - mx)))
    y1 = max(0, int(round(y1 - my)))
    x2 = min(iw, int(round(x2 + mx)))
    y2 = min(ih, int(round(y2 + my)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return _normalize_line_image(crop)


# ── Evaluation helpers ────────────────────────────────────────────

def _char_accuracy(truth: str, pred: str):
    """
    Character accuracy of pred vs truth plus confusion pairs, using
    sequence alignment (so an inserted/dropped character doesn't shift
    the whole comparison).
    """
    sm = SequenceMatcher(a=truth, b=pred, autojunk=False)
    correct = 0
    confusions = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            correct += a2 - a1
        elif op == "replace" and (a2 - a1) == (b2 - b1):
            for t, p in zip(truth[a1:a2], pred[b1:b2]):
                confusions.append((t, p))
    return correct, len(truth), confusions


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ── Celery task ───────────────────────────────────────────────────

@celery_app.task(name="app.tasks.tesseract_training.train_tesseract_model", bind=True)
def train_tesseract_model(
    self,
    project_id: str,
    max_iterations: int = 800,
    val_ratio: float = 0.2,
    learning_rate: float = 1e-4,
):
    """
    Fine-tune Google's pretrained eng LSTM on this project's labeled
    engraved lines and export <CUSTOM_LANG>.traineddata.
    """
    from ..connectors.statedb_connector import StateDBConnector

    missing = _missing_binaries()
    if missing:
        return {"error": "Tesseract training tools not installed on the server: "
                         f"missing {', '.join(missing)}. Rebuild the backend image "
                         "(the Dockerfile installs the tesseract-ocr package)."}

    db = StateDBConnector()
    rng = random.Random(42)

    # Clear any stale stop flag from a previous cancel
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

    # ── Phase 2: build ground-truth line images ──────────────────
    work = settings.model_dir / project_id / "ocr_tesseract" / "_train_tmp"
    shutil.rmtree(work, ignore_errors=True)
    gt_dir = work / "ground-truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    samples = []  # (png_path, gt_text)
    char_counts = defaultdict(int)
    total_imgs = len(img_rows)
    for idx, img_row in enumerate(img_rows):
        real_path = Path(".") / img_row["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
        img = cv2.imread(str(real_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for li, line in enumerate(_boxes_to_lines(anns_by_image.get(img_row["id"], []), iw, ih)):
            crop = _crop_line(img, line)
            if crop is None:
                continue
            text = "".join(c[0] for c in line)
            base = gt_dir / f"line_{len(samples):05d}"
            cv2.imwrite(str(base.with_suffix(".png")), crop)
            base.with_suffix(".gt.txt").write_text(text + "\n")
            samples.append((base, text))
            for ch in text:
                char_counts[ch] += 1

        if (idx + 1) % 5 == 0 or idx + 1 == total_imgs:
            try:
                self.update_state(state="STARTED", meta={
                    "phase": "extracting_lines",
                    "current": idx + 1, "total": total_imgs,
                    "lines": len(samples),
                })
            except Exception:
                pass

    if len(samples) < 4:
        shutil.rmtree(work, ignore_errors=True)
        return {"error": f"Only {len(samples)} text lines could be built from the labeled "
                         "boxes — Tesseract fine-tuning needs at least 4 (more is much better). "
                         "Label more photos (one box per character), then retry."}

    # ── Phase 3: pretrained data + .lstmf generation ─────────────
    tessdata = _get_tessdata_best(task=self)

    rng.shuffle(samples)
    n_val = max(1, int(round(len(samples) * val_ratio)))
    val_samples, train_samples = samples[:n_val], samples[n_val:]

    def make_lstmf(items, phase_name):
        paths = []
        for i, (base, _text) in enumerate(items):
            r = _run([
                "tesseract", str(base.with_suffix(".png")), str(base),
                "--tessdata-dir", str(tessdata), "-l", "eng",
                "--psm", "13", "lstm.train",
            ])
            lstmf = base.with_suffix(".lstmf")
            if r.returncode == 0 and lstmf.exists():
                paths.append(lstmf)
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                try:
                    self.update_state(state="STARTED", meta={
                        "phase": phase_name, "current": i + 1, "total": len(items),
                    })
                except Exception:
                    pass
        return paths

    train_lstmf = make_lstmf(train_samples, "preparing_train_lines")
    val_lstmf = make_lstmf(val_samples, "preparing_eval_lines")
    if len(train_lstmf) < 2:
        shutil.rmtree(work, ignore_errors=True)
        return {"error": "Could not generate Tesseract training files from the line "
                         "images (lstm.train failed). Check that tesseract-ocr is "
                         "installed correctly on the server."}

    train_list = work / "train.listfile"
    train_list.write_text("\n".join(str(p) for p in train_lstmf) + "\n")
    eval_list = work / "eval.listfile"
    eval_list.write_text("\n".join(str(p) for p in (val_lstmf or train_lstmf)) + "\n")

    # Extract the LSTM network out of the pretrained traineddata
    eng_lstm = work / "eng.lstm"
    r = _run(["combine_tessdata", "-e", str(tessdata / "eng.traineddata"), str(eng_lstm)])
    if r.returncode != 0 or not eng_lstm.exists():
        shutil.rmtree(work, ignore_errors=True)
        return {"error": f"combine_tessdata failed to extract the pretrained LSTM: {r.stderr[-400:]}"}

    # ── Phase 4: lstmtraining (fine-tune) with live progress ─────
    ckpt_prefix = work / "ft"
    cmd = [
        "lstmtraining",
        "--model_output", str(ckpt_prefix),
        "--continue_from", str(eng_lstm),
        "--traineddata", str(tessdata / "eng.traineddata"),
        "--train_listfile", str(train_list),
        "--eval_listfile", str(eval_list),
        "--max_iterations", str(int(max_iterations)),
        "--learning_rate", str(learning_rate),
        "--debug_interval", "0",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    iter_re = re.compile(r"At iteration (\d+)/(\d+)/(\d+).*?char train=([\d.]+)%")
    history, stopped = [], False
    last_push = 0.0
    for line in proc.stdout:
        m = iter_re.search(line)
        if m:
            entry = {
                "iteration": int(m.group(3)),
                "char_error": _safe_float(m.group(4)),
            }
            if not history or history[-1]["iteration"] != entry["iteration"]:
                history.append(entry)
        now = time.time()
        if now - last_push > 1.5:
            last_push = now
            try:
                self.update_state(state="STARTED", meta={
                    "phase": "training",
                    "engine": "tesseract",
                    "iteration": history[-1]["iteration"] if history else 0,
                    "total_iterations": int(max_iterations),
                    "char_error": history[-1]["char_error"] if history else None,
                    "tess_history": history[-120:],
                })
            except Exception:
                pass
            # Cooperative cancel — same Redis flag as the other trainers
            try:
                _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
                if _r.get(f"stop_training:{self.request.id}"):
                    _r.delete(f"stop_training:{self.request.id}")
                    stopped = True
                    proc.terminate()
                    break
            except Exception:
                pass
    proc.wait(timeout=120)

    if stopped:
        shutil.rmtree(work, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    checkpoint = ckpt_prefix.with_suffix(".checkpoint")
    if not checkpoint.exists():
        # lstmtraining writes <prefix>_checkpoint on some versions
        alt = Path(str(ckpt_prefix) + "_checkpoint")
        checkpoint = alt if alt.exists() else None
    if checkpoint is None:
        shutil.rmtree(work, ignore_errors=True)
        return {"error": "lstmtraining produced no checkpoint — training failed. "
                         "Try more labeled lines or fewer iterations."}

    # ── Phase 5: bake the final traineddata ──────────────────────
    out_dir = settings.model_dir / project_id / "ocr_tesseract"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_model = out_dir / f"{CUSTOM_LANG}.traineddata"
    r = _run([
        "lstmtraining", "--stop_training",
        "--continue_from", str(checkpoint),
        "--traineddata", str(tessdata / "eng.traineddata"),
        "--model_output", str(final_model),
    ])
    if r.returncode != 0 or not final_model.exists():
        shutil.rmtree(work, ignore_errors=True)
        return {"error": f"Failed to export the final traineddata: {r.stderr[-400:]}"}

    # ── Phase 6: evaluate on the held-out lines ──────────────────
    try:
        self.update_state(state="STARTED", meta={"phase": "evaluating",
                                                 "total": len(val_samples)})
    except Exception:
        pass

    correct = total = 0
    confusions = defaultdict(int)
    line_results = []
    for base, truth in val_samples:
        r = _run([
            "tesseract", str(base.with_suffix(".png")), "stdout",
            "--tessdata-dir", str(out_dir), "-l", CUSTOM_LANG,
            "--psm", "7",  # single text line
            "-c", "tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ])
        pred = re.sub(r"\s+", "", r.stdout.upper()) if r.returncode == 0 else ""
        c, t, conf = _char_accuracy(truth, pred)
        correct += c
        total += t
        for pair in conf:
            confusions[f"{pair[0]}->{pair[1]}"] += 1
        line_results.append({"truth": truth, "predicted": pred})

    val_accuracy = round(correct / total, 4) if total else None
    top_confusions = sorted(confusions.items(), key=lambda kv: -kv[1])[:10]

    meta = {
        "engine": "tesseract",
        "lang": CUSTOM_LANG,
        "base_model": "eng (tessdata_best)",
        "iterations": int(max_iterations),
        "final_char_train_error_pct": history[-1]["char_error"] if history else None,
        "val_accuracy": val_accuracy,
        "eval_lines": line_results[:50],
        "top_confusions": [{"pair": p, "count": c} for p, c in top_confusions],
        "char_counts": dict(sorted(char_counts.items())),
        "train_lines": len(train_samples),
        "val_lines": len(val_samples),
    }
    (out_dir / "tess_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(work, ignore_errors=True)

    return {
        "status": "success",
        "engine": "tesseract",
        "model_path": str(final_model),
        "lang": CUSTOM_LANG,
        "val_accuracy": val_accuracy,
        "top_confusions": meta["top_confusions"],
        "eval_lines": line_results[:20],
        "train_lines": len(train_samples),
        "val_lines": len(val_samples),
        "iterations_run": history[-1]["iteration"] if history else 0,
        "final_char_train_error_pct": meta["final_char_train_error_pct"],
        "traineddata_size_kb": round(final_model.stat().st_size / 1024, 1),
    }
