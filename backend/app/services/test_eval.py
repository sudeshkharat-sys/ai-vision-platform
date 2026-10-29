"""
test_eval.py
~~~~~~~~~~~~
Pure helpers (no cv2 / torch / DB) for the Testing panel: score a model's
predictions against an image's ground-truth boxes and decide pass / fail.

Boxes are normalized [xc, yc, w, h]; a pred also carries ``class_name`` and
``conf``.
"""

from __future__ import annotations


def _xyxy(b):
    xc, yc, w, h = b
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def score_detections(gts: list, preds: list, iou_thr: float = 0.5, allow_extra: bool = False) -> dict:
    """Greedy match (highest confidence first) of predictions to GT boxes.

    gts:   [{"class_name", "bbox"}]      preds: [{"class_name", "bbox", "conf"}]
    A pred is a hit when it overlaps a free GT box by >= iou_thr with the same
    class. Overlapping a GT box with another class counts as ``wrong_class``.
    Left-over GT boxes are ``missed``; left-over preds are ``extra`` (false
    boxes). The image passes when nothing is missed / wrong, and (unless
    ``allow_extra``) nothing extra.
    """
    free = list(range(len(gts)))
    hit = wrong = extra = 0
    problems = []
    for p in sorted(preds, key=lambda x: -x.get("conf", 0)):
        best, best_iou = None, 0.0
        for gi in free:
            v = iou(p["bbox"], gts[gi]["bbox"])
            if v >= iou_thr and v > best_iou:
                best, best_iou = gi, v
        if best is None:
            extra += 1
            problems.append({"kind": "extra", "class_name": p["class_name"], "bbox": p["bbox"]})
            continue
        free.remove(best)
        if p["class_name"] == gts[best]["class_name"]:
            hit += 1
        else:
            wrong += 1
            problems.append({"kind": "wrong_class", "class_name": p["class_name"],
                             "expected": gts[best]["class_name"], "bbox": p["bbox"]})
    for gi in free:
        problems.append({"kind": "missed", "class_name": gts[gi]["class_name"], "bbox": gts[gi]["bbox"]})
    missed = len(free)
    passed = missed == 0 and wrong == 0 and (allow_extra or extra == 0)
    return {"passed": passed, "hit": hit, "missed": missed, "wrong_class": wrong,
            "extra": extra, "problems": problems}


def summarize(rows: list) -> dict:
    """rows: per-image dicts with ``passed`` and the counts from score_detections."""
    n = len(rows)
    ok = sum(1 for r in rows if r["passed"])
    tot = lambda k: sum(r.get(k, 0) for r in rows)  # noqa: E731
    hits, missed, wrong, extra = tot("hit"), tot("missed"), tot("wrong_class"), tot("extra")
    gt_total = hits + missed + wrong
    pred_total = hits + wrong + extra
    return {
        "images": n, "passed": ok, "failed": n - ok,
        "pass_rate": round(ok / n, 4) if n else None,
        "hit": hits, "missed": missed, "wrong_class": wrong, "extra": extra,
        "recall": round(hits / gt_total, 4) if gt_total else None,
        "precision": round(hits / pred_total, 4) if pred_total else None,
    }


def score_classification(expected: str, got: dict) -> dict:
    """got: result of predict_crop_cls. Passes only for a confident, correct state."""
    status, state = got.get("status"), got.get("state")
    passed = status == "ok" and state == expected
    reason = ("ok" if passed else status if status != "ok" else f"predicted {state}, expected {expected}")
    return {"passed": passed, "reason": reason, "expected": expected, "predicted": state,
            "confidence": got.get("confidence"), "status": status}
