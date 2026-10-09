"""
data_segments.py
~~~~~~~~~~~~~~~~~
Pure helpers (no cv2 / DB imports) for "Data Segments": named rules over the
annotations an image carries, used to pick which images feed a training run
and to give each of them an image-level label.

A segment is a dict::

    {"name": "Fully locked",          # unique per project
     "label": "locked",               # image-level class for the classifier
     "counts": {"door": {"min": 1, "max": 1},
                "lock": {"min": 2, "max": 2},
                "unlock": {"min": 0, "max": 0}},
     "allow_other": False,            # other classes may be present?
     "total": None}                   # optional {"min": n, "max": n}

Classes not listed in ``counts`` must be absent unless ``allow_other`` is
true. ``max`` of None means "no upper bound". An image goes to the first
matching segment; images matching several are counted in ``overlaps``.
"""

from __future__ import annotations

from collections import Counter


def _int_or_none(v):
    if v is None or v == "":
        return None
    return max(0, int(v))


def normalize_segment(seg: dict) -> dict:
    """Validate + clean one segment. Raises ValueError with a readable message."""
    name = str(seg.get("name") or "").strip()
    if not name:
        raise ValueError("Every segment needs a name")
    label = str(seg.get("label") or name).strip()
    counts = {}
    for cls, rng in (seg.get("counts") or {}).items():
        rng = rng or {}
        lo, hi = _int_or_none(rng.get("min")) or 0, _int_or_none(rng.get("max"))
        if hi is not None and hi < lo:
            raise ValueError(f"Segment '{name}': max < min for class '{cls}'")
        counts[str(cls)] = {"min": lo, "max": hi}
    total = seg.get("total")
    if total:
        lo, hi = _int_or_none(total.get("min")) or 0, _int_or_none(total.get("max"))
        if hi is not None and hi < lo:
            raise ValueError(f"Segment '{name}': total max < min")
        total = {"min": lo, "max": hi}
    else:
        total = None
    if not counts and not total:
        raise ValueError(f"Segment '{name}' has no rules")
    return {"name": name, "label": label, "counts": counts,
            "allow_other": bool(seg.get("allow_other")), "total": total}


def normalize_segments(segments) -> list:
    out, seen = [], set()
    for seg in segments or []:
        s = normalize_segment(seg)
        if s["name"] in seen:
            raise ValueError(f"Duplicate segment name '{s['name']}'")
        seen.add(s["name"])
        out.append(s)
    return out


def image_matches(class_counts: Counter, seg: dict) -> bool:
    """Does an image with these per-class box counts satisfy the segment?"""
    for cls, rng in seg["counts"].items():
        n = class_counts.get(cls, 0)
        if n < rng["min"] or (rng["max"] is not None and n > rng["max"]):
            return False
    if not seg.get("allow_other") and any(
            c not in seg["counts"] and n > 0 for c, n in class_counts.items()):
        return False
    t = seg.get("total")
    if t:
        n = sum(class_counts.values())
        if n < t["min"] or (t["max"] is not None and n > t["max"]):
            return False
    return True


def assign_segments(anns_by_image: dict, segments: list, image_ids=None):
    """Assign every image (``image_ids`` or all keys of anns_by_image; images
    without annotations count as empty) to its first matching segment.

    Returns (assignment, summary):
      assignment: {image_id: segment}
      summary: {"segments": [{"name","label","images"}], "matched": n,
                "unmatched": n, "overlaps": n,
                "unmatched_combos": {"door + lock": n, ...} (top 8)}
    """
    segments = normalize_segments(segments)
    ids = list(image_ids) if image_ids is not None else list(anns_by_image)
    per_seg, assignment = Counter(), {}
    overlaps, unmatched, combos = 0, 0, Counter()
    by_total = Counter()
    for iid in ids:
        counts = Counter(a.get("class_name") for a in anns_by_image.get(iid, []))
        by_total[sum(counts.values())] += 1
        hits = [s for s in segments if image_matches(counts, s)]
        if not hits:
            unmatched += 1
            combos[" + ".join(f"{c} x{n}" if n > 1 else c
                              for c, n in sorted(counts.items())) or "(no annotations)"] += 1
            continue
        if len(hits) > 1:
            overlaps += 1
        assignment[iid] = hits[0]
        per_seg[hits[0]["name"]] += 1
    return assignment, {
        "segments": [{"name": s["name"], "label": s["label"], "images": per_seg[s["name"]]}
                     for s in segments],
        "matched": len(assignment),
        "unmatched": unmatched,
        "overlaps": overlaps,
        "unmatched_combos": dict(combos.most_common(8)),
        # images per number of annotations, e.g. {1: 12, 2: 40, 3: 300}
        "by_annotation_count": {str(k): v for k, v in sorted(by_total.items())},
    }


def filter_by_segments(img_rows, anns_by_image, segments):
    """Keep only the images that match a segment.

    Returns (img_rows, anns_by_image, labels, summary) where labels is
    {image_id: segment label}. The DB is never touched.
    """
    assignment, summary = assign_segments(
        anns_by_image, segments, image_ids=[i["id"] for i in img_rows])
    kept = [i for i in img_rows if i["id"] in assignment]
    new_anns = {iid: anns_by_image.get(iid, []) for iid in assignment}
    return kept, new_anns, {iid: s["label"] for iid, s in assignment.items()}, summary


def label_samples(anns_by_image: dict, labels: dict, crop_class=None):
    """Image-level labels -> classifier samples ({class_name, state, bbox}).

    With ``crop_class`` every box of that class becomes a sample labelled by
    its image's segment (e.g. each door crop gets "partial"). Without it the
    whole frame is one sample.
    """
    out = {}
    for iid, label in labels.items():
        if crop_class:
            boxes = [a for a in anns_by_image.get(iid, []) if a.get("class_name") == crop_class]
            samples = [{**a, "state": label} for a in boxes]
        else:
            samples = [{"class_name": label, "state": label, "bbox": [0.5, 0.5, 1.0, 1.0],
                        "points": None, "annotation_type": "bbox", "source": "segment"}]
        if samples:
            out[iid] = samples
    return out
