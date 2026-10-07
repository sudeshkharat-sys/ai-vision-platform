"""
class_select.py
~~~~~~~~~~~~~~~~
Pure helpers (no cv2 / DB imports) for working with a subset of a
project's annotation classes:

* select_train_classes -- let a detector/segmenter train on only the
  classes the user ticked (e.g. just ``engine``) while every other class
  stays untouched in the annotations table.
* derive_region_labels -- build classifier samples from nested boxes: a
  big "crop" box (e.g. ``engine``) is labelled by whichever "label" box
  (``full_cover`` / ``cut_cover``) lies inside it, or by ``empty_label``
  (``no_cover``) when nothing does.

Boxes are normalized [xc, yc, w, h], same as Annotation.bbox.
"""

from __future__ import annotations

from collections import Counter


def select_train_classes(classes, anns_by_image, img_rows, train_classes):
    """Restrict a training run to ``train_classes``.

    Returns (classes, anns_by_image, img_rows):
      * classes keeps the project's original order, filtered to the ticked ones;
      * annotations of other classes are dropped from the view (the DB is
        never touched);
      * images left with no annotation after filtering are dropped, so a
        frame that only carries e.g. cover boxes isn't fed to an
        ``engine`` detector as a false negative.

    ``train_classes`` empty/None means "no selection" -> inputs unchanged.
    Raises ValueError if none of the ticked classes exist.
    """
    if not train_classes:
        return classes, anns_by_image, img_rows
    wanted = set(train_classes)
    kept_classes = [c for c in classes if c in wanted]
    if not kept_classes:
        raise ValueError(
            f"None of the selected classes {sorted(wanted)} exist in this project"
        )
    new_anns = {}
    for iid, anns in anns_by_image.items():
        kept = [a for a in anns if a.get("class_name") in wanted]
        if kept:
            new_anns[iid] = kept
    new_imgs = [i for i in img_rows if i["id"] in new_anns]
    return kept_classes, new_anns, new_imgs


def _area(b):
    return max(b[2], 0.0) * max(b[3], 0.0)


def _inside_fraction(inner, outer):
    """Fraction of ``inner``'s area that lies within ``outer`` (both xc,yc,w,h)."""
    ix1, iy1 = inner[0] - inner[2] / 2, inner[1] - inner[3] / 2
    ix2, iy2 = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2
    ox1, oy1 = outer[0] - outer[2] / 2, outer[1] - outer[3] / 2
    ox2, oy2 = outer[0] + outer[2] / 2, outer[1] + outer[3] / 2
    w = min(ix2, ox2) - max(ix1, ox1)
    h = min(iy2, oy2) - max(iy1, oy1)
    a = _area(inner)
    if w <= 0 or h <= 0 or a <= 0:
        return 0.0
    return (w * h) / a


def derive_region_labels(
    anns,
    crop_class: str,
    label_classes,
    empty_label: str | None = "no_cover",
    min_overlap: float = 0.5,
):
    """Label every ``crop_class`` box in one image by the boxes inside it.

    A label box belongs to a crop box when at least ``min_overlap`` of its
    area lies inside it; a label box that fits several crop boxes goes to
    the one containing the largest share (ties -> the smaller crop box).

    Returns (samples, stats). Each sample is the crop annotation dict
    plus ``"state"`` (its label). stats counts, for this image:
    {"empty": crop boxes with nothing inside (labelled ``empty_label``, or
    skipped when it is falsy), "conflict": crop boxes holding two
    different label classes (skipped)}.
    """
    label_set = set(label_classes)
    crops = [a for a in anns if a.get("class_name") == crop_class and _valid(a)]
    labels = [a for a in anns if a.get("class_name") in label_set and _valid(a)]

    inside: list[list[str]] = [[] for _ in crops]
    for lab in labels:
        best, best_key = None, None
        for idx, crop in enumerate(crops):
            frac = _inside_fraction(lab["bbox"], crop["bbox"])
            if frac < min_overlap:
                continue
            key = (frac, -_area(crop["bbox"]))
            if best_key is None or key > best_key:
                best, best_key = idx, key
        if best is not None:
            inside[best].append(lab["class_name"])

    samples, stats = [], Counter()
    for crop, names in zip(crops, inside):
        distinct = sorted(set(names))
        if len(distinct) > 1:
            stats["conflict"] += 1
            continue
        if distinct:
            state = distinct[0]
        elif empty_label:
            state = empty_label
            stats["empty"] += 1
        else:
            stats["empty"] += 1
            continue
        samples.append({**crop, "state": state})
    return samples, stats


def _valid(a):
    b = a.get("bbox")
    return bool(b) and len(b) == 4 and b[2] > 0 and b[3] > 0


def derive_dataset_labels(anns_by_image, crop_class, label_classes,
                          empty_label="no_cover", min_overlap=0.5):
    """derive_region_labels over every image.

    Returns (samples_by_image, summary). samples_by_image only keeps
    images that yield at least one sample. summary:
      {"per_class": {label: n}, "images": n_images_used,
       "empty": n, "conflict": n, "images_without_crop": n}
    """
    out, per_class = {}, Counter()
    empty = conflict = no_crop = 0
    for iid, anns in anns_by_image.items():
        samples, st = derive_region_labels(
            anns, crop_class, label_classes, empty_label, min_overlap)
        empty += st["empty"]
        conflict += st["conflict"]
        if not any(a.get("class_name") == crop_class for a in anns):
            no_crop += 1
        if samples:
            out[iid] = samples
            per_class.update(s["state"] for s in samples)
    return out, {
        "per_class": dict(per_class),
        "images": len(out),
        "empty": empty,
        "conflict": conflict,
        "images_without_crop": no_crop,
    }
