"""Pure helpers for reading live numbers off an Ultralytics trainer."""

from __future__ import annotations


def _as_list(v):
    """tensor / ndarray / scalar / list -> flat list of floats-ish (None stays None)."""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v) if isinstance(v, (list, tuple)) else [v]


def trainer_losses(trainer, default_names=("box_loss", "cls_loss", "dfl_loss")) -> dict:
    """{loss name: value} for the epoch just finished.

    Detection / segmentation trainers keep a vector of loss items (box, cls,
    dfl ...). The classification trainer keeps ONE scalar (0-d tensor), which
    the vector code could not zip with the names, so the train-loss curve was
    silently empty for classifiers. A scalar is wrapped here, and for it the
    epoch-mean running loss (``tloss``) is preferred over the last batch's.
    """
    vals = _as_list(getattr(trainer, "loss_items", None))
    if vals is None:
        return {}
    names = list(getattr(trainer, "loss_names", None) or default_names)
    if len(vals) == 1:
        mean = _as_list(getattr(trainer, "tloss", None))
        if mean and len(mean) == 1:
            vals = mean
        names = names if len(names) == 1 else ["loss"]
    return dict(zip(names, vals))
