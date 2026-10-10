import importlib.util
import pathlib
import types

_p = pathlib.Path(__file__).resolve().parents[1] / "app/services/train_metrics.py"
_spec = importlib.util.spec_from_file_location("train_metrics", _p)
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)


class T0:                       # stands in for a 0-d / n-d tensor
    def __init__(self, v): self.v = v
    def tolist(self): return self.v


def test_detection_vector_unchanged():
    t = types.SimpleNamespace(loss_items=T0([1.0, 2.0, 3.0]), loss_names=["box_loss", "cls_loss", "dfl_loss"])
    assert tm.trainer_losses(t) == {"box_loss": 1.0, "cls_loss": 2.0, "dfl_loss": 3.0}


def test_classifier_scalar_uses_epoch_mean():
    t = types.SimpleNamespace(loss_items=T0(0.9), tloss=T0(0.25), loss_names=["loss"])
    assert tm.trainer_losses(t) == {"loss": 0.25}
    t = types.SimpleNamespace(loss_items=T0(0.9), loss_names=["loss"])      # no tloss
    assert tm.trainer_losses(t) == {"loss": 0.9}


def test_missing_is_empty():
    assert tm.trainer_losses(types.SimpleNamespace()) == {}
