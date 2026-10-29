import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "app/services/test_eval.py"
_spec = importlib.util.spec_from_file_location("test_eval", _p)
te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(te)
iou, score_classification, score_detections, summarize = te.iou, te.score_classification, te.score_detections, te.summarize

B = [0.5, 0.5, 0.2, 0.2]


def gt(c="a", b=B):
    return {"class_name": c, "bbox": b}


def pr(c="a", b=B, conf=0.9):
    return {"class_name": c, "bbox": b, "conf": conf}


def test_iou():
    assert iou(B, B) == 1.0
    assert iou(B, [0.9, 0.9, 0.1, 0.1]) == 0.0


def test_perfect():
    assert score_detections([gt()], [pr()])["passed"]


def test_missed_extra_wrong():
    r = score_detections([gt()], [])
    assert not r["passed"] and r["missed"] == 1
    r = score_detections([gt()], [pr(), pr(b=[0.1, 0.1, 0.1, 0.1])])
    assert not r["passed"] and r["extra"] == 1
    assert score_detections([gt()], [pr(), pr(b=[0.1, 0.1, 0.1, 0.1])], allow_extra=True)["passed"]
    r = score_detections([gt("a")], [pr("b")])
    assert r["wrong_class"] == 1 and not r["passed"]


def test_summary():
    rows = [score_detections([gt()], [pr()]), score_detections([gt()], [])]
    s = summarize(rows)
    assert s["passed"] == 1 and s["failed"] == 1 and s["recall"] == 0.5


def test_classification():
    assert score_classification("x", {"status": "ok", "state": "x"})["passed"]
    assert not score_classification("x", {"status": "uncertain", "state": "x"})["passed"]
    assert not score_classification("x", {"status": "ok", "state": "y"})["passed"]


def test_variants_keep_shape_and_count():
    import numpy as np
    vp = pathlib.Path(__file__).resolve().parents[1] / "app/services/test_variants.py"
    sp = importlib.util.spec_from_file_location("test_variants", vp)
    tv = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(tv)
    img = (np.random.default_rng(1).random((90, 160, 3)) * 255).astype(np.uint8)
    for n in (0, 1, 2, 3, 9):
        out = tv.make_variants(img, n)
        assert len(out) == min(n, 3)
        assert all(o.shape == img.shape and o.dtype == np.uint8 for o in out)
    a, b = tv.make_variants(img, 2)
    assert not np.array_equal(a, img) and not np.array_equal(b, img)
