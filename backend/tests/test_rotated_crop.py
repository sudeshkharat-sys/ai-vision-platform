import ast
import pathlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_src = (pathlib.Path(__file__).resolve().parents[1] / "app/tasks/crop_cls_training.py").read_text()
_fn = next(n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name == "rotated_crop")
_ns = {"cv2": cv2}
exec(compile(ast.Module([_fn], []), "rotated_crop", "exec"), _ns)
rotated_crop = _ns["rotated_crop"]


def _img():
    img = np.zeros((200, 300, 3), np.uint8)
    img[80:120, 120:180] = (255, 255, 255)          # white block, centre 150,100
    return img


def test_same_window_size_as_plain_crop():
    out = rotated_crop(_img(), (100, 60, 200, 140), 0.0, 37)
    assert out.shape[:2] == (80, 100)


def test_zero_angle_is_plain_crop():
    img = _img()
    out = rotated_crop(img, (100, 60, 200, 140), 0.0, 0)
    assert np.array_equal(out, img[60:140, 100:200])


def test_ninety_degrees_moves_content():
    img = _img()
    out = rotated_crop(img, (100, 60, 200, 140), 0.0, 90)
    plain = img[60:140, 100:200]
    assert out.shape == plain.shape and not np.array_equal(out, plain)
    # the centre stays put: the white block still covers the middle pixel
    assert out[40, 50].tolist() == [255, 255, 255]


def test_degenerate_box_is_none():
    assert rotated_crop(_img(), (10, 10, 11, 11), 0.0, 10) is None
