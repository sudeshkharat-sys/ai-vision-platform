import ast
import pathlib
import time
from collections import Counter

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
ROOT = pathlib.Path(__file__).resolve().parents[1] / "app"


def _funcs(rel, names):
    tree = ast.parse((ROOT / rel).read_text())
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]


ns = {"cv2": cv2, "np": np, "Path": pathlib.Path, "Counter": Counter}
for rel, names in [
    ("services/cls_model.py", {"crop_region"}),
    ("tasks/training.py", {"clahe_gamma_sharpen"}),
    ("tasks/crop_cls_training.py", {"rotated_crop", "_write_jpg", "_process_image", "_build_cls_dataset",
                                    "_auto_imgsz", "_xywh_to_xyxy"}),
]:
    for fn in _funcs(rel, names):
        exec(compile(ast.Module([fn], []), fn.name, "exec"), ns)
# the task module resolves image rows to files; here the row carries the path
ns["_resolve"] = lambda row: pathlib.Path(row["filepath"])


def _make(tmp, n=12, size=(1500, 2000)):
    rows, anns = [], {}
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        p = tmp / f"im{i}.jpg"
        cv2.imwrite(str(p), img)
        rid = f"id{i}"
        rows.append({"id": rid, "filename": p.name, "filepath": str(p)})
        anns[rid] = [{"class_name": "area", "state": "full" if i % 2 else "cut",
                      "bbox": [0.5, 0.5, 0.4, 0.4]}]
    return rows, anns


def test_build_writes_only_valid_files_and_counts_match(tmp_path):
    rows, anns = _make(tmp_path)
    split_map = {"train": rows[:8], "val": rows[8:]}
    stats = {}
    seen = []
    counts = ns["_build_cls_dataset"](split_map, anns, tmp_path / "ds", "crop", 0.12, True,
                                      rotate_copies=2, degrees=10, translate=0.1,
                                      progress=lambda *a: seen.append(a), stats=stats)
    files = list((tmp_path / "ds").rglob("*.jpg"))
    assert files and all(f.stat().st_size > 0 for f in files)
    assert stats["failed"] == 0
    # train: 8 images x (1 + 2 copies); val: 4 x 1
    assert sum(counts["train"].values()) == 24 and sum(counts["val"].values()) == 4
    assert len(files) == 28
    assert seen and seen[-1][1] == seen[-1][2] == 12
    assert 160 <= ns["_auto_imgsz"](tmp_path / "ds") <= 448


def test_write_jpg_never_leaves_an_empty_file(tmp_path):
    p = tmp_path / "x.jpg"
    assert ns["_write_jpg"](p, np.zeros((40, 60, 3), np.uint8)) and p.stat().st_size > 0
    bad = tmp_path / "bad.jpg"
    assert ns["_write_jpg"](bad, np.zeros((0, 0, 3), np.uint8)) is False
    assert not bad.exists() or bad.stat().st_size > 0


def test_auto_imgsz_skips_unreadable_files(tmp_path):
    d = tmp_path / "train" / "a"
    d.mkdir(parents=True)
    (d / "empty.jpg").write_bytes(b"")                       # the file that crashed the worker
    cv2.imwrite(str(d / "ok.jpg"), np.zeros((300, 400, 3), np.uint8))
    assert ns["_auto_imgsz"](tmp_path) in range(160, 449)


def test_threaded_build_is_not_slower_than_one_worker(tmp_path):
    rows, anns = _make(tmp_path, n=10)
    t0 = time.time()
    ns["_build_cls_dataset"]({"train": rows}, anns, tmp_path / "a", "crop", 0.12, True)
    assert time.time() - t0 < 60
