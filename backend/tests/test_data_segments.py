import importlib.util
import pathlib

import pytest

_p = pathlib.Path(__file__).resolve().parents[1] / "app/services/data_segments.py"
_spec = importlib.util.spec_from_file_location("data_segments", _p)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def a(name):
    return {"class_name": name, "bbox": [0.5, 0.5, 0.2, 0.2]}


def seg(name, label, **counts):
    return {"name": name, "label": label,
            "counts": {c: {"min": n, "max": n} for c, n in counts.items()}}


SEGS = [seg("Locked", "locked", door=1, lock=2, unlock=0),
        seg("Unlocked", "unlocked", door=1, unlock=2, lock=0),
        seg("Partial", "partial", door=1, lock=1, unlock=1)]

ANNS = {
    1: [a("door"), a("lock"), a("lock")],
    2: [a("door"), a("unlock"), a("unlock")],
    3: [a("door"), a("lock"), a("unlock")],
    4: [a("door"), a("lock")],            # only one lock visible -> no segment
    5: [a("door"), a("lock"), a("lock"), a("tyre")],   # extra class -> excluded
    6: [],
}


def test_assigns_each_image_to_its_segment():
    asg, summ = ds.assign_segments(ANNS, SEGS)
    assert {i: s["label"] for i, s in asg.items()} == {1: "locked", 2: "unlocked", 3: "partial"}
    assert summ["matched"] == 3 and summ["unmatched"] == 3
    assert summ["by_annotation_count"] == {"0": 1, "2": 1, "3": 3, "4": 1}
    assert [s["images"] for s in summ["segments"]] == [1, 1, 1]
    assert summ["unmatched_combos"]["door + lock"] == 1


def test_allow_other_and_total():
    s = seg("L", "locked", door=1, lock=2)
    s["allow_other"] = True
    assert ds.assign_segments(ANNS, [s])[1]["matched"] == 2     # image 1 and 5
    s["total"] = {"min": 3, "max": 3}
    assert ds.assign_segments(ANNS, [s])[1]["matched"] == 1


def test_open_ended_max_and_overlap():
    s1 = {"name": "any lock", "counts": {"lock": {"min": 1, "max": None}}, "allow_other": True}
    asg, summ = ds.assign_segments(ANNS, [s1, SEGS[0]])
    assert summ["overlaps"] >= 1 and asg[1]["name"] == "any lock"


def test_filter_and_labels():
    imgs = [{"id": i} for i in ANNS]
    kept, anns, labels, _ = ds.filter_by_segments(imgs, ANNS, SEGS)
    assert [i["id"] for i in kept] == [1, 2, 3] and set(anns) == {1, 2, 3}
    crops = ds.label_samples(anns, labels, crop_class="door")
    assert {k: [x["state"] for x in v] for k, v in crops.items()} == {
        1: ["locked"], 2: ["unlocked"], 3: ["partial"]}
    whole = ds.label_samples(anns, labels)
    assert whole[3][0]["state"] == "partial" and whole[3][0]["bbox"] == [0.5, 0.5, 1.0, 1.0]


def test_validation():
    with pytest.raises(ValueError):
        ds.normalize_segment({"name": "", "counts": {"a": {}}})
    with pytest.raises(ValueError):
        ds.normalize_segment({"name": "x", "counts": {}})
    with pytest.raises(ValueError):
        ds.normalize_segment({"name": "x", "counts": {"a": {"min": 2, "max": 1}}})
    with pytest.raises(ValueError):
        ds.normalize_segments([SEGS[0], SEGS[0]])
