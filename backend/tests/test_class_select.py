import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "app/services/class_select.py"
_spec = importlib.util.spec_from_file_location("class_select", _p)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

ENGINE = [0.5, 0.5, 0.6, 0.6]


def ann(name, bbox):
    return {"class_name": name, "bbox": bbox}


def test_label_by_inner_box_and_empty_is_no_cover():
    full = ann("full_cover", [0.5, 0.5, 0.4, 0.4])
    s, st = cs.derive_region_labels([ann("engine", ENGINE), full], "engine", ["full_cover", "cut_cover"])
    assert [x["state"] for x in s] == ["full_cover"]
    s, st = cs.derive_region_labels([ann("engine", ENGINE)], "engine", ["full_cover", "cut_cover"])
    assert [x["state"] for x in s] == ["no_cover"] and st["empty"] == 1


def test_empty_label_none_skips():
    s, st = cs.derive_region_labels([ann("engine", ENGINE)], "engine", ["full_cover"], empty_label=None)
    assert s == [] and st["empty"] == 1


def test_conflict_skipped_and_outside_box_ignored():
    anns = [ann("engine", ENGINE), ann("full_cover", [0.5, 0.5, 0.2, 0.2]),
            ann("cut_cover", [0.5, 0.45, 0.2, 0.2])]
    s, st = cs.derive_region_labels(anns, "engine", ["full_cover", "cut_cover"])
    assert s == [] and st["conflict"] == 1
    # cover box far away from the engine does not label it
    s, _ = cs.derive_region_labels(
        [ann("engine", ENGINE), ann("cut_cover", [0.95, 0.95, 0.05, 0.05])],
        "engine", ["cut_cover"])
    assert [x["state"] for x in s] == ["no_cover"]


def test_two_engines_each_get_own_label():
    e1, e2 = [0.25, 0.5, 0.4, 0.4], [0.75, 0.5, 0.4, 0.4]
    anns = [ann("engine", e1), ann("engine", e2), ann("cut_cover", [0.75, 0.5, 0.2, 0.2])]
    s, _ = cs.derive_region_labels(anns, "engine", ["cut_cover"])
    assert [x["state"] for x in s] == ["no_cover", "cut_cover"]


def test_select_train_classes_filters_and_drops_images():
    anns = {"a": [ann("engine", ENGINE), ann("cut_cover", ENGINE)], "b": [ann("cut_cover", ENGINE)]}
    imgs = [{"id": "a"}, {"id": "b"}]
    classes, new_anns, new_imgs = cs.select_train_classes(
        ["engine", "cut_cover"], anns, imgs, ["engine"])
    assert classes == ["engine"] and list(new_anns) == ["a"] and new_imgs == [{"id": "a"}]
    assert [a["class_name"] for a in new_anns["a"]] == ["engine"]
    assert anns["a"][1]["class_name"] == "cut_cover"  # input untouched
    assert cs.select_train_classes(["engine"], anns, imgs, None)[0] == ["engine"]


def test_dataset_summary():
    anns = {"a": [ann("engine", ENGINE)], "b": [ann("cut_cover", ENGINE)]}
    out, summ = cs.derive_dataset_labels(anns, "engine", ["cut_cover"])
    assert list(out) == ["a"] and summ["per_class"] == {"no_cover": 1} and summ["images_without_crop"] == 1


def test_folder_label_helpers():
    import ast, pathlib, re
    src = (pathlib.Path(__file__).resolve().parents[1] / "app/api/crop_cls.py").read_text()
    tree = ast.parse(src)
    ns = {"re": re, "Optional": __import__("typing").Optional}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("_clean_label", "label_from_path"):
            exec(compile(ast.Module([n], []), "x", "exec"), ns)
    f = ns["label_from_path"]
    assert f("cut_cover/a.jpg") == "cut_cover"
    assert f("dataset/full cover/sub/a.jpg") == "sub"
    assert f("data\\no_cover\\a.jpg") == "no_cover"
    assert f("a.jpg") is None and f("__MACOSX/a.jpg") is None
    assert ns["_clean_label"]("  full   cover ") == "full cover"


def test_bbox_from_points_fallback():
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "app/tasks/crop_cls_training.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_bbox_from_points")
    ns = {}
    exec(compile(ast.Module([fn], []), "x", "exec"), ns)
    f = ns["_bbox_from_points"]
    assert f([[0.2, 0.2], [0.6, 0.2], [0.6, 0.8]]) == [0.4, 0.5, 0.4, 0.6000000000000001] or \
        [round(v, 3) for v in f([[0.2, 0.2], [0.6, 0.2], [0.6, 0.8]])] == [0.4, 0.5, 0.4, 0.6]
    assert f([[0.1, 0.1]]) is None and f(None) is None


def test_task_log_carries_stage_logs_and_epoch_lines():
    import ast, pathlib, time
    src = (pathlib.Path(__file__).resolve().parents[1] / "app/tasks/crop_cls_training.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "_TaskLog")
    ns = {"time": time}
    exec(compile(ast.Module([cls], []), "x", "exec"), ns)

    class FakeTask:
        request = type("R", (), {"id": "t1"})()
        states = []
        def update_state(self, state, meta):
            self.states.append(meta)

    ft = FakeTask()
    tl = ns["_TaskLog"](ft)
    assert tl.request.id == "t1"
    tl.log("hello")
    tl.set("dataset", epoch=0, total_epochs=3, history=[])
    # what the shared epoch callback sends: no stage, no logs
    tl.update_state(state="STARTED", meta={"epoch": 1, "total_epochs": 3, "eta_seconds": 5,
                                           "history": [{"epoch": 1, "loss": 0.5, "accuracy_top1": 0.8}]})
    last = ft.states[-1]
    assert last["stage"] == "dataset" and last["epoch"] == 1
    assert any("hello" in l for l in last["logs"]) and any("epoch 1/3" in l and "0.800" in l for l in last["logs"])
