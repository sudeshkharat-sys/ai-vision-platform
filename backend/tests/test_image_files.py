import importlib.util, pathlib, sys, types

root = pathlib.Path(__file__).resolve().parents[1] / "app"
# minimal stand-ins so the module loads without the app's dependencies
cfg = types.ModuleType("app.config"); pkg = types.ModuleType("app"); svc = types.ModuleType("app.services")
cfg.settings = types.SimpleNamespace(upload_dir=pathlib.Path("."))
for n, m in (("app", pkg), ("app.config", cfg), ("app.services", svc)):
    sys.modules[n] = m
pkg.__path__ = [str(root)]; svc.__path__ = [str(root / "services")]
spec = importlib.util.spec_from_file_location("app.services.image_files", root / "services/image_files.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)


def test_resolves_upload_and_video_frame(tmp_path):
    cfg.settings.upload_dir = tmp_path / "uploads"
    (tmp_path / "uploads/P/video_frames/V").mkdir(parents=True)
    (tmp_path / "uploads/P/a.jpg").write_bytes(b"x")
    (tmp_path / "uploads/P/video_frames/V/f1.jpg").write_bytes(b"x")
    assert mod.resolve_image_file("P", "/uploads/P/a.jpg").name == "a.jpg"
    assert mod.resolve_image_file("P", "/uploads/P/video_frames/V/f1.jpg").parent.name == "V"
    assert mod.resolve_image_file("P", "/uploads/P/video_frames/V/nope.jpg") is None
