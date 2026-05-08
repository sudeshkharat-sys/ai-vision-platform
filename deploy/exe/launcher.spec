# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for AI Vision Platform Windows EXE
#
# Build with:   pyinstaller launcher.spec
# Output:       dist/AIVision/aivision.exe
#
# Prerequisites (run build.bat — it handles all of this):
#   pip install pyinstaller
#   Place portable PostgreSQL 17 in:  deploy/exe/resources/postgres/
#   Place portable Redis 7 in:        deploy/exe/resources/redis/
#   Build React frontend:             cd frontend && npm run build

import sys
import os
import sysconfig
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

REPO_ROOT = Path(SPECPATH).parent.parent        # ai-vision-platform/
BACKEND   = REPO_ROOT / "backend"
FRONTEND  = REPO_ROOT / "frontend" / "build"
RESOURCES = Path(SPECPATH) / "resources"
SCRIPTS   = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Conda stdlib fix — PyInstaller sometimes fails to locate the Python stdlib
# when building inside a Conda environment, causing 'No module named encodings'
# at runtime. Explicitly bundle the entire stdlib into the _internal folder.
# ---------------------------------------------------------------------------
_stdlib_dir = sysconfig.get_path("stdlib")
_platstdlib_dir = sysconfig.get_path("platstdlib")
datas = []
if _stdlib_dir and os.path.isdir(_stdlib_dir):
    datas += [(_stdlib_dir, "lib-stdlib")]
if _platstdlib_dir and os.path.isdir(_platstdlib_dir) and _platstdlib_dir != _stdlib_dir:
    datas += [(_platstdlib_dir, "lib-platstdlib")]
# ---------------------------------------------------------------------------
binaries  = []
hiddenimports = []

# Explicitly collect all encodings submodules (Conda stdlib fix)
hiddenimports += collect_submodules("encodings")

# matplotlib — use collect_submodules + collect_data_files instead of
# collect_all to avoid the '_c_internal_utils circular import' bug where
# PyInstaller double-bundles the package and partially initializes it
from PyInstaller.utils.hooks import collect_data_files
hiddenimports += collect_submodules("matplotlib")
datas += collect_data_files("matplotlib")

for pkg in [
    "uvicorn", "fastapi", "starlette",
    "sqlalchemy", "asyncpg", "psycopg2",
    "celery", "redis", "kombu", "billiard",
    "transformers", "tokenizers", "huggingface_hub",
    "ultralytics",
    "supervision",
    "PIL",
    "cv2",
    "torch", "torchvision",
    "sentencepiece",
    "scipy",
    "sklearn",
]:
    try:
        d, b, h = collect_all(pkg)
        datas    += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Backend source — copy only app code, never data/ (models & uploads are
# created fresh at runtime and must never be bundled into the EXE).
# Also skip any model-weight files (.pt / .pth / .onnx / .bin) and training
# output directories (runs/, weights/, checkpoints/) that might be present in
# the working tree at build time.
_SKIP_DIRS  = {"data", "__pycache__", "runs", "weights", "checkpoints", ".git"}
_SKIP_EXTS  = {
    # model weights
    ".pt", ".pth", ".onnx", ".bin", ".npy", ".npz", ".pkl",
    # uploaded / training images — created at runtime, never bundled
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif",
}

for item in BACKEND.iterdir():
    if item.name in _SKIP_DIRS:
        continue
    if item.is_dir():
        for f in item.rglob("*"):
            if not f.is_file():
                continue
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            if f.suffix.lower() in _SKIP_EXTS:
                continue
            rel_dest = "backend/" + str(f.relative_to(BACKEND).parent).replace("\\", "/")
            datas += [(str(f), rel_dest)]
    elif item.is_file():
        if item.suffix.lower() not in _SKIP_EXTS:
            datas += [(str(item), "backend")]

# React build artefacts
if FRONTEND.exists():
    datas += [(str(FRONTEND), "frontend_build")]
else:
    print(f"WARNING: React build not found at {FRONTEND}. Run 'npm run build' first.")

# Database init SQL
datas += [(str(SCRIPTS / "init-db.sql"), ".")]

# Portable service binaries
pg_bin = RESOURCES / "postgres"
if pg_bin.exists():
    datas += [(str(pg_bin), "postgres")]
else:
    print(f"WARNING: Portable PostgreSQL not found at {pg_bin}")

redis_bin = RESOURCES / "redis"
if redis_bin.exists():
    datas += [(str(redis_bin), "redis")]
else:
    print(f"WARNING: Portable Redis not found at {redis_bin}")

# Services package (launcher helpers)
datas += [(str(Path(SPECPATH) / "services"), "services")]
datas += [(str(Path(SPECPATH) / "cuda_check.py"), ".")]

# Additional hidden imports for async drivers and task modules
hiddenimports += [
    "asyncpg",
    "asyncpg.pgproto.pgproto",
    "psycopg2",
    "celery.app.amqp",
    "celery.backends.redis",
    "celery.loaders.app",
    "kombu.transport.redis",
    "app.tasks.training",
    "app.tasks.auto_annotate",
    "app.tasks.ai_prompt",
    "app.tasks.video_processing",
    "app.tasks.active_learning",
    "app.api.projects",
    "app.api.images",
    "app.api.annotations",
    "app.api.pipeline",
    "app.api.auth",
    "app.api.videos",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "email_validator",
    "multipart",
    # passlib dynamically loads password handlers via its registry
    "passlib.handlers.pbkdf2",
    "passlib.handlers.bcrypt",
    "passlib.handlers.sha2_crypt",
    "passlib.handlers.md5_crypt",
    "passlib.handlers.des_crypt",
    "passlib.handlers.argon2",
    "passlib.handlers.scrypt",
    "passlib.utils.pbkdf2",
    "passlib.utils.handlers",
    "passlib.utils.binary",
    "passlib.utils.decor",
    "passlib.crypto.digest",
    "passlib.crypto.scrypt",
    # matplotlib C extensions — must be explicit or PyInstaller leaves them
    # out causing 'partially initialized module' circular import at runtime
    "matplotlib._c_internal_utils",
    "matplotlib._image",
    "matplotlib._path",
    "matplotlib._qhull",
    "matplotlib._tri",
    "matplotlib._ttconv",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    # supervision optional deps
    "supervision.draw.color",
    "supervision.annotators.core",
    "supervision.annotators.utils",
]

# ---------------------------------------------------------------------------
a = Analysis(
    ["launcher.py"],
    pathex=[str(BACKEND)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "notebook", "IPython"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # --onedir keeps binaries separate (faster startup)
    name="aivision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # UPX can corrupt CUDA DLLs — keep off
    console=True,               # Show console window so users can see status
    icon=str(REPO_ROOT / "frontend" / "public" / "favicon.ico") if (REPO_ROOT / "frontend" / "public" / "favicon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AIVision",
)
