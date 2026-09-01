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

# ---------------------------------------------------------------------------
# OpenSSL DLL fix — some bundled packages (e.g. cryptography, pulled in via
# celery/redis) ship their own copy of libssl-3.dll / libcrypto-3.dll. When
# PyInstaller's dependency collection picks up both copies, whichever one
# lands in the TOC first wins the shared filename, and if it's the wrong
# (mismatched) build the interpreter's own _ssl.pyd fails at runtime with
# "DLL load failed while importing _ssl: The specified procedure could not
# be found." Force the *interpreter's own* OpenSSL DLLs to the front of the
# binaries list so they always win the name collision, regardless of what
# any other collected package also ships under the same filename.
# ---------------------------------------------------------------------------
_conda_prefix = Path(sys.base_prefix)
_ssl_dll_dirs = [_conda_prefix / "DLLs", _conda_prefix / "Library" / "bin", _conda_prefix]
_ssl_dll_names = [
    "libssl-3.dll", "libcrypto-3.dll",
    "libssl-3-x64.dll", "libcrypto-3-x64.dll",
    "libssl-1_1.dll", "libcrypto-1_1.dll",
    "libssl-1_1-x64.dll", "libcrypto-1_1-x64.dll",
]
_seen_ssl_dlls = set()
for _dll_name in _ssl_dll_names:
    if _dll_name in _seen_ssl_dlls:
        continue
    for _dll_dir in _ssl_dll_dirs:
        _dll_path = _dll_dir / _dll_name
        if _dll_path.exists():
            binaries.insert(0, (str(_dll_path), "."))
            _seen_ssl_dlls.add(_dll_name)
            break
if not _seen_ssl_dlls:
    print("WARNING: Could not locate the interpreter's own OpenSSL DLLs "
          "(libssl/libcrypto) under the Python prefix — the built EXE may "
          "hit 'DLL load failed while importing _ssl' at runtime.")

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
# created fresh at runtime and must never be bundled into the EXE)
for item in BACKEND.iterdir():
    if item.name in ("data", "__pycache__", ".env", ".env.example"):
        continue
    if item.is_dir():
        # Recursively add subdirectory, skipping any __pycache__ inside
        for f in item.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                rel_dest = "backend/" + str(f.relative_to(BACKEND).parent).replace("\\", "/")
                datas += [(str(f), rel_dest)]
    elif item.is_file():
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
    excludes=["tkinter", "matplotlib", "notebook", "IPython"],
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
