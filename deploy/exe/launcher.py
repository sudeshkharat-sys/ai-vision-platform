"""
AI Vision Platform — Windows EXE Launcher
==========================================
Starts all required services in order:
  1. Verify CUDA / GPU
  2. Embedded PostgreSQL
  3. Embedded Redis
  4. FastAPI backend  (serves React frontend in EXE mode)
  5. Celery worker

Run directly:   python launcher.py
Compiled EXE:   aivision.exe  (PyInstaller --onedir build)

Configuration is read from  aivision.cfg  next to the launcher (created on
first run with sensible defaults).  Edit the file to change ports or
credentials.
"""

from __future__ import annotations

import configparser
import os
import signal
import socket
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve base directory
# When compiled by PyInstaller the exe lives inside the dist/ directory.
# We keep all runtime data (DB, uploads, logs) in a sibling "data" tree so
# that reinstalling the app does not wipe user data.
# ---------------------------------------------------------------------------
if hasattr(sys, "_MEIPASS"):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Add local services package to path when running as plain Python script
sys.path.insert(0, str(Path(__file__).parent))

from services.postgres import PostgresManager
from services.redis_mgr import RedisManager
from services.celery_worker import CeleryWorker
from services.weights_downloader import check_weights, summary, print_download_progress
import services.backend_svc as backend_svc
import cuda_check

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG_FILE = BASE_DIR / "aivision.cfg"

DEFAULTS = {
    "postgres_port": "5432",
    "redis_port": "6379",
    "api_port": "8000",
    "db_name": "ai_vision",
    "db_user": "aivision",
    "db_password": "aivision_local_pass",
    "open_browser": "true",
    "skip_cuda_check": "false",
}


def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["aivision"] = DEFAULTS
    if CFG_FILE.exists():
        cfg.read(CFG_FILE)
    else:
        with open(CFG_FILE, "w") as f:
            cfg.write(f)
        print(f"[config] Created default config at {CFG_FILE}")
    return cfg


# ---------------------------------------------------------------------------
# Port check
# ---------------------------------------------------------------------------
def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def assert_port_free(port: int, service: str) -> None:
    if port_in_use(port):
        print(f"\n[ERROR] Port {port} is already in use — cannot start {service}.")
        print(f"        Close whatever is using port {port} and try again.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Wait for backend health
# ---------------------------------------------------------------------------
def wait_for_backend(port: int, timeout: int = 120) -> bool:
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  AI Vision Platform — Starting up")
    print("=" * 60)

    cfg = load_config()
    c = cfg["aivision"]

    pg_port = int(c["postgres_port"])
    redis_port = int(c["redis_port"])
    api_port = int(c["api_port"])
    db_name = c["db_name"]
    db_user = c["db_user"]
    db_password = c["db_password"]
    open_browser = c.getboolean("open_browser", fallback=True)
    skip_cuda = c.getboolean("skip_cuda_check", fallback=False)

    # ------------------------------------------------------------------
    # Step 1 — CUDA verification
    # ------------------------------------------------------------------
    print("\n[Step 1/6] Verifying CUDA / GPU...")
    if skip_cuda:
        print("[cuda] Skipping CUDA check (skip_cuda_check=true in config).")
        cuda_status = None
    else:
        cuda_status = cuda_check.verify()
        cuda_status.print_report()
        if not cuda_status.gpu_ready:
            print("\n[WARNING] GPU acceleration is not available.")
            print("          The application will start in CPU-only mode.")
            print("          See CUDA_INSTALL_GUIDE.md for setup instructions.")
            print("          Continuing in 5 seconds... (Ctrl+C to abort)\n")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                print("Aborted.")
                sys.exit(0)

    # ------------------------------------------------------------------
    # Step 2 — YOLO base weights
    # ------------------------------------------------------------------
    weights_dir = BASE_DIR / "data" / "yolo_weights"
    print(f"\n[Step 2/6] Checking YOLO base weights  ({weights_dir})")
    n_ok, n_total, total_mb = summary(weights_dir)

    if n_ok == n_total:
        print(f"  All {n_total} weights present  ({total_mb:.0f} MB)  — offline ready.")
    elif n_ok > 0:
        print(f"  {n_ok}/{n_total} weights present ({total_mb:.0f} MB).")
        missing = [m for m, ok in check_weights(weights_dir).items() if not ok]
        print(f"  Missing: {', '.join(missing)}")
        print("  Attempting to download missing weights (requires internet)…")
        failed = print_download_progress(weights_dir)
        if failed:
            print(f"  [WARNING] Could not download: {', '.join(failed)}")
            print("            Selecting those models during training will require internet.")
        else:
            n_ok2, _, mb2 = summary(weights_dir)
            print(f"  Download complete — {n_ok2}/{n_total} weights present ({mb2:.0f} MB).")
    else:
        print("  No pre-downloaded weights found.")
        print("  Attempting to download all weights (this may take several minutes)…")
        failed = print_download_progress(weights_dir)
        n_ok2, _, mb2 = summary(weights_dir)
        if n_ok2 > 0:
            print(f"  Downloaded {n_ok2}/{n_total} weights ({mb2:.0f} MB).")
        if failed:
            print(f"  [WARNING] {len(failed)} weights unavailable — those models need internet at training time.")

    # ------------------------------------------------------------------
    # Step 3 — PostgreSQL
    # ------------------------------------------------------------------
    print("\n[Step 3/6] Starting PostgreSQL...")
    assert_port_free(pg_port, "PostgreSQL")

    pg = PostgresManager(BASE_DIR, db_name, db_user, db_password, pg_port)
    if not pg.is_initialized():
        pg.initialize()
    pg.start()
    pg.create_db()

    # ------------------------------------------------------------------
    # Step 4 — Redis
    # ------------------------------------------------------------------
    print("\n[Step 4/6] Starting Redis...")
    assert_port_free(redis_port, "Redis")

    redis = RedisManager(BASE_DIR, redis_port)
    redis.start()

    # ------------------------------------------------------------------
    # Step 5 — Backend environment + Uvicorn
    # ------------------------------------------------------------------
    print("\n[Step 5/6] Starting FastAPI backend...")
    assert_port_free(api_port, "Backend API")

    backend_svc.configure_env(BASE_DIR, db_user, db_password, db_name, pg_port, redis_port)
    uvicorn_thread = backend_svc.start(host="127.0.0.1", port=api_port)

    print("[backend] Waiting for health check...")
    if not wait_for_backend(api_port):
        print("[ERROR] Backend did not become healthy within 120 s. Check logs/.")
        _shutdown(pg, redis, None)
        sys.exit(1)
    print("[backend] Ready.")

    # ------------------------------------------------------------------
    # Step 6 — Celery worker
    # ------------------------------------------------------------------
    print("\n[Step 6/6] Starting Celery worker...")
    celery = CeleryWorker(BASE_DIR / "logs")
    celery.start()

    # ------------------------------------------------------------------
    # All services up
    # ------------------------------------------------------------------
    app_url = f"http://127.0.0.1:{api_port}"
    print("\n" + "=" * 60)
    print("  AI Vision Platform is RUNNING")
    print(f"  Open your browser at: {app_url}")
    print("  Press Ctrl+C to stop all services and exit.")
    print("=" * 60 + "\n")

    if open_browser:
        webbrowser.open(app_url)

    # ------------------------------------------------------------------
    # Graceful shutdown on Ctrl+C / SIGTERM
    # ------------------------------------------------------------------
    def _on_signal(sig, frame):
        print("\n[launcher] Shutdown signal received.")
        _shutdown(pg, redis, celery)
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Keep the main thread alive
    while True:
        time.sleep(1)
        if not uvicorn_thread.is_alive():
            print("[ERROR] Uvicorn thread exited unexpectedly. Shutting down.")
            _shutdown(pg, redis, celery)
            sys.exit(1)


def _shutdown(pg: PostgresManager, redis: RedisManager, celery: CeleryWorker | None) -> None:
    print("[launcher] Stopping services...")
    if celery:
        try:
            celery.stop()
        except Exception as e:
            print(f"[celery] Stop error: {e}")
    try:
        redis.stop()
    except Exception as e:
        print(f"[redis] Stop error: {e}")
    try:
        pg.stop()
    except Exception as e:
        print(f"[postgres] Stop error: {e}")
    print("[launcher] All services stopped. Goodbye.")


if __name__ == "__main__":
    # --- Celery Worker Entry Point ---
    # When running as a compiled EXE, the Celery worker is started by re-executing
    # the EXE with '-m celery' arguments. We must intercept this to avoid
    # starting the full launcher (DB, Redis, etc.) again in the worker process.
    if len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2] == "celery":
        try:
            import celery.__main__
        except ImportError:
            print("[ERROR] Celery not found in bundle.")
            sys.exit(1)

        # Ensure the backend directory is in sys.path so the Celery app can be imported
        if hasattr(sys, "_MEIPASS"):
            backend_path = Path(sys._MEIPASS) / "backend"
        else:
            # Fallback for local development runs
            backend_path = Path(__file__).resolve().parent.parent.parent / "backend"
        
        if backend_path.exists() and str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))
            
        # The celery main entry point typically reads from sys.argv.
        # We shift sys.argv so that it looks like ['celery', 'worker', ...]
        sys.argv = sys.argv[2:]
        sys.exit(celery.__main__.main())
    # ---------------------------------

    main()
