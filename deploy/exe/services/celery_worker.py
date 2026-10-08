"""Starts the Celery worker as a subprocess for EXE deployment."""

import os
import subprocess
import sys
import time
from pathlib import Path


class CeleryWorker:
    def __init__(self):
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        print("[celery] Starting worker...")

        if hasattr(sys, "_MEIPASS"):
            backend_path = str(Path(sys._MEIPASS) / "backend")
            python_exe = sys.executable
        else:
            backend_path = str(Path(__file__).parent.parent.parent / "backend")
            python_exe = sys.executable

        env = {**os.environ, "PYTHONPATH": backend_path}

        if getattr(sys, "frozen", False):
            # sys.executable is the launcher exe itself: re-enter it in worker mode.
            cmd = [python_exe, "--celery-worker"]
            cwd = None
        else:
            cmd = [python_exe, "-m", "celery", "-A", "app.tasks.celery_app", "worker",
                   "--loglevel=info", "--pool=solo", "-Q", "celery"]
            cwd = backend_path

        log_dir = Path(os.environ.get("UPLOAD_DIR", ".")).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(log_dir / "celery.log", "ab")

        self.process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        time.sleep(3)
        if self.process.poll() is not None:
            print(f"[celery] ERROR: worker exited immediately (code {self.process.returncode}). See {log_dir / 'celery.log'}")
        print("[celery] Worker started.")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            print("[celery] Stopping worker...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[celery] Worker stopped.")
