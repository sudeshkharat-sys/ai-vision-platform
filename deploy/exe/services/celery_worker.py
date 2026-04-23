"""Starts the Celery worker as a subprocess for EXE deployment."""

import os
import subprocess
import sys
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

        self.process = subprocess.Popen(
            [
                python_exe, "-m", "celery",
                "-A", "app.tasks.celery_app", "worker",
                "--loglevel=info",
                "--pool=solo",
                "-Q", "celery",
            ],
            cwd=backend_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
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
