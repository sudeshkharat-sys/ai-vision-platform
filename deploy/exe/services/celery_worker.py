"""Starts the Celery worker as a subprocess for EXE deployment."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


class CeleryWorker:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    def start(self, log_dir: Path | None = None) -> None:
        print("[celery] Starting worker...")

        if hasattr(sys, "_MEIPASS"):
            # PyInstaller EXE: re-launch the EXE itself in worker mode.
            # The launcher detects _AIVISION_CELERY_WORKER and hands off to
            # the Celery CLI before any other code runs.
            backend_path = str(Path(sys._MEIPASS) / "backend")
            cmd = [sys.executable]
            extra_env = {"_AIVISION_CELERY_WORKER": "1"}
        else:
            backend_path = str(Path(__file__).parent.parent.parent / "backend")
            cmd = [sys.executable, "-m", "celery",
                   "-A", "app.tasks.celery_app", "worker",
                   "--loglevel=info", "--pool=solo", "-Q", "celery"]
            extra_env = {}

        env = {**os.environ, "PYTHONPATH": backend_path, **extra_env}

        log_file_path = None
        if log_dir is not None:
            log_file_path = Path(log_dir) / "celery.log"

        self.process = subprocess.Popen(
            cmd,
            cwd=backend_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Drain stdout in a background thread so the pipe never blocks.
        self._log_thread = threading.Thread(
            target=self._drain_output,
            args=(self.process, log_file_path),
            daemon=True,
            name="celery-log",
        )
        self._log_thread.start()

        # Give the worker a moment and check it didn't crash immediately.
        time.sleep(2)
        if self.process.poll() is not None:
            hint = (
                f"  Check {log_file_path} for details."
                if log_file_path
                else "  Enable log_dir to capture Celery output."
            )
            raise RuntimeError(
                f"[celery] Worker process exited immediately (code {self.process.returncode}).\n{hint}"
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

    @staticmethod
    def _drain_output(process: subprocess.Popen, log_file_path: Path | None) -> None:
        """Read process stdout until EOF; write to log file or console."""
        if log_file_path:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8", errors="replace") as f:
                for line in process.stdout:
                    f.write(line.decode("utf-8", errors="replace"))
                    f.flush()
        else:
            for line in process.stdout:
                print("[celery]", line.decode("utf-8", errors="replace"), end="")
