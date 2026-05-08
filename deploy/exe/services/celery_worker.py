"""Starts the Celery worker as a subprocess for EXE deployment."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


class CeleryWorker:
    def __init__(self, log_dir: Path | None = None):
        self.process: subprocess.Popen | None = None
        self._log_dir = log_dir
        self._reader_thread: threading.Thread | None = None

    def start(self) -> None:
        print("[celery] Starting worker...")

        if hasattr(sys, "_MEIPASS"):
            backend_path = str(Path(sys._MEIPASS) / "backend")
            python_exe = sys.executable
        else:
            backend_path = str(Path(__file__).parent.parent.parent / "backend")
            python_exe = sys.executable

        env = {**os.environ, "PYTHONPATH": backend_path}

        log_file = None
        if self._log_dir:
            Path(self._log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(self._log_dir) / "celery.log"
            log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            print(f"[celery] Logging to {log_path}")

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

        # Drain stdout in a background thread — prevents pipe-buffer deadlock and
        # streams every worker log line to the main console + log file.
        self._reader_thread = threading.Thread(
            target=self._stream_output,
            args=(log_file,),
            daemon=True,
            name="celery-log",
        )
        self._reader_thread.start()

        # Give the worker a few seconds to boot, then verify it didn't crash.
        time.sleep(4)
        if self.process.poll() is not None:
            raise RuntimeError(
                f"[celery] Worker exited immediately (exit code {self.process.returncode}). "
                f"Check logs/celery.log for the full error."
            )

        print("[celery] Worker running.")

    def _stream_output(self, log_file) -> None:
        try:
            for raw in self.process.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                print(f"[celery] {line}", flush=True)
                if log_file:
                    log_file.write(line + "\n")
        finally:
            if log_file:
                log_file.close()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            print("[celery] Stopping worker...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[celery] Worker stopped.")
