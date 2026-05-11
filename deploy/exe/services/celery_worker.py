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
            backend_path = str(Path(sys._MEIPASS) / "backend")
            python_exe = self._find_python_for_exe()
        else:
            backend_path = str(Path(__file__).parent.parent.parent / "backend")
            python_exe = sys.executable

        env = {**os.environ, "PYTHONPATH": backend_path}

        log_file_path = None
        if log_dir is not None:
            log_file_path = Path(log_dir) / "celery.log"

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

        # Drain stdout in a background thread so the pipe never blocks.
        # Output goes to a log file if one was configured, otherwise to console.
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

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            print("[celery] Stopping worker...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("[celery] Worker stopped.")

    # ------------------------------------------------------------------
    @staticmethod
    def _find_python_for_exe() -> str:
        """Locate a usable Python interpreter when running inside a PyInstaller bundle.

        PyInstaller's sys.executable is the compiled EXE — not a Python interpreter —
        so we must find a real python.exe to spawn the Celery worker subprocess.
        Search order:
          1. python.exe bundled inside _MEIPASS (explicit bundle strategy)
          2. python.exe / python3.exe next to the running EXE
          3. 'python' on PATH
        """
        meipass = Path(sys._MEIPASS)

        candidates = [
            meipass / "python.exe",
            meipass / "python3.exe",
            Path(sys.executable).parent / "python.exe",
            Path(sys.executable).parent / "python3.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        # Fall back to whatever 'python' resolves to on PATH.
        return "python"

    @staticmethod
    def _drain_output(process: subprocess.Popen, log_file_path: Path | None) -> None:
        """Read process stdout line-by-line until EOF, writing to log file or stdout."""
        if log_file_path:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8", errors="replace") as f:
                for line in process.stdout:
                    f.write(line.decode("utf-8", errors="replace"))
                    f.flush()
        else:
            for line in process.stdout:
                print("[celery]", line.decode("utf-8", errors="replace"), end="")
