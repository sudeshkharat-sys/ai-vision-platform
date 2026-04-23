"""Manages an embedded portable Redis instance for Windows EXE deployment."""

import subprocess
import sys
import time
from pathlib import Path


class RedisManager:
    def __init__(self, base_dir: Path, port: int = 6379):
        self.base_dir = base_dir
        self.port = port
        self.aof_dir = base_dir / "data" / "redis"
        self.log_file = base_dir / "logs" / "redis.log"
        self.process: subprocess.Popen | None = None
        self._bin = self._locate_redis_bin()

    def _locate_redis_bin(self) -> Path:
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS) / "redis"
        return Path(__file__).parent.parent / "resources" / "redis"

    def _redis_server(self) -> str:
        exe = self._bin / "redis-server.exe"
        if not exe.exists():
            raise FileNotFoundError(
                f"Redis binary not found: {exe}\n"
                "Place redis-server.exe in deploy/exe/resources/redis/"
            )
        return str(exe)

    def _redis_cli(self) -> str:
        return str(self._bin / "redis-cli.exe")

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.aof_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[redis] Starting on port {self.port}...")

        self.process = subprocess.Popen(
            [
                self._redis_server(),
                "--port", str(self.port),
                "--appendonly", "yes",
                "--dir", str(self.aof_dir),
                "--logfile", str(self.log_file),
                "--loglevel", "notice",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait until Redis responds to PING
        for _ in range(30):
            time.sleep(0.5)
            try:
                result = subprocess.run(
                    [self._redis_cli(), "-p", str(self.port), "ping"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.stdout.strip() == "PONG":
                    print("[redis] Started.")
                    return
            except Exception:
                pass

        raise RuntimeError("Redis did not respond to PING within 15 seconds.")

    def stop(self) -> None:
        print("[redis] Stopping...")
        try:
            subprocess.run(
                [self._redis_cli(), "-p", str(self.port), "shutdown", "nosave"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        print("[redis] Stopped.")
