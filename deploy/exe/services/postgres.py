"""Manages an embedded portable PostgreSQL instance for Windows EXE deployment."""

import os
import subprocess
import sys
import time
import shutil
import signal
from pathlib import Path


class PostgresManager:
    def __init__(self, base_dir: Path, db_name: str, db_user: str, db_password: str, port: int = 5432):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "pgdata"
        self.log_file = base_dir / "logs" / "postgres.log"
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.port = port
        self.process: subprocess.Popen | None = None

        # Portable PostgreSQL binaries are extracted next to the launcher
        self._bin = self._locate_pg_bin()

    def _locate_pg_bin(self) -> Path:
        # When bundled with PyInstaller, _MEIPASS holds extracted resources
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS) / "postgres" / "bin"
        # Development fallback: look relative to this file
        return Path(__file__).parent.parent / "resources" / "postgres" / "bin"

    def _pg(self, binary: str) -> str:
        exe = self._bin / f"{binary}.exe"
        if not exe.exists():
            raise FileNotFoundError(
                f"PostgreSQL binary not found: {exe}\n"
                "Make sure portable PostgreSQL binaries are placed in deploy/exe/resources/postgres/"
            )
        return str(exe)

    # ------------------------------------------------------------------
    def is_initialized(self) -> bool:
        return (self.data_dir / "PG_VERSION").exists()

    def initialize(self) -> None:
        """Run initdb to create a fresh cluster on first launch."""
        print("[postgres] Initializing database cluster...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        pwfile = self.base_dir / ".pgpass_tmp"
        pwfile.write_text(self.db_password)
        try:
            result = subprocess.run(
                [
                    self._pg("initdb"),
                    "--pgdata", str(self.data_dir),
                    "--username", self.db_user,
                    "--pwfile", str(pwfile),
                    "--encoding", "UTF8",
                    "--auth", "md5",
                ],
                capture_output=True,
                text=True,
            )
        finally:
            pwfile.unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(f"initdb failed:\n{result.stderr}")

        # Allow password authentication over localhost
        hba = self.data_dir / "pg_hba.conf"
        content = hba.read_text()
        # Replace ident/peer with md5 for host connections
        content = content.replace("host    all             all             127.0.0.1/32            ident",
                                  "host    all             all             127.0.0.1/32            md5")
        content = content.replace("host    all             all             ::1/128                 ident",
                                  "host    all             all             ::1/128                 md5")
        hba.write_text(content)

        print("[postgres] Cluster initialized.")

    def start(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[postgres] Starting on port {self.port}...")
        self.process = subprocess.Popen(
            [
                self._pg("pg_ctl"),
                "start",
                "--pgdata", str(self.data_dir),
                "--log", str(self.log_file),
                "--wait",
                "-o", f"-p {self.port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.process.wait()
        if self.process.returncode != 0:
            err = self.process.stderr.read().decode()
            raise RuntimeError(f"pg_ctl start failed:\n{err}")
        print("[postgres] Started.")

    def create_db(self) -> None:
        """Create the application database if it does not exist yet."""
        env = {**os.environ, "PGPASSWORD": self.db_password}
        result = subprocess.run(
            [
                self._pg("psql"),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--username", self.db_user,
                "--dbname", "postgres",
                "--tuples-only",
                "--command", f"SELECT 1 FROM pg_database WHERE datname='{self.db_name}';",
            ],
            capture_output=True, text=True, env=env,
        )
        if "1" not in result.stdout:
            print(f"[postgres] Creating database '{self.db_name}'...")
            subprocess.run(
                [
                    self._pg("createdb"),
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--username", self.db_user,
                    self.db_name,
                ],
                check=True, env=env,
            )
            self._run_init_sql(env)
            print(f"[postgres] Database '{self.db_name}' created.")

    def _run_init_sql(self, env: dict) -> None:
        if hasattr(sys, "_MEIPASS"):
            sql_file = Path(sys._MEIPASS) / "init-db.sql"
        else:
            sql_file = Path(__file__).parent.parent.parent.parent / "scripts" / "init-db.sql"

        if sql_file.exists():
            subprocess.run(
                [
                    self._pg("psql"),
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--username", self.db_user,
                    "--dbname", self.db_name,
                    "--file", str(sql_file),
                ],
                check=True, env=env,
            )

    def stop(self) -> None:
        print("[postgres] Stopping...")
        subprocess.run(
            [
                self._pg("pg_ctl"),
                "stop",
                "--pgdata", str(self.data_dir),
                "--mode", "fast",
                "--wait",
            ],
            capture_output=True,
        )
        print("[postgres] Stopped.")
