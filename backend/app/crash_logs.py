"""Crash diagnostics (works for local run AND the EXE).

Training sometimes kills the backend and the worker with no error message.
This file writes logs that survive the crash, so we can see the cause:

  logs/console_<name>.log - everything printed (console text is lost on crash)
  logs/crash_<name>.log  - Python tracebacks AND native crashes (CUDA / segfault)
  logs/resources.log   - every 30 s: free RAM, free disk, GPU memory, which
                         services are still alive and their exit codes

How to read them after a crash:
  * RAM "free" going near 0 in resources.log      -> out of memory
  * GPU "used" near "total"                        -> GPU out of memory
  * disk free near 0                               -> disk full
  * exit code 3221225477 (0xC0000005)              -> native crash (access violation)
  * exit code 3221226505 (0xC0000409)              -> native crash (often CUDA / driver)
  * exit code 3221225725 (0xC00000FD)              -> stack overflow
  * "Fatal Python error" in crash_*.log            -> native crash, stack shown there
"""

import ctypes
import datetime
import faulthandler
import os
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path

_open_files = []  # keep crash log files open for the whole run


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_dir: Path, file_name: str, text: str) -> None:
    """Append one line with a timestamp. Never raises."""
    try:
        with open(log_dir / file_name, "a", encoding="utf-8") as f:
            f.write(f"[{_now()}] {text}\n")
    except Exception:
        pass


class _Tee:
    """Write to the console and to a log file at the same time."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        try:
            if self.console:
                self.console.write(text)
        except Exception:
            pass
        try:
            self.log_file.write(text)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            if self.console:
                self.console.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def copy_console_to_file(log_dir: Path, file_name: str) -> None:
    """Send print() output to a log file too (the launcher console closes on crash)."""
    log_file = open(log_dir / file_name, "a", encoding="utf-8", buffering=1)
    log_file.write(f"\n===== started {_now()} pid={os.getpid()} =====\n")
    _open_files.append(log_file)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def copy_logging_to_file(log_dir: Path, file_name: str) -> None:
    """Also write uvicorn / celery log lines (INFO, errors, "Shutting down",
    "Reloading") to the console log file. They use the logging module, not
    print(), so copy_console_to_file alone does not catch them."""
    import logging
    handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)        # root: celery and most libraries
    uvicorn_logger = logging.getLogger("uvicorn")
    if not uvicorn_logger.propagate:               # uvicorn does not pass its lines to root
        uvicorn_logger.addHandler(handler)


def enable_crash_logging(log_dir: Path, name: str) -> None:
    """Log native crashes and uncaught Python errors to logs/crash_<name>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_file = open(log_dir / f"crash_{name}.log", "a", encoding="utf-8")
    crash_file.write(f"\n===== {name} started {_now()} pid={os.getpid()} =====\n")
    crash_file.flush()
    _open_files.append(crash_file)

    # Native crashes (segfault, access violation, CUDA abort) -> stack of all threads
    faulthandler.enable(file=crash_file, all_threads=True)

    def write_error(title, exc_type, exc_value, exc_tb):
        try:
            crash_file.write(f"[{_now()}] {title}\n")
            crash_file.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            crash_file.flush()
        except Exception:
            pass

    def on_main_error(exc_type, exc_value, exc_tb):
        write_error("Uncaught error in main thread", exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def on_thread_error(args):
        write_error(f"Uncaught error in thread '{args.thread.name}'",
                    args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = on_main_error
    threading.excepthook = on_thread_error


# ---------------------------------------------------------------------------
# Resource readings
# ---------------------------------------------------------------------------

def ram_text() -> str:
    """Free / total RAM. Uses Windows API, falls back to /proc/meminfo on Linux."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            gb = 1024 ** 3
            return (f"RAM free {m.ullAvailPhys / gb:.1f}/{m.ullTotalPhys / gb:.1f} GB "
                    f"({m.dwMemoryLoad}% used), pagefile free "
                    f"{m.ullAvailPageFile / gb:.1f}/{m.ullTotalPageFile / gb:.1f} GB")
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) / 1024 / 1024
        return f"RAM free {info['MemAvailable']:.1f}/{info['MemTotal']:.1f} GB"
    except Exception as e:
        return f"RAM ? ({e})"


def disk_text(path: Path) -> str:
    try:
        d = shutil.disk_usage(path)
        return f"disk free {d.free / 1024 ** 3:.1f}/{d.total / 1024 ** 3:.1f} GB"
    except Exception as e:
        return f"disk ? ({e})"


def gpu_text() -> str:
    """GPU memory / temperature from nvidia-smi (works without torch)."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout.strip()
        if not out:
            return "GPU ? (nvidia-smi gave no output)"
        used, total, temp, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return f"GPU used {used}/{total} MB, {temp}C, {util}% busy"
    except FileNotFoundError:
        return "GPU ? (nvidia-smi not found)"
    except Exception as e:
        return f"GPU ? ({e})"


def process_text(name: str, proc) -> str:
    """'alive' or 'DEAD code=...' for a subprocess.Popen."""
    if proc is None:
        return f"{name}=not started"
    code = proc.poll()
    if code is None:
        return f"{name}=alive"
    return f"{name}=DEAD code={code} (hex {code & 0xFFFFFFFF:#010x})"


def start_resource_monitor(log_dir: Path, data_dir: Path, get_processes=None,
                           every_seconds: int = 30) -> None:
    """Background thread: write RAM/disk/GPU/service status to logs/resources.log.

    get_processes: function returning {"name": Popen-or-None}; when a service
    dies, its exit code is written to resources.log and launcher.log at once.
    """
    reported_dead = set()

    def loop():
        import time
        while True:
            try:
                parts = [f"pid={os.getpid()}", ram_text(), disk_text(data_dir), gpu_text()]
                if get_processes:
                    for name, proc in get_processes().items():
                        parts.append(process_text(name, proc))
                        if proc is not None and proc.poll() is not None and name not in reported_dead:
                            reported_dead.add(name)
                            msg = (f"[monitor] {name} has STOPPED. "
                                   f"{process_text(name, proc)}. {ram_text()}. {gpu_text()}")
                            print(msg)
                            log_line(log_dir, "resources.log", msg)
                log_line(log_dir, "resources.log", " | ".join(parts))
            except Exception as e:
                log_line(log_dir, "resources.log", f"monitor error: {e}")
            time.sleep(every_seconds)

    threading.Thread(target=loop, daemon=True, name="resource-monitor").start()


_done = False


def get_log_dir() -> Path:
    """Logs go next to the data folder: <data>/../logs (e.g. backend/logs locally)."""
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "./data/uploads")).resolve()
    log_dir = upload_dir.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup(name: str) -> None:
    """Call once per process: name is "backend" or "worker"."""
    global _done
    if _done:
        return
    _done = True
    try:
        log_dir = get_log_dir()
        copy_console_to_file(log_dir, f"console_{name}.log")
        enable_crash_logging(log_dir, name)
        copy_logging_to_file(log_dir, f"console_{name}.log")
        start_resource_monitor(log_dir, log_dir.parent)
        log_line(log_dir, "resources.log", f"{name} process started pid={os.getpid()}")
        print(f"[crash_logs] {name}: logs in {log_dir}")
    except Exception as e:
        print(f"[crash_logs] could not set up logging: {e}")


def log_training(text: str) -> None:
    """One line in logs/training.log with RAM + GPU, used at each training step."""
    try:
        extra = ram_text()
        try:
            import torch
            if torch.cuda.is_available():
                gb = 1024 ** 3
                extra += (f", torch GPU allocated {torch.cuda.memory_allocated() / gb:.2f} GB"
                          f", reserved {torch.cuda.memory_reserved() / gb:.2f} GB")
        except Exception:
            pass
        log_line(get_log_dir(), "training.log", f"pid={os.getpid()} {text} | {extra}")
    except Exception:
        pass
