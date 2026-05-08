"""
CUDA availability verification for Windows EXE deployment.

Checks (in order):
  1. nvidia-smi  — confirms a driver is installed and a GPU is visible
  2. torch.cuda  — confirms PyTorch can see the GPU
  3. CUDA version compatibility with bundled PyTorch

Returns a CudaStatus dataclass so the launcher can decide whether to
warn, abort, or continue in CPU-only mode.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class CudaStatus:
    driver_found: bool = False
    driver_version: str = ""
    gpu_name: str = ""
    torch_cuda_available: bool = False
    torch_cuda_version: str = ""
    torch_version: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def gpu_ready(self) -> bool:
        return self.driver_found and self.torch_cuda_available

    def print_report(self) -> None:
        print("=" * 60)
        print("  CUDA / GPU Verification Report")
        print("=" * 60)

        # Driver
        if self.driver_found:
            print(f"  [OK] NVIDIA Driver      : {self.driver_version}")
            print(f"  [OK] GPU                : {self.gpu_name}")
        else:
            print("  [!!] NVIDIA Driver      : NOT FOUND")

        # PyTorch
        if self.torch_version:
            print(f"  [OK] PyTorch            : {self.torch_version}")
        if self.torch_cuda_available:
            print(f"  [OK] PyTorch CUDA       : {self.torch_cuda_version}")
        else:
            print("  [!!] PyTorch CUDA       : NOT available (CPU mode)")

        if self.warnings:
            print()
            for w in self.warnings:
                print(f"  [WW] {w}")

        if self.errors:
            print()
            for e in self.errors:
                print(f"  [EE] {e}")

        print()
        if self.gpu_ready:
            print("  RESULT: GPU acceleration is ENABLED.")
        else:
            print("  RESULT: Running in CPU-only mode (GPU acceleration disabled).")
            print("          Inference and training will be significantly slower.")
        print("=" * 60)


# ---------------------------------------------------------------------------

def _check_nvidia_smi() -> tuple[bool, str, str]:
    """Returns (found, driver_version, gpu_name)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=driver_version,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            driver = parts[0] if len(parts) > 0 else "unknown"
            name = parts[1] if len(parts) > 1 else "unknown"
            return True, driver, name
    except FileNotFoundError:
        pass  # nvidia-smi not on PATH — driver not installed
    except Exception as exc:
        pass
    return False, "", ""


def _check_torch_cuda() -> tuple[bool, str, str]:
    """Returns (available, cuda_version, torch_version)."""
    try:
        import torch
        available = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "n/a"
        torch_ver = torch.__version__
        return available, cuda_ver, torch_ver
    except ImportError:
        return False, "", ""


# ---------------------------------------------------------------------------

def verify() -> CudaStatus:
    status = CudaStatus()

    # 1. NVIDIA driver
    status.driver_found, status.driver_version, status.gpu_name = _check_nvidia_smi()

    if not status.driver_found:
        status.errors.append(
            "NVIDIA driver not detected. Install the driver from "
            "https://www.nvidia.com/Download/index.aspx"
        )

    # 2. PyTorch CUDA
    status.torch_cuda_available, status.torch_cuda_version, status.torch_version = \
        _check_torch_cuda()

    if not status.torch_cuda_available and status.driver_found:
        status.warnings.append(
            "NVIDIA driver found but PyTorch cannot access CUDA. "
            "This usually means the CUDA Toolkit version bundled with PyTorch "
            "does not match the installed driver. "
            "Update your driver to >= 525 for CUDA 12.x support."
        )

    if not status.torch_version:
        status.errors.append("PyTorch is not importable. The installation may be corrupted.")

    # 3. Driver version sanity check for CUDA 12.8 (requires driver >= 525)
    if status.driver_found:
        try:
            major = int(status.driver_version.split(".")[0])
            if major < 525:
                status.warnings.append(
                    f"Driver version {status.driver_version} is too old for CUDA 12.8 "
                    "(requires >= 525.xx). Update from https://www.nvidia.com/Download/index.aspx"
                )
        except (ValueError, IndexError):
            pass

    return status


# ---------------------------------------------------------------------------
# Allow running as a standalone check: python cuda_check.py

if __name__ == "__main__":
    s = verify()
    s.print_report()
    sys.exit(0 if s.gpu_ready else 1)
