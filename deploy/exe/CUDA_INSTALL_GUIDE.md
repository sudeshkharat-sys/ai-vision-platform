# NVIDIA Driver & CUDA Toolkit — Windows Installation Guide

This guide is for users running **AI Vision Platform** on Windows who want
GPU-accelerated inference and training.

> **Skip this guide** if you do not have an NVIDIA GPU — the application runs
> fine in CPU-only mode (slower inference, no CUDA errors).

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU | GeForce GTX 10xx / Quadro P-series | RTX 30xx / 40xx, T4 |
| NVIDIA Driver | 525.xx | Latest (≥ 560.xx) |
| CUDA Toolkit | 12.x | 12.8 |
| Windows | 10 (64-bit) | 11 (64-bit) |
| RAM | 8 GB | 16 GB+ |
| VRAM | 4 GB | 8 GB+ |

---

## Step 1 — Install NVIDIA Display Driver

1. Open **Device Manager** → **Display Adapters** to confirm your GPU model.

2. Go to the NVIDIA driver download page:
   ```
   https://www.nvidia.com/Download/index.aspx
   ```

3. Select:
   - **Product Type**: GeForce / Quadro / Tesla (match your GPU)
   - **Product Series**: (match your GPU series)
   - **Product**: (match your exact GPU)
   - **Operating System**: Windows 10 64-bit or Windows 11
   - **Download Type**: Game Ready Driver (GeForce) or Studio Driver

4. Download and run the installer. Choose **Express Installation**.

5. Reboot your PC after installation.

6. Verify in a Command Prompt:
   ```cmd
   nvidia-smi
   ```
   You should see your GPU listed with a driver version ≥ 525.

---

## Step 2 — Install CUDA Toolkit 12.8

> The bundled PyTorch was built against **CUDA 12.8**.
> You only need the CUDA Toolkit if you plan to compile custom CUDA extensions.
> PyTorch includes its own CUDA runtime — the Toolkit itself is **optional**
> for running inference and training with the bundled app.

If you need the full toolkit (for development):

1. Go to:
   ```
   https://developer.nvidia.com/cuda-12-8-0-download-archive
   ```

2. Select:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **11** or **10**
   - Installer Type: **exe (local)**

3. Download and run the installer.

4. In **Installation Options**, select:
   - [x] CUDA Toolkit
   - [ ] Driver components (uncheck — you installed the driver in Step 1)

5. After installation, add CUDA to your PATH if not done automatically:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
   ```

6. Verify:
   ```cmd
   nvcc --version
   ```

---

## Step 3 — Verify with the Launcher

When you start `aivision.exe`, the launcher automatically runs a GPU check
and prints a report like this:

```
============================================================
  CUDA / GPU Verification Report
============================================================
  [OK] NVIDIA Driver      : 560.94
  [OK] GPU                : NVIDIA GeForce RTX 3080
  [OK] PyTorch            : 2.11.0+cu128
  [OK] PyTorch CUDA       : 12.8

  RESULT: GPU acceleration is ENABLED.
============================================================
```

If you see `CPU-only mode`, re-read Steps 1–2 above.

You can also run the check manually at any time:
```cmd
cd dist\AIVision
python cuda_check.py
```

---

## Troubleshooting

### "NVIDIA driver not detected"
- Ensure your GPU is an NVIDIA product (not AMD/Intel).
- Reinstall the driver from Step 1.
- Check Device Manager for driver errors.

### "PyTorch CUDA NOT available" (driver IS found)
- Your driver may be too old. Driver < 525 does not support CUDA 12.x.
- Update the driver to the latest version.
- Reboot after updating.

### "nvidia-smi is not recognized"
- The driver did not add itself to PATH. Add manually:
  ```
  C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_*\
  ```
  Or simply reboot — most driver installers add this automatically.

### Application crashes with CUDA errors at runtime
- Your GPU may have insufficient VRAM for the loaded model.
- Try setting `MODEL_DIR` in `aivision.cfg` to a folder with smaller models.
- Force CPU mode by adding `skip_cuda_check = true` in `aivision.cfg`.

---

## Driver Version → CUDA Compatibility

| Driver Version | Max CUDA Version |
|---|---|
| ≥ 560.xx | CUDA 12.8 |
| ≥ 525.xx | CUDA 12.x |
| ≥ 520.xx | CUDA 11.8 |
| < 520.xx | Not supported (update required) |
