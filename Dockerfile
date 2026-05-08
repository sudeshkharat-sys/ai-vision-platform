# ── Stage 1: Build React Frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps

COPY frontend/ ./

# Build-time API URL — default is relative (/api/v1) so Nginx proxies correctly
# on any domain without rebuilding the image.
ARG REACT_APP_API_URL=/api/v1
ARG REACT_APP_BASE_URL=""
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_BASE_URL=$REACT_APP_BASE_URL

RUN npm run build

# ── Stage 2: PyTorch official image — Python 3.11 + CUDA 12.8 + pip included ──
# pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime ships:
#   • Python 3.11  • pip  • PyTorch 2.11  • CUDA 12.8  • cuDNN 9
# T4 GPU (Compute Capability 7.5) is fully supported by CUDA 12.8.
# No manual Python/pip installation needed.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_WEIGHTS_DIR=/app/data/yolo_weights

# Install system libs for OpenCV / psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        postgresql-client \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# --break-system-packages is safe in Docker (isolated container).
# Required for Python 3.12+ which enforces PEP 668 externally-managed-environment.
# PyTorch is already installed in the base image — pip will skip reinstalling it.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY backend/ ./

# ── Copy built React app from Stage 1 ─────────────────────────────────────────
# Docker will initialise the named volume frontend_build from this directory
# on first mount (when the volume is empty), making the files available to Nginx.
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /app/data/uploads /app/data/models /app/data/yolo_weights /app/logs

# ── Pre-download YOLO base weights for offline deployment ─────────────────────
# All weights land in /app/data/yolo_weights inside this image layer.
# This directory is NOT volume-mounted in docker-compose so the weights are
# always available even when the system has no internet connection.
# Failures for unreleased model families are non-fatal (script exits 0).
COPY backend/scripts/download_yolo_weights.py /tmp/download_yolo_weights.py
RUN python /tmp/download_yolo_weights.py && rm /tmp/download_yolo_weights.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
