# Project Brief: AI Vision Platform

**Date:** June 4, 2026
**Prepared by:** Sudesh Kharat
**To:** [Manager Name]

---

## Overview

The **AI Vision Platform** is an end-to-end computer vision solution that streamlines dataset management, image annotation, and YOLO model training. The platform introduces a proprietary **"Seed & Scale"** workflow that uses AI-assisted auto-annotation to dramatically reduce manual labeling effort — enabling teams to train production-ready models with as little as 5% manually annotated data.

---

## Problem Statement

Manual image annotation is the primary bottleneck in computer vision projects. It is time-consuming, expensive, and error-prone. Teams often spend weeks labeling data before a single model can be trained. This platform eliminates that bottleneck.

---

## Solution

A full-stack web platform that combines a canvas-based annotation workspace with a multi-stage AI pipeline for zero-shot object detection:

1. **Grounding DINO** — detects objects from natural language text prompts (no pre-training required)
2. **SigLIP** — verifies and filters detections for accuracy
3. **SAM 2** — generates precise pixel-level segmentation masks

The "Seed & Scale" loop:
- Manually annotate ~5% of images
- Train an initial YOLO model on that seed data
- Auto-annotate the remaining 95% using the trained model
- Refine with full-dataset retraining

---

## Key Features

| Feature | Description |
|---|---|
| Multi-model YOLO support | v8, v9, v10, v11 |
| Canvas annotation workspace | Browser-based drawing with Konva.js |
| Real-time training metrics | Live loss/accuracy charts via WebSockets |
| Zero-shot detection | Text-prompt-driven detection without labeled data |
| Video processing | Frame extraction and annotation for video datasets |
| Async background processing | Celery + Redis for non-blocking training jobs |
| Containerized deployment | Docker-based, production-ready infrastructure |

---

## Technology Stack

- **Backend:** FastAPI, SQLAlchemy, Celery, PostgreSQL, Redis
- **Frontend:** React 19, Konva.js, Recharts, Axios
- **ML/AI:** Ultralytics YOLO, Grounding DINO, SigLIP, SAM 2 (Meta)
- **Infrastructure:** Docker, PostgreSQL, Redis

---

## Current Status

The platform is **production-ready** with:
- Complete annotation-to-training pipeline implemented
- Full async job processing for long-running ML tasks
- WebSocket support for real-time UI updates
- Modular API architecture (auth, projects, images, annotations, training, video)

---

## Business Value

- Reduces annotation time by up to **95%** via AI-assisted labeling
- Supports multiple YOLO versions in a single unified platform
- Enables non-ML teams to train vision models through a guided UI workflow
- Accelerates model iteration cycles from weeks to hours

---

## Next Steps / Asks

- [ ] Review and align on deployment timeline
- [ ] Confirm infrastructure budget (PostgreSQL, Redis, GPU for training)
- [ ] Identify pilot use case / dataset for first production run
- [ ] Approval to proceed to staging environment

---

*For questions or a demo, please reach out at sudeshkharat26@gmail.com.*
