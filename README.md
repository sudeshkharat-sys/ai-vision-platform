# AI Vision Platform

An end-to-end computer vision platform for managing datasets, annotating images, and training YOLO models (v8 through v11). The platform features an automated "Seed & Scale" workflow to accelerate labeling through AI-assisted auto-annotation.

## Project Structure

```text
ai-vision-platform/
├── backend/                # FastAPI Application
│   ├── app/
│   │   ├── api/            # API Endpoints (Auth, Projects, Images, Pipeline)
│   │   ├── connectors/     # DB Connectors (SQLAlchemy, Psycopg2)
│   │   ├── models/         # SQLAlchemy Models
│   │   ├── tasks/          # Celery Tasks (Training, Auto-annotation)
│   │   ├── main.py         # Entry point
│   │   └── config.py       # Configuration & Settings
│   ├── data/               # Persistent Storage (Uploads, Models)
│   └── requirements.txt    # Python dependencies
├── frontend/               # React Application
│   ├── src/
│   │   ├── components/     # UI Components (Workspace, Panels, Charts)
│   │   ├── assets/         # Static assets
│   │   └── App.js          # Root component
│   └── package.json        # Node.js dependencies
└── README.md
```

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **PostgreSQL**: Primary database for metadata and annotations.
- **Redis**: Message broker for Celery tasks.

### 1. Setup Services

Ensure PostgreSQL and Redis are running. Create a database for the project (e.g., `ai_vision`).

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file from example
cp .env.example .env  # On Windows: copy .env.example .env

# Configure your credentials in .env:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_vision
# CELERY_BROKER_URL=redis://localhost:6371/0
# CELERY_RESULT_BACKEND=redis://localhost:6371/0

# Start the FastAPI server
uvicorn app.main:app --port 8000 --reload
```

### 3. Celery Worker Setup

Open a new terminal and run:

```bash
cd backend
# On Windows, --pool=solo is recommended for YOLO training tasks
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo
```

## AI Prompt Feature

The platform includes a powerful **AI Prompt** (Zero-Shot) feature that allows you to detect objects using natural language (e.g., "detect all blue cars"). This is powered by a 3-stage pipeline:

1.  **Discovery (Grounding DINO)**: Finds potential object candidates based on the text prompt.
2.  **Verification (SigLIP)**: Cross-verifies candidates to ensure high precision and reduce false positives.
3.  **Segmentation (SAM 2)**: Refines the bounding boxes and can generate masks for precise localization.

### AI Model Requirements

These models are large and should be placed in the `datavision_hf_models/` directory at the project root. You can download them using the `huggingface-cli`:

```bash
# 1. Install huggingface-cli if you haven't already
pip install "huggingface_hub[cli]"

# 2. Create the directory
mkdir datavision_hf_models
cd datavision_hf_models

# 3. Download Grounding DINO Base
huggingface-cli download IDEA-Research/grounding-dino-base --local-dir grounding-dino-base

# 4. Download SAM 2 Hiera Large
huggingface-cli download facebook/sam2-hiera-large --local-dir sam2-hiera-large

# 5. Download SigLIP SO400M
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
```

Ensure the directory structure looks like this:
```text
ai-vision-platform/
└── datavision_hf_models/
    ├── grounding-dino-base/
    ├── sam2-hiera-large/
    └── siglip-so400m-patch14-384/
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm start
```

## Training Workflow (Seed & Scale)

The platform follows a structured workflow to minimize manual labeling effort:

1.  **Upload Images**: Create a project and upload your raw image dataset.
2.  **Manual Annotation (5%)**: Annotate a small subset of your data (typically 5-10%) manually in the **Annotation Workspace**.
3.  **Train Seed Model**:
    - Go to the **Training Panel**.
    - Select a base YOLO model (e.g., YOLO11n).
    - Run "Seed Training" on your manually annotated images.
4.  **Auto-Annotate**:
    - Use the **Auto-Annotate Panel**.
    - Run the trained Seed Model on the remaining "Pending" images.
    - Review and correct auto-annotations as needed.
5.  **Train Main Model**:
    - Once the dataset is fully (or mostly) annotated, run "Main Training".
    - This trains on the entire dataset (Manual + Auto).
    - You can choose to fine-tune from your Seed Model weights for better results.
6.  **Download Weights**:
    - Once finished, download the `main_best.pt` weights directly from the UI for deployment.

## Key Features

- **Multi-Model Support**: Integrated with Ultralytics YOLOv8, v9, v10, and v11.
- **Rich Annotation Tool**: Canvas-based bbox drawing with zoom, pan, and label management.
- **Real-time Metrics**: Live training charts showing Box Loss, Class Loss, and mAP during training.
- **AI Prompting**: Support for text-prompt based detection (e.g., "detect all helmets") using zero-shot models.
- **Async Processing**: Long-running training jobs are handled in the background via Celery.

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Celery, Redis, Ultralytics YOLO.
- **Frontend**: React, Konva.js (Canvas), Recharts (Metrics), Axios.
- **Database**: PostgreSQL.
