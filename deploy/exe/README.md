# AI Vision Platform — Windows EXE Deployment

Run the entire platform as a standalone Windows application — no Docker, no
manual service setup.

## What gets bundled

| Component | How it runs |
|---|---|
| React frontend | Static files served by FastAPI |
| FastAPI + Uvicorn | PyInstaller bundle |
| Celery worker | Subprocess of the launcher |
| PostgreSQL 17 | Portable binaries extracted at runtime |
| Redis 7 | Portable binary extracted at runtime |

---

## Prerequisites before building

### 1. Portable PostgreSQL 17

Download the binary ZIP from EnterpriseDB:
```
https://www.enterprisedb.com/download-postgresql-binaries
```
- Choose **Windows x86-64**, version **17.x**.
- Extract so that the folder structure is:
  ```
  deploy/exe/resources/postgres/
  ├── bin/
  │   ├── pg_ctl.exe
  │   ├── initdb.exe
  │   ├── postgres.exe
  │   ├── psql.exe
  │   └── createdb.exe
  └── lib/
  ```

### 2. Portable Redis for Windows

Option A — Redis for Windows (archived, still works):
```
https://github.com/microsoftarchive/redis/releases
```
Extract `redis-server.exe` and `redis-cli.exe` to:
```
deploy/exe/resources/redis/
├── redis-server.exe
└── redis-cli.exe
```

Option B — Memurai (actively maintained Redis-compatible for Windows):
```
https://www.memurai.com/get-memurai
```

### 3. Python 3.11 with PyTorch (CUDA)

The build machine should have PyTorch installed with CUDA support:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Build

```cmd
cd deploy\exe
build.bat
```

Output: `deploy\exe\dist\AIVision\aivision.exe`

---

## Run

```cmd
cd deploy\exe\dist\AIVision
aivision.exe
```

On **first launch** the launcher will:
1. Verify CUDA / GPU availability and print a report
2. Initialize the PostgreSQL database cluster (`initdb`)
3. Create the `ai_vision` database
4. Start Redis
5. Start the FastAPI backend (with the React UI embedded)
6. Start the Celery worker
7. Open your browser at `http://localhost:8000`

On subsequent launches steps 2–3 are skipped (database already exists).

---

## Configuration

`aivision.cfg` is created next to the EXE on first run:

```ini
[aivision]
postgres_port = 5432
redis_port = 6379
api_port = 8000
db_name = ai_vision
db_user = aivision
db_password = aivision_local_pass
open_browser = true
skip_cuda_check = false
```

Edit this file to change ports or credentials.

---

## Data locations

All runtime data is stored **next to the EXE** (persists across updates):

```
dist/AIVision/
├── aivision.exe
├── aivision.cfg          ← generated on first run
├── data/
│   ├── pgdata/           ← PostgreSQL cluster
│   ├── redis/            ← Redis AOF persistence
│   ├── uploads/          ← user-uploaded files
│   └── models/           ← AI model weights
└── logs/
    ├── postgres.log
    ├── redis.log
    └── backend.log
```

---

## GPU / CUDA

See [CUDA_INSTALL_GUIDE.md](CUDA_INSTALL_GUIDE.md) for driver installation.

The launcher prints a verification report on every start. To skip it:
```ini
skip_cuda_check = true
```
