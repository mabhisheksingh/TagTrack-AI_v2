# TagTrack-AI_v2

TagTrack-AI v2 is the refreshed Automatic Number Plate Recognition (ANPR) platform. It splits responsibilities into two deployable projects:

1. **`triton-server/`** – Packages NVIDIA Triton with RT-DETR TorchScript models for high-throughput vehicle and plate-region detection (batch friendly, CPU/GPU aware).
2. **`triton_client/`** – A FastAPI + Supervision application that streams frames to Triton, tracks vehicles, runs PaddleOCR, and exposes REST/batch tooling for license plate intelligence.

## Repository Layout

```
TagTrack-AI_v2/
├── triton-server/      # Triton configs, model repository, uv tooling
├── triton_client/      # FastAPI service, CLI processors, docs
└── README.md           # Overview (this file)
```

Each subfolder contains its own README / guides for deeper setup steps.

## Goals

- Keep Triton focused on inference: RT-DETR models served via TorchScript with dynamic batching.
- Let the client layer manage orchestration: tracking, OCR, APIs, folder/batch pipelines.
- Maintain a clean contract between both halves (model names, input shapes, ports, health checks).

## Triton Server Summary (`triton-server/`)

- **Model repository**: `model_repository/<model>/config.pbtxt` + TorchScript weights under numeric version folders.
- **Dynamic batching**: `max_batch_size` + `preferred_batch_size` tuned for CPU/GPU.
- **Operational docs**: `README.md` + `triton_health_checks.md` cover conversion to TorchScript, verbose logging, polling reloads, and health endpoints.
- **Smoke test**: `uv run test-image` sends the bundled `Cars.jpg` to verify end-to-end inference.

## Client Summary (`triton_client/`)

- **FastAPI REST API**: `/v1/anpr/*` endpoints for single image, batch, and folder processing.
- **Supervision + ByteTrack**: Tracks vehicles across frames, keeps IDs, and annotates outputs.
- **PaddleOCR**: Extracts license plate text after detections are cropped.
- **Deployment options**: uv/venv, Docker, or docker-compose (see `README_NEW.md`, `QUICKSTART.md`, `PROJECT_SUMMARY.md`).

## Quick Start

### 1. Launch Triton

```bash
cd triton-server
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --log-verbose=1

# Optional local smoke test
uv run test-image
```

Ensure logs show both RT-DETR models as `READY` and health endpoints return `200`.

### 2. Run the Client

```bash
cd ../triton_client
cp .env.example .env   # set TRITON_SERVER_URL, OCR options, etc.
uv run local_setup.py  # installs deps
uv run uvicorn app.main:app --reload
```

Invoke the API:

```bash
curl -X POST "http://localhost:8000/v1/anpr/process-image" -F "file=@data/input/car.jpg"
```

### 3. End-to-End Flow

1. Triton detects vehicles & plate regions via RT-DETR.
2. Client assigns tracker IDs (Supervision/ByteTrack) and crops plates.
3. PaddleOCR extracts text; responses include JSON + annotated artifacts.

## Additional Resources

- `triton-server/README.md` – deep dive on Triton quickstart, TorchScript export, health checks.
- `triton-server/triton_health_checks.md` – what to expect in verbose logs and how to debug readiness.
- `triton_client/PROJECT_SUMMARY.md` – architecture overview, workflow diagrams, configuration options.
- `triton_client/README_NEW.md` & `QUICKSTART.md` – operational guides, docker-compose scenarios.

Use this root README as the map; jump into the subproject you’re working on for step-by-step details. Happy tagging! 🏎️📸
