# ANPR (Automatic Number Plate Recognition) System

A modern ANPR system built with FastAPI, Triton Inference Server, Supervision for tracking, and PaddleOCR for text recognition.

## Features

- **FastAPI REST API**: Modern async API for image processing
- **Triton Inference Server**: High-performance inference with batching support
- **Supervision Integration**: Advanced object tracking with ByteTrack
- **Batch Processing**: Process multiple images efficiently
- **Folder Processing**: Standalone script to process entire folders
- **OCR Integration**: PaddleOCR for number plate text recognition

## Project Structure

```
triton_client_anpr/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   │           ├── health.py      # Health check endpoints
│   │           └── anpr.py        # ANPR processing endpoints
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   └── logging.py             # Logging setup
│   ├── schemas/
│   │   └── anpr.py                # Pydantic models
│   ├── services/
│   │   ├── triton_client.py       # Triton client with batching
│   │   ├── ocr.py                 # OCR service
│   │   └── anpr_service.py        # Main ANPR processing logic
│   └── main.py                    # FastAPI application
├── main.py                        # Standalone folder processing script
├── data/
│   ├── input/                     # Input images folder
│   └── output/                    # Output annotated images folder
├── .env                           # Environment configuration
└── pyproject.toml                 # Dependencies

```

## Installation

1. Install dependencies using Poetry or pip:

```bash
# Using Poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

2. Copy the example environment file and configure:

```bash
cp .env.example .env
# Edit .env with your Triton server details
```

## Usage

### 1. FastAPI Server

Start the FastAPI server:

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python
python -m app.main
```

Access the API documentation at: `http://localhost:8000/docs`

#### API Endpoints

- **GET /health**: Health check
- **POST /v1/anpr/process-image**: Process a single image (upload)
- **POST /v1/anpr/process-batch**: Process multiple images by paths
- **POST /v1/anpr/process-folder**: Process entire folder

#### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Process single image
curl -X POST "http://localhost:8000/v1/anpr/process-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"

# Process folder
curl -X POST "http://localhost:8000/v1/anpr/process-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_folder": "data/input",
    "output_folder": "data/output"
  }'
```

### 2. Standalone Folder Processing

Process images from a folder without running the API server:

```bash
python main.py \
  --input-folder data/input \
  --output-folder data/output \
  --server-url localhost:8000 \
  --model-name yolov8 \
  --batch-size 8 \
  --save-json
```

**Arguments:**
- `--input-folder`: Path to folder containing input images
- `--output-folder`: Path to save annotated images
- `--server-url`: Triton server URL (default: localhost:8000)
- `--model-name`: Model name on Triton server (default: yolov8)
- `--batch-size`: Number of images to process in each batch (default: 8)
- `--save-json`: Save detection results as JSON file

## Configuration

Edit `.env` file to configure:

```env
# Triton Server
TRITON_SERVER_URL=localhost:8000
TRITON_MODEL_NAME=yolov8

# Processing
BATCH_SIZE=8
INPUT_FOLDER=data/input
OUTPUT_FOLDER=data/output
CONFIDENCE_THRESHOLD=0.5
```

## Features Explained

### Triton Batching

The Triton client supports batch inference for improved performance:
- Automatically batches multiple images
- Preprocesses images (resize, pad, normalize)
- Efficient GPU utilization

### Supervision Tracking

Uses ByteTrack algorithm for object tracking:
- Tracks vehicles and number plates across frames
- Assigns unique IDs to detected objects
- Maintains tracking state

### OCR Integration

PaddleOCR for number plate text recognition:
- Automatic text detection and recognition
- Multi-language support
- High accuracy on license plates

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **Services Layer**: Business logic (Triton client, OCR, ANPR processing)
- **API Layer**: FastAPI routes and endpoints
- **Schemas**: Pydantic models for request/response validation
- **Core**: Configuration and logging

## Triton Server Setup

Ensure your Triton server is running with a YOLOv8 model:

```bash
# Example Triton server startup
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:latest \
  tritonserver --model-repository=/models
```

## License

MIT License
