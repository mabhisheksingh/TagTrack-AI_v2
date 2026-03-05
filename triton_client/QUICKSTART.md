# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Triton server details
```

## 3. Run the Application

### Option A: FastAPI Server

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Access API docs
open http://localhost:8000/docs
```

### Option B: Folder Processing (Standalone)

```bash
# Place images in data/input/
# Run processing
python main.py --save-json

# Check results in data/output/
```

## 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Process folder via API
curl -X POST "http://localhost:8000/v1/anpr/process-folder" \
  -H "Content-Type: application/json" \
  -d '{
    "input_folder": "data/input",
    "output_folder": "data/output"
  }'
```

## 5. Using Docker

```bash
# Build and run
docker-compose up

# API will be available at http://localhost:8000
```

## Common Commands

```bash
# Run with custom settings
python main.py \
  --input-folder /path/to/images \
  --output-folder /path/to/output \
  --batch-size 16 \
  --server-url localhost:8000 \
  --model-name yolov8

# Run API with custom port
uvicorn app.main:app --port 8080

# Run tests
pytest tests/ -v

# Clean cache
make clean
```

## Folder Structure

```
data/
├── input/          # Place your images here
└── output/         # Processed images appear here
```

## Expected Output

- **Annotated images** with bounding boxes and labels
- **JSON results** (if --save-json flag used)
- **Console output** with detection details
- **Tracking IDs** for vehicles and plates
- **OCR text** for detected number plates
