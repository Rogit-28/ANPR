# Automatic Number Plate Recognition (ANPR) System

## Overview

The ANPR system detects and recognizes vehicle license plates from images or video streams. Built for Indian license plates, it handles diverse plate formats, various lighting conditions, and challenging scenarios.

**Tech Stack:**
- **Detection**: YOLOv11 (fine-tuned for license plates)
- **OCR**: RapidOCR (ONNX runtime, ~50-100ms per plate)
- **API**: FastAPI REST endpoints
- **Frontend**: Alpine.js + Tailwind CSS
- **Database**: SQLite

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (recommended)
- 8GB RAM minimum

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download YOLOv11 license plate model
python scripts/download_plate_model.py --variant s

# 4. Start the server
python src/main.py --mode api
```

The web UI is available at http://localhost:8000

## Usage

### Web Interface

1. Open http://localhost:8000
2. Upload a video or image
3. View detected plates in Results

### API Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Process image
curl -X POST http://localhost:8000/api/v1/image/process \
  -F "image=@plate.jpg"

# Upload video
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@video.mp4"

# Get detections
curl http://localhost:8000/api/v1/detections

# API documentation
open http://localhost:8000/api/v1/docs
```

### CLI Mode

```bash
python src/main.py --mode cli
```

## Configuration

Copy `config.template.yaml` to `config.yaml` to customize:

```yaml
models:
  yolo_model: yolov11s                           # n/s/m/l/x variants
  yolo_weights: models/yolov11s-license-plate.pt
  yolo_confidence: 0.4
  yolo_fp16: true                                # GPU acceleration

api:
  host: localhost
  port: 8000
```

Environment variables override config file:
- `ANPR_API_PORT` - API port
- `ANPR_MODELS_YOLO_CONFIDENCE` - Detection threshold
- `ANPR_MODELS_YOLO_FP16` - Enable/disable FP16

## Project Structure

```
ANPR/
├── src/anpr/
│   ├── api/          # FastAPI endpoints
│   ├── cli/          # CLI interface
│   ├── core/         # Detection, OCR, video processing
│   ├── database/     # SQLite models and queries
│   └── config/       # Configuration management
├── static/           # Frontend (Alpine.js + Tailwind)
├── models/           # YOLO weights
├── data/             # Uploads, snapshots, database
└── scripts/          # Utilities
```

## Development

```bash
# Run tests
python test_integration.py

# Check CUDA availability
python scripts/check_cuda.py

# Demo pipeline on sample images
python scripts/demo_pipeline.py
```

## Performance

On NVIDIA RTX 4050 (6GB):
- YOLOv11s inference: ~10ms per frame
- OCR extraction: ~50-100ms per plate
- Full pipeline: ~100-150ms per frame
- FPS: ~90-100 (detection only)

## Troubleshooting

**CUDA not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Model not found:**
```bash
python scripts/download_plate_model.py --variant s
```

**Memory issues:**
- Reduce `yolo_input_size` to 416
- Disable `yolo_fp16` if GPU memory is limited
  