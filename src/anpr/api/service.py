import asyncio
import json
import logging
import re
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# NVIDIA Management Library for GPU utilization
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

from .models import (
    SuccessResponse, ErrorResponse, HealthResponse, StatsResponse,
    ConfigResponse, SystemInfoResponse
)

logger = logging.getLogger(__name__)

# Global instances for the API service
pipeline_manager = None
db_manager = None
alert_manager = None
job_queue = None
video_annotator = None

# Track running jobs
running_jobs: Dict[str, Dict[str, Any]] = {}
start_time = None

# Max upload size (500MB)
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# Chunk size for streaming uploads
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB


def format_uptime(seconds: float) -> str:
    """Format uptime in a human-readable way.
    
    - Under 60s: shows as seconds (e.g., "45.23s")
    - 1-60 mins: shows as mins and secs (e.g., "5m 30s")
    - Over 60 mins: shows as hours and mins (e.g., "2h 15m")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal and other issues.
    
    Args:
        filename: Original filename from upload
        
    Returns:
        Sanitized filename safe for filesystem storage
    """
    # Get just the filename, remove any path components
    filename = os.path.basename(filename)
    # Replace any character that's not alphanumeric, dash, underscore, or dot
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    # Ensure filename is not empty
    return filename or "upload"


# Create FastAPI app
app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition API Service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def init_api_components():
    """Initialize the API components."""
    global pipeline_manager, db_manager, alert_manager, job_queue, video_annotator, start_time
    
    start_time = time.time()
    logger.info("Initializing API components...")
    
    # Initialize database manager
    from ..database.manager import DatabaseManager
    db_manager = DatabaseManager()
    db_manager.init_database()
    
    logger.info("API components initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await init_api_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db_manager
    
    if db_manager:
        db_manager.close()
    
    logger.info("API shutdown complete")


# ============================================
# Health & Status Endpoints
# ============================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health status."""
    global start_time
    
    uptime_seconds = time.time() - start_time if start_time else 0
    
    # Check GPU availability
    gpu_available = False
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_available = True
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime=format_uptime(uptime_seconds),
        gpu_available=gpu_available,
        database_connected=db_manager.test_connection() if db_manager else False
    )


@app.get("/api/config")
async def get_config_endpoint():
    """Get current configuration."""
    from .models import ConfigResponse
    
    # Return default configuration values
    return ConfigResponse(
        alerts_enabled=True,
        desktop_notifications=False,
        yolo_confidence=0.25,
        yolo_weights_path="models/yolov8n.pt",
        min_ocr_confidence=0.5,
        plate_validation_enabled=True
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    from ..database.utils import get_detection_stats
    
    stats = {"total_detections": 0, "average_confidence": 0.0}
    
    if db_manager:
        with db_manager.get_session() as session:
            stats = get_detection_stats(session)
    
    return StatsResponse(
        total_detections=stats.get('total_detections', 0),
        average_confidence=stats.get('average_confidence', 0.0),
        earliest_detection=stats.get('earliest_detection'),
        latest_detection=stats.get('latest_detection'),
        pipeline_stats=None,
        memory_usage=None
    )


@app.get("/api/system-info")
async def get_system_info():
    """Get extended system information."""
    import sys
    from .models import SystemInfoResponse
    
    global start_time
    
    uptime_seconds = time.time() - start_time if start_time else 0
    
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_memory_total = None
    gpu_memory_used = None
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_available = True
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        
        # Get GPU memory info
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_total = f"{info.total / (1024**3):.2f} GB"
                gpu_memory_used = f"{info.used / (1024**3):.2f} GB"
                pynvml.nvmlShutdown()
            except Exception:
                pass
    
    return SystemInfoResponse(
        api_version="1.0.0",
        python_version=sys.version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
        gpu_memory_used=gpu_memory_used,
        gpu_utilization=None,
        database_path="data/anpr.db",
        config_file_path="config.yaml",
        uptime=format_uptime(uptime_seconds),
        uptime_seconds=uptime_seconds,
        onnx_providers=[]
    )


def handle_api_error(error_code: str, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
    """Helper function to create standardized error responses."""
    logger.error(f"API Error - {error_code}: {message}, Details: {details}")
    raise HTTPException(
        status_code=status_code,
        detail=ErrorResponse(
            error_code=error_code,
            message=message,
            details=details
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal server error occurred",
            details={
                "error_type": type(exc).__name__,
                "request_url": str(request.url),
                "request_method": request.method
            }
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    # If detail is already an ErrorResponse, return it as is
    if isinstance(exc.detail, dict) and 'error_code' in exc.detail:
        error_response = ErrorResponse(**exc.detail)
    else:
        error_response = ErrorResponse(
            error_code="HTTP_ERROR",
            message=str(exc.detail) if exc.detail else "HTTP Error occurred",
            details={
                "status_code": exc.status_code,
                "request_url": str(request.url),
                "request_method": request.method
            }
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


# ============================================
# Detection Endpoints
# ============================================

@app.post("/api/v1/detect/image")
async def detect_image(image: UploadFile = File(...)):
    """
    Process a single image for license plate detection.
    
    Accepts image files (jpg, png, bmp, webp). Returns detected plates with OCR results.
    """
    try:
        # Validate file type
        allowed_types = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        filename = image.filename or "image.jpg"
        ext = Path(filename).suffix.lower()
        
        if ext not in allowed_types:
            handle_api_error(
                "INVALID_FILE_TYPE",
                f"Image type {ext} not supported. Allowed: {', '.join(allowed_types)}",
                status_code=400
            )
        
        # Read image content
        content = await image.read()
        
        # Decode image
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            handle_api_error("INVALID_IMAGE", "Could not decode image", status_code=400)
        
        h, w = frame.shape[:2]
        
        # TODO: Implement actual detection pipeline
        # For now, return placeholder response
        return SuccessResponse(
            data={
                "detections": [],
                "image_width": w,
                "image_height": h,
                "processing_time_ms": 0
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("IMAGE_DETECT_ERROR", str(e))


@app.post("/api/v1/detect/video")
async def detect_video(video: UploadFile = File(...)):
    """
    Process a video file for license plate detection.
    
    Accepts video files (mp4, avi, mov, mkv). Returns job ID for tracking.
    """
    try:
        # Validate file type
        allowed_types = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        filename = video.filename or "video.mp4"
        ext = Path(filename).suffix.lower()
        
        if ext not in allowed_types:
            handle_api_error(
                "INVALID_FILE_TYPE",
                f"Video type {ext} not supported. Allowed: {', '.join(allowed_types)}",
                status_code=400
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # TODO: Save video and queue for processing
        
        return SuccessResponse(
            data={
                "job_id": job_id,
                "status": "queued",
                "message": f"Video {filename} queued for processing"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("VIDEO_DETECT_ERROR", str(e))
