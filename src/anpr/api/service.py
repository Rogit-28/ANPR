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


# ============================================
# Database Query Endpoints
# ============================================

@app.get("/api/v1/plates/search")
async def search_plates(
    plate_text: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Search plate detections with filters.
    
    Args:
        plate_text: Filter by plate text (partial match)
        start_time: Filter by start time (ISO format)
        end_time: Filter by end time (ISO format)
        min_confidence: Minimum confidence threshold (0-1)
        limit: Maximum results to return
        offset: Pagination offset
    """
    try:
        from ..database.utils import query_detections
        
        if not db_manager:
            handle_api_error("DB_NOT_INITIALIZED", "Database not initialized", status_code=500)
        
        with db_manager.get_session() as session:
            detections = query_detections(
                db_session=session,
                plate_text=plate_text,
                start_time=start_time,
                end_time=end_time,
                min_confidence=min_confidence,
                limit=limit,
                offset=offset
            )
            
            results = []
            for det in detections:
                results.append({
                    "id": det.id,
                    "plate_text": det.plate_text,
                    "confidence": det.confidence,
                    "timestamp": det.timestamp,
                    "source_type": det.source_type,
                    "source_identifier": det.source_identifier,
                    "snapshot_path": det.snapshot_path
                })
        
        return SuccessResponse(
            data={
                "total": len(results),
                "detections": results,
                "limit": limit,
                "offset": offset
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("SEARCH_ERROR", str(e))


@app.get("/api/v1/plates/recent")
async def get_recent_plates(limit: int = 10):
    """
    Get the most recent plate detections.
    
    Args:
        limit: Number of recent detections to return (max 100)
    """
    try:
        from ..database.utils import query_detections
        
        if not db_manager:
            handle_api_error("DB_NOT_INITIALIZED", "Database not initialized", status_code=500)
        
        # Clamp limit
        limit = min(max(1, limit), 100)
        
        with db_manager.get_session() as session:
            detections = query_detections(
                db_session=session,
                limit=limit,
                offset=0
            )
            
            results = []
            for det in detections:
                results.append({
                    "id": det.id,
                    "plate_text": det.plate_text,
                    "confidence": det.confidence,
                    "timestamp": det.timestamp,
                    "source_type": det.source_type,
                    "source_identifier": det.source_identifier
                })
        
        return SuccessResponse(
            data={
                "count": len(results),
                "detections": results
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("RECENT_PLATES_ERROR", str(e))


# ============================================
# Video Processing Endpoints
# ============================================

@app.post("/api/v1/upload")
async def upload_video(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None)
):
    """
    Upload a video file for processing.
    
    Accepts video files up to 500MB. Supported formats: mp4, avi, mov, mkv, webm.
    Returns a job ID that can be used to track processing status.
    """
    try:
        # Sanitize and validate filename
        raw_filename = file.filename or "upload.mp4"
        filename = sanitize_filename(raw_filename)
        ext = Path(filename).suffix.lower()
        
        # Ensure extension is preserved after sanitization
        if not ext:
            ext = Path(raw_filename).suffix.lower()
            filename = filename + ext
        
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        
        if ext not in allowed_extensions:
            handle_api_error(
                "INVALID_FILE_TYPE",
                f"File type {ext} not supported. Allowed: {', '.join(allowed_extensions)}",
                status_code=400
            )
        
        # Read file content in chunks with size tracking
        content_chunks = []
        total_size = 0
        
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE:
                handle_api_error(
                    "FILE_TOO_LARGE",
                    f"File exceeds maximum 500MB",
                    status_code=400
                )
            content_chunks.append(chunk)
        
        content = b''.join(content_chunks)
        size_mb = total_size / (1024 * 1024)
        
        logger.info(f"Received upload: {filename} ({size_mb:.1f}MB)")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Track the job
        running_jobs[job_id] = {
            "type": "video_upload",
            "status": "queued",
            "start_time": datetime.now().isoformat(),
            "filename": filename,
            "size_mb": size_mb,
            "camera_id": camera_id
        }
        
        # TODO: Save file and queue for processing
        
        return SuccessResponse(
            data={
                "job_id": job_id,
                "status": "queued",
                "message": f"Video uploaded and queued for processing",
                "filename": filename
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("UPLOAD_ERROR", str(e))


@app.get("/api/v1/video/status/{job_id}")
async def get_video_status(job_id: str):
    """
    Get video processing status.
    
    Returns the current status of a video processing job.
    """
    try:
        # Check if job exists in our tracking
        if job_id in running_jobs:
            job_info = running_jobs[job_id]
            return SuccessResponse(
                data={
                    "job_id": job_id,
                    "status": job_info.get("status", "unknown"),
                    "type": job_info.get("type"),
                    "start_time": job_info.get("start_time"),
                    "filename": job_info.get("filename"),
                    "progress": job_info.get("progress", 0),
                    "error": job_info.get("error")
                }
            )
        else:
            handle_api_error(
                "JOB_NOT_FOUND",
                f"Job {job_id} not found",
                status_code=404
            )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("VIDEO_STATUS_ERROR", str(e))


# ============================================
# Job Management Endpoints
# ============================================

@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """
    List all processing jobs.
    
    Returns paginated list of jobs, ordered by creation time (newest first).
    """
    try:
        # Get jobs from running_jobs dict for now
        jobs_list = list(running_jobs.items())
        
        # Sort by start_time descending
        jobs_list.sort(key=lambda x: x[1].get("start_time", ""), reverse=True)
        
        # Apply pagination
        total = len(jobs_list)
        jobs_list = jobs_list[offset:offset + limit]
        
        job_responses = []
        for job_id, job_info in jobs_list:
            job_responses.append({
                "id": job_id,
                "status": job_info.get("status"),
                "type": job_info.get("type"),
                "start_time": job_info.get("start_time"),
                "filename": job_info.get("filename"),
                "progress": job_info.get("progress", 0),
                "camera_id": job_info.get("camera_id")
            })
        
        return SuccessResponse(
            data={
                "total": total,
                "jobs": job_responses,
                "limit": limit,
                "offset": offset
            }
        )
        
    except Exception as e:
        handle_api_error("JOBS_LIST_ERROR", str(e))


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get details of a specific job.
    """
    try:
        if job_id not in running_jobs:
            handle_api_error("JOB_NOT_FOUND", f"Job {job_id} not found", status_code=404)
        
        job_info = running_jobs[job_id]
        
        return SuccessResponse(
            data={
                "id": job_id,
                "status": job_info.get("status"),
                "type": job_info.get("type"),
                "start_time": job_info.get("start_time"),
                "filename": job_info.get("filename"),
                "progress": job_info.get("progress", 0),
                "camera_id": job_info.get("camera_id"),
                "error": job_info.get("error")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("JOB_GET_ERROR", str(e))


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a pending or processing job.
    """
    try:
        if job_id not in running_jobs:
            handle_api_error("JOB_NOT_FOUND", f"Job {job_id} not found", status_code=404)
        
        job_info = running_jobs[job_id]
        current_status = job_info.get("status")
        
        # Can only cancel pending or processing jobs
        if current_status in ["completed", "cancelled", "error"]:
            handle_api_error(
                "CANCEL_FAILED",
                f"Cannot cancel job with status: {current_status}",
                status_code=400
            )
        
        # Update status
        running_jobs[job_id]["status"] = "cancelled"
        
        return SuccessResponse(
            data={
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job cancelled successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("CANCEL_ERROR", str(e))


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and all associated files.
    """
    try:
        if job_id not in running_jobs:
            handle_api_error("JOB_NOT_FOUND", f"Job {job_id} not found", status_code=404)
        
        # Remove from tracking
        del running_jobs[job_id]
        
        # TODO: Delete associated files
        
        return SuccessResponse(
            data={
                "job_id": job_id,
                "status": "deleted",
                "message": "Job deleted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        handle_api_error("DELETE_ERROR", str(e))


# ============================================
# Static File Serving
# ============================================

# Determine the static files directory
static_dir = Path(__file__).parent.parent.parent.parent / "static"
dist_dir = static_dir / "dist"

# Use dist directory if it exists (production build), otherwise use static root
frontend_dir = dist_dir if dist_dir.exists() else static_dir

if frontend_dir.exists():
    # Mount the assets folder for JS/CSS bundles
    assets_dir = frontend_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    # Also mount the static folder for other static files (images, etc.)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    # Try dist/index.html first (production build)
    index_path = dist_dir / "index.html"
    if not index_path.exists():
        # Fall back to static/index.html (development)
        index_path = static_dir / "index.html"
    
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return SuccessResponse(
            data={
                "message": "ANPR API is running",
                "docs": "/api/v1/docs",
                "health": "/api/health"
            }
        )


# ============================================
# WebSocket for Real-time Updates
# ============================================

# SSE Alert Queue - stores alerts to be pushed to connected clients
alert_queues: List[asyncio.Queue] = []
alert_queues_lock = asyncio.Lock()
sse_shutdown_event = asyncio.Event()


async def broadcast_alert(alert_data: Dict[str, Any]):
    """
    Broadcast an alert to all connected SSE clients.
    
    Args:
        alert_data: Alert data dictionary to send to clients
    """
    async with alert_queues_lock:
        client_count = len(alert_queues)
        if not client_count:
            logger.debug("No SSE clients connected for alert broadcast")
            return
        
        logger.info(f"Broadcasting alert to {client_count} SSE clients")
        
        for queue in alert_queues:
            try:
                queue.put_nowait(alert_data)
            except asyncio.QueueFull:
                logger.warning("SSE client queue full, dropping alert")
            except Exception as e:
                logger.debug(f"Failed to queue alert for SSE client: {e}")


async def alert_event_generator(queue: asyncio.Queue):
    """
    Generator that yields SSE events from the alert queue.
    """
    try:
        # Send initial connection event
        client_count = len(alert_queues)
        yield f"event: connected\ndata: {json.dumps({'message': 'Connected to alert stream', 'clients': client_count})}\n\n"
        
        while not sse_shutdown_event.is_set():
            try:
                alert_data = await asyncio.wait_for(queue.get(), timeout=15.0)
                
                if isinstance(alert_data, dict) and alert_data.get("__shutdown__"):
                    yield f"event: shutdown\ndata: {json.dumps({'message': 'Server shutting down'})}\n\n"
                    break
                
                event_data = json.dumps(alert_data)
                yield f"event: alert\ndata: {event_data}\n\n"
                
            except asyncio.TimeoutError:
                if sse_shutdown_event.is_set():
                    break
                yield ": keepalive\n\n"
                
    except GeneratorExit:
        logger.debug("SSE client disconnected")
    except asyncio.CancelledError:
        logger.debug("SSE generator cancelled")
    except Exception as e:
        logger.error(f"SSE generator error: {e}")


@app.get("/api/v1/alerts/stream")
async def alert_stream(request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time alert notifications.
    
    Connect to this endpoint to receive instant push notifications when
    alerts are triggered (watchlist matches, new plates, etc.)
    """
    queue = asyncio.Queue(maxsize=100)
    
    async with alert_queues_lock:
        alert_queues.append(queue)
        logger.info(f"SSE client connected. Total clients: {len(alert_queues)}")
    
    async def event_generator_with_cleanup():
        try:
            async for event in alert_event_generator(queue):
                yield event
        finally:
            async with alert_queues_lock:
                if queue in alert_queues:
                    alert_queues.remove(queue)
                    logger.info(f"SSE client removed. Remaining clients: {len(alert_queues)}")
    
    return StreamingResponse(
        event_generator_with_cleanup(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/v1/alerts/stream/status")
async def get_sse_status():
    """
    Get the status of SSE alert connections.
    """
    async with alert_queues_lock:
        client_count = len(alert_queues)
    
    return SuccessResponse(
        data={
            "connected_clients": client_count,
            "endpoint": "/api/v1/alerts/stream"
        }
    )
