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
    SuccessResponse, ErrorResponse, HealthResponse, StatsResponse
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
    return {
        "status": "success",
        "message": "Configuration endpoint - not yet implemented"
    }
