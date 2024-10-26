import asyncio
import logging
import time
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum

from src.anpr.core.video_handler import VideoHandler
from src.anpr.core.yolo_detector import YOLODetector
from src.anpr.core.ocr_extractor import OCRExtractor
from src.anpr.core.night_vision_processor import NightVisionProcessor
from src.anpr.core.alert_manager import AlertManager
from src.anpr.core.memory_manager import MemoryManager
from src.anpr.database.manager import DatabaseManager
from src.anpr.database.models import Detection
from src.anpr.database.utils import insert_detection
from src.anpr.utils.image_utils import validate_indian_plate_format
from src.anpr.config.config import get_config


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class DetectionResult:
    """Result of a single detection pipeline execution."""
    frame_id: int
    plate_text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    original_frame_path: str
    cropped_plate_path: str
    timestamp: datetime
    camera_id: str
    processed_frame: Optional[np.ndarray] = None


@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""
    frame_skip_interval: int = 2  # Process every Nth frame (2 = 50% of frames)
    min_confidence_threshold: float = 0.5  # Lowered from 0.7 for better detection
    max_image_size: Tuple[int, int] = (1920, 1080)  # (width, height)
    duplicate_detection_window: int = 10  # seconds
    enable_night_vision: bool = False  # Disabled by default for performance
    batch_size: int = 1
    gpu_memory_fraction: float = 0.9  # 90% VRAM cap (increased from 0.8)
    save_cropped_plates: bool = True
    plate_validation_enabled: bool = True
    
    @classmethod
    def from_global_config(cls) -> 'PipelineConfig':
        """Create PipelineConfig from global configuration."""
        config = get_config()
        return cls(
            frame_skip_interval=config.pipeline.frame_skip_interval,
            min_confidence_threshold=config.pipeline.min_confidence_threshold,
            max_image_size=(config.pipeline.max_image_width, config.pipeline.max_image_height),
            duplicate_detection_window=config.pipeline.duplicate_detection_window,
            enable_night_vision=config.pipeline.enable_night_vision,
            batch_size=config.pipeline.batch_size,
            gpu_memory_fraction=config.models.gpu_memory_fraction,
            save_cropped_plates=config.pipeline.save_cropped_plates,
            plate_validation_enabled=config.pipeline.plate_validation_enabled,
        )


class PipelineManager:
    """
    Manages the ANPR detection pipeline orchestrating video ingestion,
    plate detection, OCR extraction, and result persistence.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        # Load config from global settings if not provided
        self.config = config or PipelineConfig.from_global_config()
        self.status = PipelineStatus.IDLE
        self._job_queue = asyncio.Queue()
        self._is_running = False
        
        # Initialize components
        self.video_handler: Optional[VideoHandler] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.ocr_extractor: Optional[OCRExtractor] = None
        self.night_vision_processor: Optional[NightVisionProcessor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.db_manager: Optional[DatabaseManager] = None
        
        # Performance tracking - use deque with maxlen to prevent memory leak
        self._max_timing_samples = 1000
        self.frame_processing_times: deque = deque(maxlen=self._max_timing_samples)
        self.detection_count = 0
        self.current_job_id = None
        
        # GPU memory management - use config value
        self.gpu_memory_threshold = self.config.gpu_memory_fraction
        
        # Performance optimization features
        self.frame_skip_interval = self.config.frame_skip_interval
        self.duplicate_cache = {}  # Cache for recently detected plates
        self.duplicate_window = self.config.duplicate_detection_window  # seconds
        self.processing_batch_size = self.config.batch_size
        self.memory_manager = MemoryManager(
            gpu_memory_threshold=self.config.gpu_memory_fraction
        )
        
        logger.info(f"PipelineManager initialized with GPU memory threshold: {self.gpu_memory_threshold:.0%}")
