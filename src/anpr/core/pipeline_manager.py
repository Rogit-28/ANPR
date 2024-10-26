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

    async def initialize_components(self):
        """Initialize all required components for the pipeline."""
        try:
            # Get configuration
            config = get_config()
            
            # Initialize video handler
            self.video_handler = VideoHandler()
            
            # Initialize YOLO detector with config settings
            self.yolo_detector = YOLODetector(
                model_path=config.models.yolo_weights,
                confidence_threshold=config.models.yolo_confidence,
                iou_threshold=config.models.yolo_iou,
                input_size=config.models.yolo_input_size,
                fp16=config.models.yolo_fp16
            )
            
            # Initialize OCR extractor
            self.ocr_extractor = OCRExtractor()
            
            # Initialize night vision processor
            self.night_vision_processor = NightVisionProcessor()
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            
            # Initialize alert manager
            self.alert_manager = AlertManager()
            
            logger.info(f"Pipeline initialized with YOLOv11 model: {config.models.yolo_weights}")
            logger.info(f"Detection settings: conf={config.models.yolo_confidence}, iou={config.models.yolo_iou}, fp16={config.models.yolo_fp16}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise

    async def start_pipeline(self, video_source: str, camera_id: str, job_id: str):
        """
        Start the detection pipeline for a given video source.
        
        Args:
            video_source: Path to video file or URL for live stream
            camera_id: Unique identifier for the camera/source
            job_id: Unique identifier for this processing job
        """
        if self.status != PipelineStatus.IDLE:
            raise RuntimeError(f"Pipeline is not idle. Current status: {self.status.value}")
        
        self.current_job_id = job_id
        self.status = PipelineStatus.PROCESSING
        self._is_running = True
        
        logger.info(f"Starting pipeline for video source: {video_source}, camera: {camera_id}, job: {job_id}")
        
        try:
            await self._run_pipeline(video_source, camera_id)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.status = PipelineStatus.ERROR
            raise
        finally:
            self._is_running = False
            # Reset to IDLE after completion so next job can start
            # Only keep ERROR status if that's what was set
            if self.status == PipelineStatus.PROCESSING:
                self.status = PipelineStatus.IDLE
            elif self.status != PipelineStatus.ERROR:
                # Completed successfully, reset to IDLE
                self.status = PipelineStatus.IDLE
            logger.info(f"Pipeline completed for job: {job_id}, status reset to: {self.status.value}")

    async def _run_pipeline(self, video_source: str, camera_id: str):
        """Main pipeline execution loop."""
        logger.info("Pipeline execution started")
        
        # Initialize components if not already done
        if not all([self.video_handler, self.yolo_detector, self.ocr_extractor]):
            await self.initialize_components()
        
        # Verify YOLO model is actually loaded (defensive check)
        if not self.yolo_detector.is_model_loaded():
            logger.warning("YOLO model not loaded, attempting to load now...")
            if not self.yolo_detector.load_model():
                raise RuntimeError("Failed to load YOLO model")
        
        frame_count = 0
        processed_frames = 0  # Frames actually sent through detection pipeline
        detection_count = 0   # Successful detections found
        
        # Determine if source is a file (no retry needed) or a live stream (retry on error)
        is_live_stream = video_source.startswith(('rtsp://', 'http://', 'https://'))
        max_stream_retries = 3 if is_live_stream else 1  # No retries for file uploads
        
        try:
            # Process video stream using VideoHandler with error handling
            stream_retry_count = 0
            
            while stream_retry_count < max_stream_retries and self._is_running:
                try:
                    # Create the generator - this is synchronous but doesn't block yet
                    frame_generator = self.video_handler.process_stream(video_source)
                    
                    while self._is_running:
                        # Check for pause status
                        if self.status == PipelineStatus.PAUSED:
                            await asyncio.sleep(0.1)
                            continue
                        
                        # Get next frame in a thread to avoid blocking the event loop
                        # This is critical: process_stream's stdout.read() is blocking I/O
                        frame = await asyncio.to_thread(next, frame_generator, None)
                        
                        if frame is None:
                            # Stream ended
                            logger.info(f"Video stream ended after {frame_count} frames read")
                            break
                        
                        frame_count += 1
                        
                        # Skip frames based on interval setting
                        if frame_count % self.config.frame_skip_interval != 0:
                            # Yield control periodically to prevent blocking even when skipping
                            if frame_count % 10 == 0:
                                await asyncio.sleep(0)
                            continue
                        
                        processed_frames += 1  # Count frames sent to pipeline
                        
                        # Check GPU memory periodically (every 100 frames)
                        if processed_frames % 100 == 0:
                            await asyncio.to_thread(self._check_and_manage_gpu_memory)
                        
                        # Process frame through pipeline
                        try:
                            # Yield control before heavy processing
                            await asyncio.sleep(0)
                            detection_result = await self._process_single_frame(frame, frame_count, camera_id)
                            
                            if detection_result:
                                # Store result in database
                                await self._store_detection_result(detection_result)
                                
                                # Trigger alerts if needed
                                await self._trigger_alerts_if_needed(detection_result)
                                
                                detection_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_count}: {e}")
                            # Continue processing other frames despite individual frame failures
                            
                        # Report progress periodically (every 50 processed frames)
                        if processed_frames % 50 == 0:
                            logger.info(f"Progress: {processed_frames} frames processed, {detection_count} detections, {frame_count} total read")
                    
                    # If we get here, the stream ended normally
                    logger.info(f"Video stream processing completed normally. Exiting retry loop.")
                    break
                    
                except Exception as stream_error:
                    stream_retry_count += 1
                    logger.warning(f"Stream error occurred (attempt {stream_retry_count}/{max_stream_retries}): {stream_error}")
                    
                    if is_live_stream and stream_retry_count < max_stream_retries:
                        logger.info("Attempting to reconnect to live stream...")
                        await asyncio.sleep(5)  # Wait before retrying
                    else:
                        if not is_live_stream:
                            logger.error("Video file processing failed, not retrying to avoid duplicate processing")
                        else:
                            logger.error("Max stream retries reached, stopping pipeline")
                        raise
                        
        except Exception as e:
            logger.error(f"Error in video stream processing: {e}")
            self.status = PipelineStatus.ERROR
            raise
        finally:
            logger.info(f"Pipeline finished. Processed {processed_frames} frames, found {detection_count} detections out of {frame_count} total frames read")

    async def _process_single_frame(self, frame: np.ndarray, frame_id: int, camera_id: str) -> Optional[DetectionResult]:
        """Process a single frame through the detection pipeline."""
        start_time = time.time()
        
        try:
            # Preprocess frame for night vision only if enabled
            if self.config.enable_night_vision:
                try:
                    processed_frame, _ = self.night_vision_processor.process_with_fallback(frame)
                except Exception as e:
                    logger.warning(f"Night vision processing failed: {e}, using original frame")
                    processed_frame = frame
                    # Disable night vision for the rest of the session on GPU errors
                    if "CUDA" in str(e).upper() or "GPU" in str(e).upper():
                        logger.info("Disabling night vision processing due to GPU error")
                        self.config.enable_night_vision = False
            else:
                processed_frame = frame
            
            # Run YOLO detection to locate license plates
            try:
                # Run YOLO in a thread to prevent blocking the event loop
                detections = await asyncio.to_thread(
                    self.yolo_detector.detect_license_plate, processed_frame
                )
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")
                return None
            
            # Filter detections by confidence threshold
            high_conf_detections = [
                det for det in detections
                if det['confidence'] >= self.config.min_confidence_threshold
            ]
            
            if not high_conf_detections:
                if detections:
                    logger.debug(f"[SKIP] {len(detections)} detection(s) below confidence threshold {self.config.min_confidence_threshold}")
                return None
            
            logger.debug(f"Processing {len(high_conf_detections)} high-confidence detection(s) from frame {frame_id}")
            
            # Process each detected plate
            for detection in high_conf_detections:
                bbox = detection['bbox']
                
                # Extract plate region with padding
                x1, y1, x2, y2 = map(int, bbox)
                
                # Add 10% padding around the plate region
                height, width = processed_frame.shape[:2]
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(width, x2 + pad_x)
                y2 = min(height, y2 + pad_y)
                
                plate_region = processed_frame[y1:y2, x1:x2]
                
                if plate_region.size == 0:
                    continue
                
                # Perform OCR on the plate region with error handling
                # Run OCR in a thread pool to prevent blocking the async event loop
                try:
                    ocr_result = await asyncio.to_thread(
                        self.ocr_extractor.extract_with_validation, plate_region
                    )
                except Exception as e:
                    logger.error(f"OCR extraction failed: {e}")
                    continue
                
                # Extract text and confidence from OCR result
                plate_text = ocr_result['text']
                ocr_confidence = ocr_result['aggregate_confidence']
                
                # Get current timestamp for duplicate checking and result creation
                current_timestamp = datetime.now()
                
                # Check for duplicate detection
                if self._is_duplicate_detection(plate_text, current_timestamp):
                    logger.info(f"[SKIP] Duplicate plate '{plate_text}' detected within {self.duplicate_window}s window")
                    continue
                
                # Update duplicate cache
                self._update_duplicate_cache(plate_text, current_timestamp)
                
                # Validate plate format if enabled
                if self.config.plate_validation_enabled:
                    # First check the OCR result format validation
                    if not ocr_result['format_valid']:
                        logger.info(f"[SKIP] Plate '{plate_text}' failed OCR format validation (conf={ocr_confidence:.2f})")
                        continue
                    
                    # Additional validation using the image_utils function
                    if not validate_indian_plate_format(plate_text):
                        logger.info(f"[SKIP] Plate '{plate_text}' failed image_utils format validation")
                        continue
                
                # Generate unique filenames for saving
                timestamp_str = current_timestamp.strftime("%Y%m%d_%H%M%S_%f")
                
                # Ensure detections directory exists
                detections_dir = Path("detections")
                detections_dir.mkdir(parents=True, exist_ok=True)
                
                original_frame_path = f"detections/original_{camera_id}_{timestamp_str}_{frame_id}.jpg"
                cropped_plate_path = f"detections/cropped_{camera_id}_{timestamp_str}_{frame_id}.jpg"
                
                # Save original frame and cropped plate with error handling
                try:
                    if self.config.save_cropped_plates:
                        # Offload disk I/O to thread
                        await asyncio.to_thread(cv2.imwrite, original_frame_path, frame)
                        await asyncio.to_thread(cv2.imwrite, cropped_plate_path, plate_region)
                except Exception as e:
                    logger.error(f"Failed to save image files: {e}")
                    # Continue processing even if saving fails
                
                # Use the combined confidence from both detection and OCR
                combined_confidence = (detection['confidence'] + ocr_confidence) / 2.0
                
                # Create detection result
                result = DetectionResult(
                    frame_id=frame_id,
                    plate_text=plate_text,
                    confidence=combined_confidence,
                    bbox=(x1, y1, x2, y2),
                    original_frame_path=original_frame_path,
                    cropped_plate_path=cropped_plate_path,
                    timestamp=current_timestamp,
                    camera_id=camera_id,
                    processed_frame=processed_frame
                )
                
                # Track performance
                processing_time = time.time() - start_time
                self.frame_processing_times.append(processing_time)
                self.detection_count += 1
                
                logger.info(f"Detected plate: {plate_text} with confidence {result.confidence:.2f}")
                
                return result
                
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during frame processing, clearing cache and continuing")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(f"Error in frame processing pipeline: {e}")
            return None
        
        return None

    async def _store_detection_result(self, result: DetectionResult):
        """Store detection result in the database."""
        def _do_insert(session):
            """Helper to insert detection with all required fields."""
            insert_detection(
                db_session=session,
                plate_text=result.plate_text,
                confidence=result.confidence,
                timestamp=result.timestamp.isoformat(),
                source_type='camera',
                source_identifier=result.camera_id,
                frame_number=result.frame_id,
                bbox_x=result.bbox[0],
                bbox_y=result.bbox[1],
                bbox_width=result.bbox[2] - result.bbox[0],
                bbox_height=result.bbox[3] - result.bbox[1],
                snapshot_path=result.cropped_plate_path,
                raw_frame_path=result.original_frame_path,
                job_id=self.current_job_id,
                camera_id=result.camera_id,
                bbox_x1=result.bbox[0],
                bbox_y1=result.bbox[1],
                bbox_x2=result.bbox[2],
                bbox_y2=result.bbox[3]
            )
        
        try:
            with self.db_manager.get_session() as session:
                _do_insert(session)
            logger.debug(f"Detection saved to database: {result.plate_text}")
            
        except Exception as e:
            logger.error(f"Failed to store detection result: {e}")
            try:
                if not self.db_manager.test_connection():
                    logger.warning("Database connection lost, attempting to reinitialize")
                    self.db_manager = DatabaseManager()
                    with self.db_manager.get_session() as session:
                        _do_insert(session)
                    logger.info("Detection saved after database recovery")
            except Exception as recovery_error:
                logger.error(f"Database recovery failed: {recovery_error}")
    
    def _check_and_manage_gpu_memory(self):
        """Check GPU memory usage and clear cache if needed."""
        if torch.cuda.is_available():
            try:
                # Get current GPU memory usage
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                max_memory = torch.cuda.get_device_properties(0).total_memory
                
                allocated_ratio = allocated_memory / max_memory
                reserved_ratio = reserved_memory / max_memory
                
                # If memory usage exceeds threshold, clear cache
                if allocated_ratio > self.gpu_memory_threshold or reserved_ratio > self.gpu_memory_threshold:
                    logger.warning(f"GPU memory usage high: {allocated_ratio:.2%} allocated, {reserved_ratio:.2%} reserved. Clearing cache.")
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error checking GPU memory: {e}")
    
    def _is_duplicate_detection(self, plate_text: str, timestamp: datetime) -> bool:
        """Check if this plate detection is a duplicate within the time window."""
        if plate_text not in self.duplicate_cache:
            return False
        
        # Get the last detection time for this plate
        last_detection_time = self.duplicate_cache[plate_text]
        
        # Check if it's within the duplicate detection window
        time_diff = abs((timestamp - last_detection_time).total_seconds())
        
        return time_diff < self.duplicate_window
    
    def _update_duplicate_cache(self, plate_text: str, timestamp: datetime):
        """Update the duplicate detection cache with the latest detection."""
        self.duplicate_cache[plate_text] = timestamp
        
        # Clean up old entries to prevent memory buildup
        current_time = datetime.now()
        expired_plates = [
            plate for plate, time in self.duplicate_cache.items()
            if (current_time - time).total_seconds() > self.duplicate_window * 2
        ]
        
        for plate in expired_plates:
            del self.duplicate_cache[plate]
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        avg_processing_time = (
            sum(self.frame_processing_times) / len(self.frame_processing_times)
            if self.frame_processing_times else 0
        )
        
        return {
            'status': self.status.value,
            'job_id': self.current_job_id,  # Match StreamStatusResponse field name
            'is_running': self._is_running,
            'detection_count': self.detection_count,
            'avg_frame_processing_time': avg_processing_time,
            'total_processed_frames': len(self.frame_processing_times),
            'frame_skip_interval': self.config.frame_skip_interval,
            'memory_usage': self.get_memory_metrics()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        if not self.frame_processing_times:
            return {}
        
        processing_times = list(self.frame_processing_times)
        total_time = sum(processing_times)
        return {
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'avg_processing_time': total_time / len(processing_times),
            'total_processing_time': total_time,
            'detections_per_second': self.detection_count / total_time if total_time > 0 else 0,
            'total_detections': self.detection_count
        }
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get current memory usage metrics."""
        if hasattr(self, 'memory_manager'):
            metrics = self.memory_manager.get_memory_metrics()
            return {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'memory_available_gb': metrics.memory_available_gb,
                'memory_used_gb': metrics.memory_used_gb,
                'gpu_memory_allocated_gb': metrics.gpu_memory_allocated_gb,
                'gpu_memory_reserved_gb': metrics.gpu_memory_reserved_gb,
                'gpu_utilization_percent': metrics.gpu_utilization_percent
            }
        else:
            return {}
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        status = self.get_pipeline_status()
        perf_metrics = self.get_performance_metrics()
        memory_metrics = self.get_memory_metrics()
        
        # Determine health status based on various metrics
        health_status = "healthy"
        issues = []
        
        # Check if processing is too slow
        if perf_metrics.get('avg_processing_time', 0) > 1.0:  # More than 1 second per frame
            health_status = "degraded"
            issues.append("Slow processing speed")
        
        # Check if memory usage is high
        if memory_metrics.get('memory_percent', 0) > 80:
            health_status = "degraded" if health_status == "healthy" else "unhealthy"
            issues.append("High memory usage")
        
        if (memory_metrics.get('gpu_utilization_percent', 0) or 0) > 80:
            health_status = "degraded" if health_status == "healthy" else "unhealthy"
            issues.append("High GPU memory usage")
        
        # Check if detection rate is too low
        if perf_metrics.get('detections_per_second', 0) < 0.1:  # Less than 0.1 detections per second
            issues.append("Low detection rate")
        
        return {
            'status': health_status,
            'issues': issues,
            'timestamp': datetime.now().isoformat(),
            'pipeline_status': status,
            'performance_metrics': perf_metrics,
            'memory_metrics': memory_metrics
        }
    
    def get_job_progress(self) -> Dict[str, Any]:
        """Get job progress information."""
        return {
            'current_job_id': self.current_job_id,
            'status': self.status.value,
            'detections_count': self.detection_count,
            'frames_processed': len(self.frame_processing_times),
            'avg_processing_time': sum(self.frame_processing_times) / len(self.frame_processing_times) if self.frame_processing_times else 0,
            'estimated_completion': None  # Would need total frames to calculate
        }
    
    def update_config(self, new_config: PipelineConfig):
        """Update pipeline configuration."""
        self.config = new_config
        logger.info("Pipeline configuration updated")
        
        # Update related parameters
        self.frame_skip_interval = self.config.frame_skip_interval
        self.duplicate_window = self.config.duplicate_detection_window
        self.processing_batch_size = self.config.batch_size

    async def _trigger_alerts_if_needed(self, result: DetectionResult):
        """Trigger alerts based on detection results."""
        if not self.alert_manager:
            logger.warning("AlertManager not initialized, skipping alert triggering")
            return
        
        try:
            # Check if an alert should be triggered and send it
            self.alert_manager.check_and_trigger(
                plate_text=result.plate_text,
                camera_id=result.camera_id,
                confidence=result.confidence
            )
            
            logger.info(f"Alert triggered for plate: {result.plate_text} from camera {result.camera_id}")
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    async def pause_pipeline(self):
        """Pause the current pipeline execution."""
        self.status = PipelineStatus.PAUSED
        logger.info("Pipeline paused")

    async def resume_pipeline(self):
        """Resume the paused pipeline execution."""
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.PROCESSING
            logger.info("Pipeline resumed")

    async def stop_pipeline(self):
        """Stop the current pipeline execution."""
        self._is_running = False
        self.status = PipelineStatus.IDLE
        logger.info("Pipeline stopped")

    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        logger.info("Cleaning up pipeline resources")
        
        if self._is_running:
            await self.stop_pipeline()
        
        # Close components that have cleanup methods
        if self.yolo_detector:
            try:
                self.yolo_detector.cleanup()  # Releases GPU model memory
            except Exception as e:
                logger.warning(f"Error cleaning up YOLO detector: {e}")
        
        if self.ocr_extractor:
            try:
                self.ocr_extractor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up OCR extractor: {e}")
        
        if self.video_handler:
            try:
                self.video_handler.cleanup()  # Kills any FFmpeg processes
            except Exception as e:
                logger.warning(f"Error cleaning up video handler: {e}")
        
        if self.db_manager:
            try:
                self.db_manager.close()
            except Exception as e:
                logger.warning(f"Error closing database manager: {e}")
        
        # Clear CUDA cache (redundant if yolo_detector.cleanup() called, but safe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear performance metrics
        self.frame_processing_times.clear()
        self.detection_count = 0
        self.current_job_id = None
        
        logger.info("Pipeline cleanup complete")

    async def process_single_image(self, image_path: str, camera_id: str) -> List[DetectionResult]:
        """
        Process a single image file through the pipeline.
        
        Args:
            image_path: Path to the image file
            camera_id: Camera identifier for the detection
            
        Returns:
            List of detection results
        """
        if not all([self.video_handler, self.yolo_detector, self.ocr_extractor]):
            await self.initialize_components()
        
        try:
            # Load image (offload to thread to avoid blocking)
            frame = await asyncio.to_thread(cv2.imread, image_path)
            if frame is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Process single frame
            result = await self._process_single_frame(frame, 0, camera_id)
            
            if result:
                await self._store_detection_result(result)
                await self._trigger_alerts_if_needed(result)
                return [result]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error processing single image: {e}")
            raise
