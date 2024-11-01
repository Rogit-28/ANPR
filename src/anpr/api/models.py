from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class SuccessResponse(BaseModel):
    """
    Standard success response format with status, data, and metadata.
    """
    status: str = "success"
    data: Any = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """
    Standard error response format with status, error code, message, and details.
    """
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StreamStartRequest(BaseModel):
    """Request model for starting live stream processing."""
    video_source: str = Field(..., description="URL or path to the video stream")
    camera_id: str = Field(..., description="Unique identifier for the camera")
    job_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier")


class StreamStartResponse(BaseModel):
    """Response model for starting live stream processing."""
    job_id: str
    status: str
    message: str


class StreamStatusResponse(BaseModel):
    """Response model for stream processing status."""
    job_id: Optional[str]
    status: str
    is_running: bool
    detection_count: int
    avg_frame_processing_time: float
    total_processed_frames: int
    memory_usage: Optional[Dict[str, Any]]


class StreamStopResponse(BaseModel):
    """Response model for stopping stream processing."""
    job_id: Optional[str]
    status: str
    message: str


class VideoProcessRequest(BaseModel):
    """Request model for video processing."""
    video_path: str = Field(..., description="Path to the video file to process")
    camera_id: str = Field(..., description="Camera identifier for the video source")
    job_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier")


class VideoProcessResponse(BaseModel):
    """Response model for video processing request."""
    job_id: str
    status: str
    message: str


class VideoStatusResponse(BaseModel):
    """Response model for video processing status."""
    job_id: str
    status: str
    progress: Optional[Dict[str, Any]]
    detection_count: int
    completed: bool


class DetectionFilter(BaseModel):
    """Filter model for querying detections."""
    plate_text: Optional[str] = None
    source_identifier: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    min_confidence: Optional[float] = None
    limit: Optional[int] = Field(default=100, ge=1, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)


class DetectionResponse(BaseModel):
    """Response model for a single detection."""
    id: int
    plate_text: str
    confidence: float
    timestamp: str
    source_type: str
    source_identifier: str
    frame_number: Optional[int]
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    snapshot_path: str
    raw_frame_path: Optional[str]
    lighting_condition: Optional[str]
    processing_time_ms: int
    job_id: Optional[str]
    camera_id: Optional[str]
    bbox_x1: Optional[int]
    bbox_y1: Optional[int]
    bbox_x2: Optional[int]
    bbox_y2: Optional[int]


class DetectionsQueryResponse(BaseModel):
    """Response model for detections query."""
    total: int
    detections: List[DetectionResponse]
    filters_applied: DetectionFilter


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    uptime: Optional[str]
    version: str = "1.0.0"
    gpu_available: Optional[bool] = None
    gpu_memory_used: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    gpu_memory_percent: Optional[float] = None  # Memory usage percentage (0-100)
    gpu_utilization: Optional[float] = None     # GPU compute utilization percentage (0-100)
    database_connected: Optional[bool] = None


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    total_detections: int
    average_confidence: float
    earliest_detection: Optional[str]
    latest_detection: Optional[str]
    pipeline_stats: Optional[Dict[str, Any]]
    memory_usage: Optional[Dict[str, Any]]


# ============================================
# Job Queue Models (for frontend support)
# ============================================

class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    UPLOADING = "uploading"
    NORMALIZING = "normalizing"
    PROCESSING = "processing"
    ANNOTATING = "annotating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadResponse(BaseModel):
    """Response model for video upload."""
    job_id: str
    status: str
    message: str
    filename: str


class JobResponse(BaseModel):
    """Response model for a processing job."""
    id: str
    status: str
    progress: int
    original_filename: str
    original_path: Optional[str]
    normalized_path: Optional[str]
    annotated_path: Optional[str]
    duration_seconds: Optional[float]
    frame_count: Optional[int]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    detection_count: int
    frames_processed: int
    processing_time_ms: Optional[int]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    camera_id: Optional[str]
    source_type: Optional[str] = "video"  # 'video' or 'image'


class JobListResponse(BaseModel):
    """Response model for job list."""
    total: int
    jobs: List[JobResponse]
    limit: int
    offset: int


class JobDetectionsResponse(BaseModel):
    """Response model for job detections."""
    job_id: str
    total: int
    detections: List[DetectionResponse]


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""
    is_processing: bool
    current_job_id: Optional[str]
    queue_size: int
    total_jobs: int


# ============================================
# Watchlist Models
# ============================================

class WatchlistRequest(BaseModel):
    """Request model for watchlist operations."""
    plate_text: str = Field(..., description="License plate text")
    description: Optional[str] = None


class WatchlistResponse(BaseModel):
    """Response model for watchlist operations."""
    plate_text: str
    added_at: str
    description: Optional[str] = None


class WatchlistCheckResponse(BaseModel):
    """Response model for checking if a plate is in the watchlist."""
    plate_text: str
    in_watchlist: bool


# ============================================
# Alert Models
# ============================================

class AlertStatsResponse(BaseModel):
    """Response model for alert statistics."""
    enabled: bool
    total_alerts: int = 0
    unacknowledged: int = 0
    today: int = 0
    watchlist_count: int
    desktop_notifications_available: bool


class AlertEntry(BaseModel):
    """Model for a single alert history entry."""
    id: Optional[str] = None
    timestamp: str
    type: str
    plate: str
    camera: str
    confidence: float
    message: str
    metadata: Optional[Dict[str, Any]] = None
    acknowledged: bool = False


class AlertHistoryResponse(BaseModel):
    """Response model for alert history."""
    total: int
    alerts: List[AlertEntry]


class AlertToggleResponse(BaseModel):
    """Response model for enabling/disabling alerts."""
    enabled: bool
    message: str


# ============================================
# Image Processing Models
# ============================================

class ImageProcessResponse(BaseModel):
    """Response model for single image processing."""
    job_id: str
    detections: List[DetectionResponse]
    total: int
    processing_time_ms: int
    image_width: int
    image_height: int


# ============================================
# Detection Events Models (Grouped Detections)
# ============================================

class DetectionEvent(BaseModel):
    """
    Represents a grouped detection event.
    
    An event groups multiple detections of the same plate within a time window,
    returning only the highest-confidence detection as representative.
    """
    plate_text: str
    first_seen: str
    last_seen: str
    detection_count: int
    best_confidence: float
    best_detection_id: int
    best_snapshot_path: str
    best_frame_number: Optional[int]
    source_type: str
    source_identifier: str
    job_id: Optional[str]
    camera_id: Optional[str]
    first_frame: Optional[int]
    last_frame: Optional[int]
    total_frames: Optional[int]


class DetectionEventsResponse(BaseModel):
    """Response model for detection events query."""
    total: int
    events: List[DetectionEvent]
    time_window_seconds: int
    limit: int
    offset: int


class EventDetectionsResponse(BaseModel):
    """Response model for individual detections within an event."""
    plate_text: str
    first_seen: str
    last_seen: str
    total: int
    detections: List[DetectionResponse]


# ============================================
# Settings & Configuration Models
# ============================================

class SystemInfoResponse(BaseModel):
    """Extended system information for Settings page."""
    api_version: str
    python_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_total: Optional[str] = None
    gpu_memory_used: Optional[str] = None
    gpu_utilization: Optional[float] = None
    database_path: str
    config_file_path: str
    uptime: Optional[str] = None
    uptime_seconds: Optional[float] = None
    onnx_providers: List[str] = []


class ConfigResponse(BaseModel):
    """User-editable configuration values."""
    # Alerts
    alerts_enabled: bool
    desktop_notifications: bool
    # Detection
    yolo_confidence: float
    yolo_weights_path: str
    min_ocr_confidence: float
    plate_validation_enabled: bool


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    alerts_enabled: Optional[bool] = None
    desktop_notifications: Optional[bool] = None
    yolo_confidence: Optional[float] = Field(None, ge=0.1, le=1.0)
    yolo_weights_path: Optional[str] = None
    min_ocr_confidence: Optional[float] = Field(None, ge=0.1, le=1.0)
    plate_validation_enabled: Optional[bool] = None


class ConfigUpdateResponse(BaseModel):
    """Response after config update."""
    success: bool
    message: str
    restart_required: bool
    updated_fields: List[str] = []
    validation_errors: Optional[List[str]] = None


class ConfigResetResponse(BaseModel):
    """Response after config reset."""
    success: bool
    message: str
    restart_required: bool


class TestAlertResponse(BaseModel):
    """Response after triggering a test alert."""
    success: bool
    message: str
    alert_id: Optional[str] = None


class ServerRestartResponse(BaseModel):
    """Response after triggering server restart."""
    success: bool
    message: str
    restart_scheduled: bool


class WatchlistClearResponse(BaseModel):
    """Response after clearing watchlist."""
    success: bool
    message: str
    cleared_count: int
