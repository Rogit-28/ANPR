from sqlalchemy import Column, Integer, String, Float, Text, Boolean, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum

Base = declarative_base()


class JobStatus(enum.Enum):
    """Enum for processing job status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    NORMALIZING = "normalizing"
    PROCESSING = "processing"
    ANNOTATING = "annotating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJob(Base):
    """
    SQLAlchemy model for the processing_jobs table.
    
    Tracks video upload → process → annotate jobs for the frontend.
    """
    __tablename__ = 'processing_jobs'

    # Primary key
    id = Column(Text, primary_key=True)  # UUID

    # Job status
    status = Column(Text, nullable=False, default=JobStatus.PENDING.value)
    
    # Progress tracking (0-100)
    progress = Column(Integer, nullable=False, default=0)
    
    # File paths
    original_filename = Column(Text, nullable=False)
    original_path = Column(Text, nullable=True)  # Path to uploaded file
    normalized_path = Column(Text, nullable=True)  # Path to normalized video
    annotated_path = Column(Text, nullable=True)  # Path to annotated video
    
    # Video metadata
    duration_seconds = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    
    # Processing stats
    detection_count = Column(Integer, nullable=False, default=0)
    frames_processed = Column(Integer, nullable=False, default=0)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Timestamps (stored as ISO format strings)
    created_at = Column(Text, nullable=False)
    started_at = Column(Text, nullable=True)
    completed_at = Column(Text, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    # Camera identifier (optional)
    camera_id = Column(Text, nullable=True)
    
    # Source type: 'video' or 'image'
    source_type = Column(Text, nullable=True, default='video')

    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_created_at', 'created_at'),
        Index('idx_job_source_type', 'source_type'),
    )


class Detection(Base):
    """
    SQLAlchemy model for the detections table.
    
    Represents an ANPR detection record with all required fields.
    """
    __tablename__ = 'detections'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Plate information
    plate_text = Column(Text, nullable=False)

    # Confidence score
    confidence = Column(Float, nullable=False)

    # Timestamp of detection
    timestamp = Column(Text, nullable=False)

    # Source information
    source_type = Column(Text, nullable=False)
    source_identifier = Column(Text, nullable=False)

    # Frame information
    frame_number = Column(Integer, nullable=True)

    # Bounding box coordinates
    bbox_x = Column(Integer, nullable=False)
    bbox_y = Column(Integer, nullable=False)
    bbox_width = Column(Integer, nullable=False)
    bbox_height = Column(Integer, nullable=False)

    # File paths
    snapshot_path = Column(Text, nullable=False)
    raw_frame_path = Column(Text, nullable=True)

    # Additional metadata
    lighting_condition = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=False)
    
    # Additional fields for enhanced metadata
    job_id = Column(Text, nullable=True)  # Job ID for batch processing
    camera_id = Column(Text, nullable=True)  # Camera identifier
    bbox_x1 = Column(Integer, nullable=True)  # Top-left x coordinate
    bbox_y1 = Column(Integer, nullable=True)  # Top-left y coordinate
    bbox_x2 = Column(Integer, nullable=True)  # Bottom-right x coordinate
    bbox_y2 = Column(Integer, nullable=True)  # Bottom-right y coordinate

    __table_args__ = (
        Index('idx_plate_text', 'plate_text'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_source_identifier', 'source_identifier'),
        Index('idx_detection_job_id', 'job_id'),
        Index('idx_detection_job_frame', 'job_id', 'frame_number'),
        Index('idx_detection_confidence', 'confidence'),
    )


class AlertRecord(Base):
    """
    SQLAlchemy model for storing alert history in the database.
    
    Provides persistent storage for alerts so they can be queried
    even after server restart (unlike the current file-based approach).
    """
    __tablename__ = 'alert_records'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Alert identification (UUID for uniqueness)
    alert_id = Column(Text, nullable=False, unique=True)
    
    # Alert type (watchlist_match, new_plate, repeated_plate, etc.)
    alert_type = Column(Text, nullable=False)
    
    # Plate information
    plate_text = Column(Text, nullable=False)
    
    # Source information
    camera_id = Column(Text, nullable=True)
    
    # Detection confidence
    confidence = Column(Float, nullable=True)
    
    # Alert message
    message = Column(Text, nullable=True)
    
    # JSON metadata (stored as text)
    metadata_json = Column(Text, nullable=True)
    
    # Timestamp when alert was triggered
    timestamp = Column(Text, nullable=False)
    
    # Whether alert has been acknowledged/dismissed by user
    acknowledged = Column(Boolean, default=False)
    
    # When alert was acknowledged
    acknowledged_at = Column(Text, nullable=True)
    
    # Record creation timestamp
    created_at = Column(Text, nullable=False)
    
    __table_args__ = (
        Index('idx_alert_type', 'alert_type'),
        Index('idx_alert_timestamp', 'timestamp'),
        Index('idx_alert_plate', 'plate_text'),
        Index('idx_alert_acknowledged', 'acknowledged'),
        Index('idx_alert_camera', 'camera_id'),
    )


class WatchlistEntry(Base):
    """
    SQLAlchemy model for the watchlist table.
    
    Stores plates that should trigger high-priority alerts when detected.
    Replaces the in-memory/file-based watchlist with persistent storage.
    """
    __tablename__ = 'watchlist'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Plate text (normalized: uppercase, no spaces)
    plate_text = Column(Text, nullable=False, unique=True)
    
    # Description/reason for watchlist (e.g., "Stolen vehicle", "VIP")
    description = Column(Text, nullable=True)
    
    # Priority level: low, normal, high, critical
    priority = Column(Text, nullable=False, default='normal')
    
    # Whether entry is active
    is_active = Column(Boolean, default=True)
    
    # When entry was added
    created_at = Column(Text, nullable=False)
    
    # Who added the entry (optional)
    created_by = Column(Text, nullable=True)
    
    # When entry was last updated
    updated_at = Column(Text, nullable=True)
    
    # Optional notes
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_watchlist_plate', 'plate_text'),
        Index('idx_watchlist_active', 'is_active'),
        Index('idx_watchlist_priority', 'priority'),
    )
