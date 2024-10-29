"""Job Queue for async ANPR video processing with status tracking and progress updates."""

import asyncio
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from ..database.models import ProcessingJob, JobStatus
from ..database.manager import DatabaseManager
from .video_handler import normalize_video, normalize_video_async, get_video_metadata

logger = logging.getLogger(__name__)


@dataclass
class JobQueueConfig:
    """Configuration for job queue."""
    data_dir: str = "data"
    uploads_dir: str = "uploads"
    normalized_dir: str = "normalized"
    annotated_dir: str = "annotated"
    max_upload_size_mb: int = 500
    allowed_extensions: set = field(default_factory=lambda: {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'})


class JobQueue:
    """
    Async job queue processing one video at a time through stages:
    PENDING -> NORMALIZING -> PROCESSING -> ANNOTATING -> COMPLETED
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[JobQueueConfig] = None):
        self.db_manager = db_manager
        self.config = config or JobQueueConfig()
        
        self._current_job_id: Optional[str] = None
        self._is_processing: bool = False
        self._should_stop: bool = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._progress_callbacks: Dict[str, Callable] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self.pipeline_manager = None
        self.video_annotator = None
        
        self._ensure_directories()
        logger.info("JobQueue initialized")
    
    def _ensure_directories(self):
        """Create necessary directories for job storage."""
        base_path = Path(self.config.data_dir)
        
        dirs = [
            base_path / self.config.uploads_dir,
            base_path / self.config.normalized_dir,
            base_path / self.config.annotated_dir,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def _get_job_dir(self, job_id: str, stage: str) -> Path:
        """Get directory path for a job at a specific stage."""
        base_path = Path(self.config.data_dir)
        
        if stage == "upload":
            return base_path / self.config.uploads_dir / job_id
        elif stage == "normalized":
            return base_path / self.config.normalized_dir / job_id
        elif stage == "annotated":
            return base_path / self.config.annotated_dir / job_id
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    async def create_job(self, original_filename: str, camera_id: Optional[str] = None, source_type: str = "video") -> ProcessingJob:
        """Create a new processing job and return detached instance.
        
        Args:
            original_filename: Name of the uploaded file
            camera_id: Optional camera identifier
            source_type: Type of source - 'video' or 'image'
        """
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        job = ProcessingJob(
            id=job_id,
            status=JobStatus.PENDING.value,
            progress=0,
            original_filename=original_filename,
            created_at=now,
            camera_id=camera_id,
            source_type=source_type
        )
        
        with self.db_manager.get_session() as session:
            session.add(job)
            session.commit()
            session.refresh(job)
            from sqlalchemy.orm import make_transient
            session.expunge(job)
            make_transient(job)
        
        for stage in ["upload", "normalized", "annotated"]:
            job_dir = self._get_job_dir(job_id, stage)
            job_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created job {job_id} for file: {original_filename}")
        return job
    
    async def save_upload(self, job_id: str, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk, validating extension and size."""
        ext = Path(filename).suffix.lower()
        if ext not in self.config.allowed_extensions:
            raise ValueError(f"File extension {ext} not allowed. Allowed: {self.config.allowed_extensions}")
        
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > self.config.max_upload_size_mb:
            raise ValueError(f"File size {size_mb:.1f}MB exceeds maximum {self.config.max_upload_size_mb}MB")
        
        job_dir = self._get_job_dir(job_id, "upload")
        file_path = job_dir / f"original{ext}"
        
        def write_file():
            with open(file_path, 'wb') as f:
                f.write(file_content)
        
        await asyncio.to_thread(write_file)
        await self._update_job(job_id, original_path=str(file_path))
        
        logger.info(f"Saved upload for job {job_id}: {file_path} ({size_mb:.1f}MB)")
        return str(file_path)
    
    async def enqueue_job(self, job_id: str):
        """Add job to queue and start worker if needed."""
        await self._queue.put(job_id)
        logger.info(f"Job {job_id} added to queue. Queue size: {self._queue.qsize()}")
        
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Worker that processes jobs from the queue."""
        logger.info("Job queue worker started")
        
        while True:
            try:
                # Get next job (blocks until available)
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                if self._should_stop:
                    # Put job back and exit
                    await self._queue.put(job_id)
                    break
                
                # Process the job
                self._current_job_id = job_id
                self._is_processing = True
                
                try:
                    await self._process_job(job_id)
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {e}")
                    await self._update_job(
                        job_id,
                        status=JobStatus.FAILED.value,
                        error_message=str(e)
                    )
                finally:
                    self._current_job_id = None
                    self._is_processing = False
                    self._queue.task_done()
                    # Clean up progress callback to prevent memory leak
                    self.unregister_progress_callback(job_id)
                    
                    # Clear GPU cache after job completion (keeps model loaded but frees intermediates)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug(f"GPU cache cleared after job {job_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clear GPU cache: {e}")
                    
            except asyncio.TimeoutError:
                # No job in queue, check if we should stop
                if self._should_stop:
                    break
                continue
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Job queue worker stopped")
    
    async def _process_job(self, job_id: str):
        """
        Process a single job through all stages.
        
        Stages:
        1. NORMALIZING - Convert video to standard format
        2. PROCESSING - Run ANPR pipeline
        3. ANNOTATING - Create annotated video with overlays
        4. COMPLETED - All done
        """
        logger.info(f"Starting processing for job {job_id}")
        
        # Get job from database
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if not job.original_path or not os.path.exists(job.original_path):
            raise ValueError(f"Original file not found for job {job_id}")
        
        # Update start time
        await self._update_job(job_id, started_at=datetime.now().isoformat())
        
        # Stage 1: Extract metadata from original video (no transcoding)
        await self._stage_extract_metadata(job_id, job.original_path)
        
        # Stage 2: Run ANPR processing on original video
        await self._stage_process(job_id)
        
        # Stage 3: Create annotated video
        await self._stage_annotate(job_id)
        
        # Mark as completed
        await self._update_job(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            completed_at=datetime.now().isoformat()
        )
        
        logger.info(f"Job {job_id} completed successfully")
    
    async def _stage_extract_metadata(self, job_id: str, input_path: str):
        """Stage 1: Extract video metadata without transcoding.
        
        This replaces the old normalization stage. We now process videos at their
        original resolution (capped at 2K) and frame rate for better detection accuracy.
        """
        logger.info(f"Job {job_id}: Extracting metadata from original video")
        
        await self._update_job(job_id, status=JobStatus.NORMALIZING.value, progress=5)
        
        # Get metadata from original video
        metadata = await asyncio.to_thread(get_video_metadata, input_path)
        
        if not metadata:
            raise RuntimeError(f"Could not read video metadata from {input_path}")
        
        # Extract dimensions and apply 2K cap for reporting
        width, height = metadata['width'], metadata['height']
        fps = metadata.get('fps', 24)
        duration = metadata.get('duration', 0)
        frame_count = metadata.get('frame_count', 0)
        
        # Apply 2K resolution cap if needed (for metadata reporting)
        MAX_RES = (2560, 1440)
        if width > MAX_RES[0] or height > MAX_RES[1]:
            scale = min(MAX_RES[0] / width, MAX_RES[1] / height)
            width = int(width * scale)
            height = int(height * scale)
            # Round to even numbers (required by many codecs)
            width = width - (width % 2)
            height = height - (height % 2)
            logger.info(f"Job {job_id}: Video exceeds 2K cap, will process at {width}x{height}")
        
        # Recalculate frame count if needed (using original fps)
        if frame_count == 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)
        
        # Update job with video info
        # Note: We don't set normalized_path since we're using original
        await self._update_job(
            job_id,
            duration_seconds=duration,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            progress=10
        )
        
        logger.info(f"Job {job_id}: Metadata extracted - {width}x{height} @ {fps:.2f}fps, {duration:.1f}s, ~{frame_count} frames")
    
    async def _stage_process(self, job_id: str):
        """Stage 2: Run ANPR processing pipeline on original video."""
        logger.info(f"Job {job_id}: Starting ANPR processing")
        
        await self._update_job(job_id, status=JobStatus.PROCESSING.value, progress=15)
        
        job = await self.get_job(job_id)
        if not job or not job.original_path:
            raise ValueError("Original video path not found")
        
        if self.pipeline_manager is None:
            raise RuntimeError("Pipeline manager not configured")
        
        # Run pipeline on original video (not normalized)
        # The VideoHandler will handle resolution capping internally
        try:
            logger.info(f"Job {job_id}: Starting pipeline on original video...")
            await self.pipeline_manager.start_pipeline(
                video_source=job.original_path,  # Use original, not normalized
                camera_id=job.camera_id or "upload",
                job_id=job_id
            )
            logger.info(f"Job {job_id}: Pipeline start_pipeline() returned. Status: {self.pipeline_manager.status.value}")
            
            # Wait for pipeline to complete (should already be completed when start_pipeline returns)
            wait_count = 0
            while self.pipeline_manager.status.value not in ['idle', 'error', 'completed']:
                wait_count += 1
                if wait_count % 20 == 0:  # Log every 10 seconds
                    logger.warning(f"Job {job_id}: Still waiting for pipeline completion. Status: {self.pipeline_manager.status.value}, waited {wait_count * 0.5}s")
                await asyncio.sleep(0.5)
                
                # Update progress based on pipeline progress
                pipeline_progress = self.pipeline_manager.get_job_progress()
                frames_processed = pipeline_progress.get('frames_processed', 0)
                total_frames = job.frame_count or 1
                
                # Map to 25-80% range
                stage_progress = min(25 + int((frames_processed / total_frames) * 55), 80)
                
                await self._update_job(
                    job_id,
                    progress=stage_progress,
                    frames_processed=frames_processed,
                    detection_count=pipeline_progress.get('detections_count', 0)
                )
        
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            raise
        
        # Get final detection count
        with self.db_manager.get_session() as session:
            from sqlalchemy import func
            from ..database.models import Detection
            detection_count = session.query(func.count(Detection.id)).filter(
                Detection.job_id == job_id
            ).scalar() or 0
        
        await self._update_job(
            job_id,
            progress=80,
            detection_count=detection_count
        )
        
        logger.info(f"Job {job_id}: ANPR processing complete. {detection_count} detections found.")
    
    async def _stage_annotate(self, job_id: str):
        """Stage 3: Create annotated video with detection overlays."""
        logger.info(f"Job {job_id}: Starting annotation")
        
        await self._update_job(job_id, status=JobStatus.ANNOTATING.value, progress=85)
        
        job = await self.get_job(job_id)
        if not job or not job.original_path:
            raise ValueError("Original video path not found")
        
        if self.video_annotator is None:
            logger.warning("Video annotator not configured, skipping annotation")
            await self._update_job(job_id, progress=95)
            return
        
        # Get output path
        output_dir = self._get_job_dir(job_id, "annotated")
        output_path = output_dir / "annotated.mp4"
        
        # Get detections for this job
        with self.db_manager.get_session() as session:
            from ..database.models import Detection
            detections = session.query(Detection).filter(
                Detection.job_id == job_id
            ).order_by(Detection.frame_number).all()
            
            # Convert to list of dicts for annotator
            detection_data = []
            for det in detections:
                detection_data.append({
                    'frame_number': det.frame_number,
                    'plate_text': det.plate_text,
                    'confidence': det.confidence,
                    'bbox': (det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2)
                })
        
        # Run annotation on original video (annotator reads dimensions from input)
        result = await self.video_annotator.annotate_video_async(
            job.original_path,  # Use original video, not normalized
            str(output_path),
            detection_data
        )
        
        if result.get('success'):
            await self._update_job(
                job_id,
                annotated_path=str(output_path),
                progress=95
            )
            logger.info(f"Job {job_id}: Annotation complete")
        else:
            logger.warning(f"Job {job_id}: Annotation failed: {result.get('error')}")
    
    async def _update_job(self, job_id: str, **kwargs):
        """Update job fields in database."""
        with self.db_manager.get_session() as session:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                session.commit()
        
        # Notify progress callbacks
        if 'progress' in kwargs and job_id in self._progress_callbacks:
            try:
                self._progress_callbacks[job_id](kwargs['progress'])
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID."""
        with self.db_manager.get_session() as session:
            job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                # Refresh to ensure all attributes are loaded, then detach
                session.refresh(job)
                from sqlalchemy.orm import make_transient
                session.expunge(job)
                make_transient(job)
            return job
    
    async def get_jobs(self, limit: int = 10, offset: int = 0) -> List[ProcessingJob]:
        """Get list of jobs, ordered by creation time (newest first)."""
        with self.db_manager.get_session() as session:
            jobs = session.query(ProcessingJob).order_by(
                ProcessingJob.created_at.desc()
            ).limit(limit).offset(offset).all()
            
            # Refresh and detach each job to avoid DetachedInstanceError
            from sqlalchemy.orm import make_transient
            for job in jobs:
                session.refresh(job)
                session.expunge(job)
                make_transient(job)
            
            return jobs
    
    async def get_job_count(self) -> int:
        """Get total number of jobs."""
        with self.db_manager.get_session() as session:
            from sqlalchemy import func
            return session.query(func.count(ProcessingJob.id)).scalar() or 0
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED.value, JobStatus.CANCELLED.value]:
            return False
        
        # If currently processing, stop the pipeline
        if self._current_job_id == job_id and self.pipeline_manager:
            await self.pipeline_manager.stop_pipeline()
        
        await self._update_job(
            job_id,
            status=JobStatus.CANCELLED.value,
            completed_at=datetime.now().isoformat()
        )
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        # Cancel if still running
        if job.status in [JobStatus.PENDING.value, JobStatus.NORMALIZING.value, 
                          JobStatus.PROCESSING.value, JobStatus.ANNOTATING.value]:
            await self.cancel_job(job_id)
        
        # Delete files
        for stage in ["upload", "normalized", "annotated"]:
            try:
                job_dir = self._get_job_dir(job_id, stage)
                if job_dir.exists():
                    shutil.rmtree(job_dir)
            except Exception as e:
                logger.warning(f"Failed to delete {stage} dir for job {job_id}: {e}")
        
        # Delete database records
        with self.db_manager.get_session() as session:
            # Delete detections
            from ..database.models import Detection
            session.query(Detection).filter(Detection.job_id == job_id).delete()
            
            # Delete job
            session.query(ProcessingJob).filter(ProcessingJob.id == job_id).delete()
            session.commit()
        
        logger.info(f"Job {job_id} deleted")
        return True
    
    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for progress updates."""
        self._progress_callbacks[job_id] = callback
    
    def unregister_progress_callback(self, job_id: str):
        """Remove a progress callback."""
        self._progress_callbacks.pop(job_id, None)
    
    @property
    def current_job_id(self) -> Optional[str]:
        """Get ID of currently processing job."""
        return self._current_job_id
    
    @property
    def is_processing(self) -> bool:
        """Check if currently processing a job."""
        return self._is_processing
    
    @property
    def queue_size(self) -> int:
        """Get number of jobs waiting in queue."""
        return self._queue.qsize()
    
    async def shutdown(self):
        """Gracefully shutdown the job queue."""
        logger.info("Shutting down job queue")
        self._should_stop = True
        
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Job queue shutdown complete")


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> Optional[JobQueue]:
    """Get global job queue instance."""
    return _job_queue


def init_job_queue(db_manager: DatabaseManager, config: Optional[JobQueueConfig] = None) -> JobQueue:
    """Initialize global job queue."""
    global _job_queue
    _job_queue = JobQueue(db_manager, config)
    return _job_queue
