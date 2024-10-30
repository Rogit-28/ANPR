from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import os
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc

from .models import Detection
from .manager import DatabaseManager

logger = logging.getLogger(__name__)


def insert_detection(
    db_session: Session,
    plate_text: str,
    confidence: float,
    timestamp: str,
    source_type: str,
    source_identifier: str,
    bbox_x: int,
    bbox_y: int,
    bbox_width: int,
    bbox_height: int,
    snapshot_path: str,
    frame_number: Optional[int] = None,
    raw_frame_path: Optional[str] = None,
    lighting_condition: Optional[str] = None,
    processing_time_ms: int = 0,
    job_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    bbox_x1: Optional[int] = None,
    bbox_y1: Optional[int] = None,
    bbox_x2: Optional[int] = None,
    bbox_y2: Optional[int] = None
) -> Detection:
    """
    Insert a new detection record into the database.
    
    Args:
        db_session: Active database session
        plate_text: Recognized plate text
        confidence: Recognition confidence score
        timestamp: Detection timestamp
        source_type: Type of source (camera, video, etc.)
        source_identifier: Identifier for the source
        bbox_x, bbox_y, bbox_width, bbox_height: Bounding box coordinates
        snapshot_path: Path to the snapshot image
        frame_number: Video frame number (optional)
        raw_frame_path: Path to the raw frame (optional)
        lighting_condition: Lighting condition at time of detection (optional)
        processing_time_ms: Processing time in milliseconds
        job_id: Job identifier for batch processing (optional)
        camera_id: Camera identifier (optional)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2: Additional bounding box coordinates (optional)
    
    Returns:
        Detection: The newly created Detection object
    """
    detection = Detection(
        plate_text=plate_text,
        confidence=confidence,
        timestamp=timestamp,
        source_type=source_type,
        source_identifier=source_identifier,
        frame_number=frame_number,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        snapshot_path=snapshot_path,
        raw_frame_path=raw_frame_path,
        lighting_condition=lighting_condition,
        processing_time_ms=processing_time_ms,
        job_id=job_id,
        camera_id=camera_id,
        bbox_x1=bbox_x1,
        bbox_y1=bbox_y1,
        bbox_x2=bbox_x2,
        bbox_y2=bbox_y2
    )
    
    db_session.add(detection)
    db_session.flush()  # Flush to get the ID without committing
    
    return detection


def query_detections(
    db_session: Session,
    plate_text: Optional[str] = None,
    source_identifier: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[Detection]:
    """
    Query detections with various filters.
    
    Args:
        db_session: Active database session
        plate_text: Filter by plate text
        source_identifier: Filter by source identifier
        start_time: Filter by minimum timestamp
        end_time: Filter by maximum timestamp
        min_confidence: Filter by minimum confidence
        limit: Maximum number of results
        offset: Offset for pagination
    
    Returns:
        List[Detection]: List of matching Detection objects (sorted by timestamp DESC)
    """
    query = db_session.query(Detection)
    
    conditions = []
    
    if plate_text:
        conditions.append(Detection.plate_text == plate_text)
    
    if source_identifier:
        conditions.append(Detection.source_identifier == source_identifier)
    
    if start_time:
        conditions.append(Detection.timestamp >= start_time)
    
    if end_time:
        conditions.append(Detection.timestamp <= end_time)
    
    if min_confidence is not None:
        conditions.append(Detection.confidence >= min_confidence)
    
    if conditions:
        query = query.filter(and_(*conditions))
    
    # Sort by timestamp descending (most recent first)
    query = query.order_by(Detection.timestamp.desc())
    
    if offset is not None:
        query = query.offset(offset)
    
    if limit is not None:
        query = query.limit(limit)
    
    return query.all()


def get_detection_by_id(db_session: Session, detection_id: int) -> Optional[Detection]:
    return db_session.query(Detection).filter(Detection.id == detection_id).first()


def get_snapshot_by_detection_id(db_session: Session, detection_id: int) -> Optional[str]:
    """Return snapshot_path for detection or None if not found."""
    detection = get_detection_by_id(db_session, detection_id)
    if detection:
        return detection.snapshot_path
    return None


def get_detection_stats(db_session: Session) -> Dict[str, Any]:
    """Return total count, avg confidence, and timestamp range."""
    from sqlalchemy import func
    
    total_count = db_session.query(func.count(Detection.id)).scalar()
    avg_confidence = db_session.query(func.avg(Detection.confidence)).scalar()
    earliest_timestamp = db_session.query(func.min(Detection.timestamp)).scalar()
    latest_timestamp = db_session.query(func.max(Detection.timestamp)).scalar()
    
    stats = {
        'total_detections': total_count,
        'average_confidence': avg_confidence if avg_confidence else 0.0,
        'earliest_detection': earliest_timestamp,
        'latest_detection': latest_timestamp
    }
    
    return stats


def cleanup_orphaned_detections(
    db_session: Session,
    base_path: Optional[str] = None,
    dry_run: bool = False
) -> Tuple[int, List[int]]:
    """
    Remove detection records where the snapshot file no longer exists.
    
    Args:
        db_session: Active database session
        base_path: Base path to resolve relative snapshot paths (defaults to cwd)
        dry_run: If True, only count orphans without deleting
    
    Returns:
        Tuple of (count of orphaned records, list of deleted IDs)
    """
    if base_path is None:
        base_path = os.getcwd()
    
    detections = db_session.query(Detection).all()
    orphaned_ids = []
    
    for detection in detections:
        if detection.snapshot_path:
            # Resolve relative paths
            if os.path.isabs(detection.snapshot_path):
                full_path = detection.snapshot_path
            else:
                full_path = os.path.join(base_path, detection.snapshot_path)
            
            if not os.path.exists(full_path):
                orphaned_ids.append(detection.id)
                if not dry_run:
                    db_session.delete(detection)
    
    if not dry_run and orphaned_ids:
        db_session.commit()
        logger.info(f"Cleaned up {len(orphaned_ids)} orphaned detection records")
    
    return len(orphaned_ids), orphaned_ids


def delete_detection(db_session: Session, detection_id: int, delete_files: bool = True) -> bool:
    """
    Delete a detection record and optionally its associated files.
    
    Args:
        db_session: Active database session
        detection_id: ID of the detection to delete
        delete_files: If True, also delete snapshot and raw frame files
    
    Returns:
        True if deletion was successful, False if detection not found
    """
    detection = get_detection_by_id(db_session, detection_id)
    if not detection:
        return False
    
    files_to_delete = []
    
    if delete_files:
        if detection.snapshot_path:
            if os.path.isabs(detection.snapshot_path):
                files_to_delete.append(detection.snapshot_path)
            else:
                files_to_delete.append(os.path.join(os.getcwd(), detection.snapshot_path))
        
        if detection.raw_frame_path:
            if os.path.isabs(detection.raw_frame_path):
                files_to_delete.append(detection.raw_frame_path)
            else:
                files_to_delete.append(os.path.join(os.getcwd(), detection.raw_frame_path))
    
    # Delete the database record
    db_session.delete(detection)
    db_session.commit()
    
    # Delete associated files
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
    
    return True


def get_detection_events(
    db_session: Session,
    time_window_seconds: int = 30,
    plate_text: Optional[str] = None,
    source_identifier: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0
) -> Dict[str, Any]:
    """
    Group detections into events based on plate number and time proximity.
    
    An "event" is defined as multiple detections of the same plate within
    a specified time window. Returns only the highest-confidence detection
    per event, along with event metadata.
    
    Args:
        db_session: Active database session
        time_window_seconds: Time window in seconds to group detections (default: 30s)
        plate_text: Filter by plate text (partial match)
        source_identifier: Filter by source identifier
        start_time: Filter by minimum timestamp
        end_time: Filter by maximum timestamp
        min_confidence: Filter by minimum confidence
        limit: Maximum number of events to return
        offset: Offset for pagination
    
    Returns:
        Dict with 'events' list and 'total' count
    """
    # First, get all detections sorted by plate_text and timestamp
    query = db_session.query(Detection).order_by(
        Detection.plate_text,
        Detection.timestamp
    )
    
    conditions = []
    
    if plate_text:
        conditions.append(Detection.plate_text.ilike(f"%{plate_text}%"))
    
    if source_identifier:
        conditions.append(Detection.source_identifier == source_identifier)
    
    if start_time:
        conditions.append(Detection.timestamp >= start_time)
    
    if end_time:
        conditions.append(Detection.timestamp <= end_time)
    
    if min_confidence is not None:
        conditions.append(Detection.confidence >= min_confidence)
    
    if conditions:
        query = query.filter(and_(*conditions))
    
    all_detections = query.all()
    
    # Group detections into events
    events = []
    current_event = None
    
    for detection in all_detections:
        try:
            det_timestamp = datetime.fromisoformat(detection.timestamp.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            # Fallback for different timestamp formats
            try:
                det_timestamp = datetime.strptime(detection.timestamp, "%Y-%m-%d %H:%M:%S")
            except:
                det_timestamp = datetime.now()
        
        # Check if this detection belongs to the current event
        should_start_new_event = True
        
        if current_event is not None:
            same_plate = current_event['plate_text'] == detection.plate_text
            
            try:
                last_seen = datetime.fromisoformat(current_event['last_seen'].replace('Z', '+00:00'))
            except:
                try:
                    last_seen = datetime.strptime(current_event['last_seen'], "%Y-%m-%d %H:%M:%S")
                except:
                    last_seen = det_timestamp - timedelta(seconds=time_window_seconds + 1)
            
            time_diff = (det_timestamp - last_seen).total_seconds()
            within_window = time_diff <= time_window_seconds
            
            if same_plate and within_window:
                should_start_new_event = False
                # Update current event
                current_event['detection_count'] += 1
                current_event['last_seen'] = detection.timestamp
                current_event['detection_ids'].append(detection.id)
                
                # Track frame range for videos
                if detection.frame_number is not None:
                    if current_event['first_frame'] is None:
                        current_event['first_frame'] = detection.frame_number
                    current_event['last_frame'] = detection.frame_number
                
                # Update best detection if this one has higher confidence
                if detection.confidence > current_event['best_confidence']:
                    current_event['best_confidence'] = detection.confidence
                    current_event['best_detection_id'] = detection.id
                    current_event['best_snapshot_path'] = detection.snapshot_path
                    current_event['best_frame_number'] = detection.frame_number
        
        if should_start_new_event:
            # Save the previous event
            if current_event is not None:
                events.append(current_event)
            
            # Start a new event
            current_event = {
                'plate_text': detection.plate_text,
                'first_seen': detection.timestamp,
                'last_seen': detection.timestamp,
                'detection_count': 1,
                'best_confidence': detection.confidence,
                'best_detection_id': detection.id,
                'best_snapshot_path': detection.snapshot_path,
                'best_frame_number': detection.frame_number,
                'source_type': detection.source_type,
                'source_identifier': detection.source_identifier,
                'job_id': detection.job_id,
                'camera_id': detection.camera_id,
                'first_frame': detection.frame_number,
                'last_frame': detection.frame_number,
                'detection_ids': [detection.id]
            }
    
    # Don't forget the last event
    if current_event is not None:
        events.append(current_event)
    
    # Sort events by most recent first
    events.sort(key=lambda e: e['last_seen'], reverse=True)
    
    total_events = len(events)
    
    # Apply pagination
    paginated_events = events[offset:offset + limit] if limit else events[offset:]
    
    # Clean up detection_ids from response (we don't need to send all IDs)
    for event in paginated_events:
        # Keep only summary info
        event['total_frames'] = event['last_frame'] - event['first_frame'] + 1 if event['first_frame'] is not None and event['last_frame'] is not None else None
        del event['detection_ids']
    
    return {
        'events': paginated_events,
        'total': total_events,
        'time_window_seconds': time_window_seconds
    }


def get_event_detections(
    db_session: Session,
    plate_text: str,
    start_time: str,
    end_time: str,
    limit: int = 100
) -> List[Detection]:
    """
    Get all individual detections for a specific event.
    
    Used when user expands an event to see all frames/detections.
    
    Args:
        db_session: Active database session
        plate_text: The plate text for this event
        start_time: Event start time (first_seen)
        end_time: Event end time (last_seen)
        limit: Maximum number of detections to return
    
    Returns:
        List of Detection objects
    """
    query = db_session.query(Detection).filter(
        and_(
            Detection.plate_text == plate_text,
            Detection.timestamp >= start_time,
            Detection.timestamp <= end_time
        )
    ).order_by(Detection.timestamp).limit(limit)
    
    return query.all()
