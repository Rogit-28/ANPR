"""
General utility functions for the ANPR system.

This module provides common helper functions used across the application
including string manipulation, path handling, and data formatting utilities.
"""

import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union


def normalize_plate_text(text: str) -> str:
    """
    Normalize license plate text for consistent comparison.
    
    Removes spaces, converts to uppercase, and removes special characters.
    
    Args:
        text: Raw plate text
        
    Returns:
        Normalized plate text
    """
    if not text:
        return ""
    # Remove spaces and convert to uppercase
    normalized = text.upper().replace(' ', '').replace('-', '')
    # Remove any non-alphanumeric characters
    normalized = re.sub(r'[^A-Z0-9]', '', normalized)
    return normalized


def is_valid_indian_plate(text: str) -> bool:
    """
    Check if text matches Indian license plate format.
    
    Indian plates follow patterns like:
    - MH01AB1234 (standard format)
    - DL3CAB1234 (Delhi format)
    - KA51MG1234 (Karnataka format)
    
    Args:
        text: Plate text to validate
        
    Returns:
        True if valid Indian plate format
    """
    if not text:
        return False
    
    normalized = normalize_plate_text(text)
    
    # Indian plate pattern: 2 letters (state) + 2 digits + 1-3 letters + 4 digits
    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$'
    return bool(re.match(pattern, normalized))


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage string.
    
    Args:
        confidence: Confidence value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence:.1%}"


def format_timestamp(dt: datetime, include_date: bool = True) -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime object
        include_date: Whether to include date portion
        
    Returns:
        Formatted timestamp string
    """
    if include_date:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%H:%M:%S")


def generate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Generate hash of a file for deduplication.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(filename: str) -> str:
    """
    Get lowercase file extension.
    
    Args:
        filename: File name or path
        
    Returns:
        Lowercase extension without dot
    """
    return Path(filename).suffix.lower().lstrip('.')


def is_video_file(filename: str) -> bool:
    """
    Check if file is a supported video format.
    
    Args:
        filename: File name or path
        
    Returns:
        True if video file
    """
    video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    return get_file_extension(filename) in video_extensions


def is_image_file(filename: str) -> bool:
    """
    Check if file is a supported image format.
    
    Args:
        filename: File name or path
        
    Returns:
        True if image file
    """
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}
    return get_file_extension(filename) in image_extensions


def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename.
    
    Removes or replaces characters that are invalid in filenames.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Replace invalid characters with underscore
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    safe = safe.strip('. ')
    return safe or 'unnamed'


def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def bytes_to_human_readable(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def seconds_to_human_readable(seconds: float) -> str:
    """
    Convert seconds to human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def parse_resolution(resolution_str: str) -> Optional[tuple]:
    """
    Parse resolution string to tuple.
    
    Args:
        resolution_str: Resolution like "1920x1080" or "1920,1080"
        
    Returns:
        Tuple of (width, height) or None if invalid
    """
    if not resolution_str:
        return None
    
    match = re.match(r'(\d+)[x,](\d+)', resolution_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def get_video_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get basic video file information.
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dictionary with file info (size, name, extension)
    """
    path = Path(file_path)
    return {
        'name': path.name,
        'extension': path.suffix.lower(),
        'size_bytes': path.stat().st_size if path.exists() else 0,
        'size_human': bytes_to_human_readable(path.stat().st_size) if path.exists() else '0 B',
        'exists': path.exists()
    }
