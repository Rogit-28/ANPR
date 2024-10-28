import cv2
import numpy as np
import subprocess
import asyncio
import os
import tempfile
from typing import Generator, Tuple, Optional, Union
import logging
from pathlib import Path

from .night_vision_processor import NightVisionProcessor

logger = logging.getLogger(__name__)

# Maximum resolution cap (2K) to prevent memory issues with 4K+ videos
MAX_RESOLUTION = (2560, 1440)


def _parse_fps(fps_str: str) -> float:
    """Parse FFprobe frame rate string (e.g., '30/1' or '29.97') to float."""
    try:
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return float(num) / float(den) if float(den) != 0 else 0.0
        else:
            return float(fps_str) if fps_str else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _check_nvenc_available() -> bool:
    """Check if NVIDIA NVENC encoder is available for FFmpeg."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'h264_nvenc' in result.stdout:
            nvidia_check = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvidia_check.returncode == 0:
                return True
        return False
    except Exception:
        return False


class VideoHandler:
    """
    VideoHandler class that wraps FFmpeg with the following responsibilities:
    - Stream ingestion (RTSP/HTTP/file)
    - Frame extraction at original or target FPS
    - Resolution handling (original with 2K cap, or forced target)
    - Proper error handling for different stream types
    
    Note: Video normalization is now bypassed for uploaded files to preserve
    original quality. Only resolution capping (2K max) is applied if needed.
    """
    
    def __init__(self, target_resolution: Optional[Tuple[int, int]] = None, target_fps: Optional[int] = None):
        """
        Initialize VideoHandler.
        
        Args:
            target_resolution: Target resolution (width, height). If None, uses original (capped at 2K).
            target_fps: Target frame rate. If None, uses original video fps.
        """
        self.target_resolution = target_resolution  # None = use original (with 2K cap)
        self.target_fps = target_fps  # None = use original fps
        self.process = None
        self.temp_files = []
        self.night_vision_processor = NightVisionProcessor()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Properly clean up FFmpeg processes and temporary files."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
                
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
                
    def validate_resolution(self, width: int, height: int) -> bool:
        """Validate that resolution meets minimum requirement of 720p."""
        return width >= 1280 and height >= 720
        
    def get_video_info(self, source: str) -> dict:
        """Get video information using FFprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                source
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise ValueError(f"FFprobe failed: {result.stderr}")
                
            import json
            info = json.loads(result.stdout)
            
            # Extract stream info
            video_stream = None
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
                    
            if not video_stream:
                raise ValueError("No video stream found in source")
                
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            fps = _parse_fps(video_stream.get('avg_frame_rate', '0'))
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'codec': video_stream.get('codec_name', 'unknown'),
                'is_valid_resolution': self.validate_resolution(width, height)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
