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
            
    def process_stream(self, source: str, use_original_resolution: bool = True) -> Generator[np.ndarray, None, None]:
        """
        Process video stream using FFmpeg with hardware acceleration.
        Yields frames in BGR format (OpenCV native).
        
        Args:
            source: Path to video file or stream URL
            use_original_resolution: If True, preserves original resolution (capped at 2K).
                                    If False, uses target_resolution.
        """
        if not self._is_valid_source(source):
            raise ValueError(f"Invalid video source: {source}")
        
        # Determine resolution and fps to use
        metadata = get_video_metadata(source)
        
        if use_original_resolution and metadata:
            # Use original video dimensions (with 2K cap)
            width, height = metadata['width'], metadata['height']
            fps = metadata['fps'] or 24  # Fallback to 24 if fps detection fails
            
            # Apply 2K resolution cap if needed
            if width > MAX_RESOLUTION[0] or height > MAX_RESOLUTION[1]:
                scale = min(MAX_RESOLUTION[0] / width, MAX_RESOLUTION[1] / height)
                width = int(width * scale)
                height = int(height * scale)
                # Round to even numbers (required by many codecs)
                width = width - (width % 2)
                height = height - (height % 2)
                scale_filter = f'scale={width}:{height}'
                logger.info(f"Video exceeds 2K cap, scaling to {width}x{height}")
            else:
                scale_filter = None  # No scaling needed
                logger.info(f"Using original resolution: {width}x{height} @ {fps}fps")
        else:
            # Use target resolution (fallback mode)
            width, height = self.target_resolution or (1280, 720)
            fps = self.target_fps or 24
            scale_filter = f'scale={width}:{height}'
            logger.info(f"Using target resolution: {width}x{height} @ {fps}fps")
            
        # Determine appropriate FFmpeg parameters based on input type
        ffmpeg_params = self._get_ffmpeg_params(source)
        
        # Check if NVIDIA GPU is available for hardware acceleration
        use_nvidia = self._check_nvidia_available()
        
        if use_nvidia:
            # Build FFmpeg command with NVIDIA hardware-accelerated decoding
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',          # GPU-accelerated decoding (NVDEC)
                '-i', source,
            ]
            # Add scale filter only if needed
            if scale_filter:
                cmd.extend(['-vf', scale_filter])
            # Note: No -r flag = preserve original frame rate
            cmd.extend([
                '-pix_fmt', 'bgr24',         # Output in BGR format for OpenCV native format
                '-f', 'rawvideo',
                '-'
            ])
        else:
            # CPU-only fallback without NVIDIA hardware acceleration
            logger.info("NVIDIA GPU not available, using CPU-based video processing")
            cmd = [
                'ffmpeg',
                '-i', source,
            ]
            # Add scale filter only if needed
            if scale_filter:
                cmd.extend(['-vf', scale_filter])
            # Note: No -r flag = preserve original frame rate
            cmd.extend([
                '-pix_fmt', 'bgr24',         # Output in BGR format for OpenCV native format
                '-f', 'rawvideo',
                '-'
            ])
        
        # Add specific parameters based on source type
        cmd = ffmpeg_params + cmd[1:]  # Insert source-specific params after 'ffmpeg'
        
        try:
            # Use stderr=subprocess.DEVNULL to prevent pipe buffer deadlock.
            # FFmpeg writes progress/warnings to stderr, and if we don't read it,
            # the OS pipe buffer fills up and FFmpeg blocks, causing a hang.
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Prevent deadlock from unread stderr
                bufsize=width * height * 3  # Frame size in bytes
            )
            
            frame_size = width * height * 3  # BGR = 3 channels
            frame_count = 0
            
            while True:
                raw_frame = self.process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    logger.info(f"Video stream ended after {frame_count} frames (read {len(raw_frame)} bytes, expected {frame_size})")
                    break
                    
                frame_count += 1
                
                # Convert to numpy array and reshape to image dimensions
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                
                # NOTE: Night vision processing is now handled by pipeline_manager
                # to avoid double processing and improve performance
                
                yield frame
                
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            raise
        finally:
            if self.process:
                logger.debug("Closing video stream process...")
                # Only close stdout if it's a valid pipe (not None)
                if self.process.stdout:
                    try:
                        self.process.stdout.close()
                    except Exception as e:
                        logger.debug(f"Error closing stdout: {e}")
                # Note: stderr is DEVNULL, not a pipe, so don't try to close it
                # Wait for process to terminate properly
                try:
                    self.process.wait(timeout=5)
                    logger.debug("Video stream process terminated")
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process did not terminate, killing it")
                    self.process.kill()
                    self.process.wait()
                self.process = None
                
    def _is_valid_source(self, source: str) -> bool:
        """Check if the source is a valid video file or stream."""
        # Check if it's a local file
        if os.path.isfile(source):
            return True
            
        # Check if it's a URL (RTSP/HTTP)
        if source.startswith(('rtsp://', 'http://', 'https://')):
            return True
            
        # If it's neither a file nor a URL, it might be invalid
        return False
        
    def _get_ffmpeg_params(self, source: str) -> list:
        """Determine appropriate FFmpeg parameters based on input type."""
        params = ['ffmpeg']
        
        # Add parameters based on source type
        if source.startswith('rtsp://'):
            # RTSP specific parameters for stability
            params.extend([
                '-rtsp_transport', 'tcp',  # Use TCP transport for reliability
                '-stimeout', '500000'     # 5 second timeout
            ])
        elif source.startswith(('http://', 'https://')):
            # HTTP specific parameters
            params.extend([
                '-timeout', '5000'       # 5 second timeout
            ])
        else:
            # Local file parameters
            if source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                pass
                
        return params
    
    def _check_nvidia_available(self) -> bool:
        """Check if NVIDIA GPU with CUDA is available for FFmpeg decoding."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
            
            result = subprocess.run(
                ['ffmpeg', '-hwaccels'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'cuda' in result.stdout.lower():
                logger.info("NVIDIA CUDA hardware acceleration available")
                return True
            
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"NVIDIA check failed: {e}")
            return False


def is_valid_video_source(source: str) -> bool:
    """
    Utility function to check if a source is a valid video file or stream.
    """
    # Check if it's a local file
    if os.path.isfile(source):
        # Check if it has a valid video extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.flv', '.wmv', '.webm', '.m4v'}
        _, ext = os.path.splitext(source.lower())
        return ext in valid_extensions
    
    # Check if it's a URL (RTSP/HTTP)
    if source.startswith(('rtsp://', 'http://', 'https://', 'mms://', 'udp://', 'rtp://')):
        return True
    
    return False


def get_ffmpeg_input_params(source: str) -> list:
    """
    Utility function to determine the appropriate FFmpeg parameters based on input type.
    """
    params = []
    
    # Add parameters based on source type
    if source.startswith('rtsp://'):
        # RTSP specific parameters for stability
        params.extend([
            '-rtsp_transport', 'tcp',  # Use TCP transport for reliability
            '-stimeout', '5000000'     # 5 second timeout
        ])
    elif source.startswith(('http://', 'https://')):
        # HTTP specific parameters
        params.extend([
            '-timeout', '5000000',      # 5 second timeout in microseconds
            '-reconnect', '1',          # Enable reconnection
            '-reconnect_at_eof', '1',   # Reconnect at end of file
            '-reconnect_streamed', '1', # Reconnect streamed data
            '-reconnect_delay_max', '2' # Maximum reconnect delay
        ])
    elif os.path.isfile(source):
        # Local file parameters
        if source.endswith(('.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.flv', '.wmv', '.webm', '.m4v')):
            # Add format-specific parameters if needed
            params.extend([
                '-re',  # Read input at native frame rate
            ])
                
    return params


def get_video_dimensions(source: str) -> Optional[Tuple[int, int, bool]]:
    """
    Utility function to calculate frame dimensions and validate resolution.
    Returns tuple of (width, height, is_valid_resolution) or None if failed.
    """
    try:
        video_handler = VideoHandler()
        info = video_handler.get_video_info(source)
        return (info['width'], info['height'], info['is_valid_resolution'])
    except Exception:
        return None


def estimate_video_duration(source: str) -> Optional[float]:
    """
    Utility function to estimate the duration of a video source in seconds.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            source
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
            
        duration_str = result.stdout.strip()
        if duration_str == 'N/A' or not duration_str:
            return None
            
        return float(duration_str)
    except Exception:
        return None


def get_video_codec(source: str) -> Optional[str]:
    """
    Utility function to get the codec of a video source.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'csv=p=0',
            source
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
            
        return result.stdout.strip()
    except Exception:
        return None


async def normalize_video_async(
    input_path: str,
    output_path: str,
    target_resolution: Tuple[int, int] = (1280, 720),
    target_fps: int = 24,
    target_codec: str = 'libx264',
    crf: int = 23,
    preset: str = 'medium',
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Async wrapper for normalizing video using FFmpeg subprocess.
    """
    try:
        # Get input video info (this is fast enough to keep synchronous or move to thread if needed)
        input_info = get_video_dimensions(input_path)
        if not input_info:
            return {
                'success': False,
                'error': f'Could not read video info from {input_path}'
            }
        
        input_width, input_height, _ = input_info
        
        # We need to run get_video_duration synchronously or wrap it too. 
        # For now, let's just use the sync version since it's a quick probe.
        duration = estimate_video_duration(input_path)
        
        # Check if NVIDIA GPU is available for hardware encoding
        # This is a synchronous check but usually fast
        use_nvenc = _check_nvenc_available() and target_codec == 'libx264'
        if use_nvenc:
            target_codec = 'h264_nvenc'
            logger.info("Using NVIDIA NVENC for hardware-accelerated encoding")
        
        # Build FFmpeg command
        scale_filter = (
            f"scale={target_resolution[0]}:{target_resolution[1]}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={target_resolution[0]}:{target_resolution[1]}:(ow-iw)/2:(oh-ih)/2:black"
        )
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', input_path,
            '-vf', scale_filter,
            '-r', str(target_fps),
            '-c:v', target_codec,
        ]
        
        if target_codec == 'h264_nvenc':
            cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', str(crf)])
        else:
            cmd.extend(['-preset', preset, '-crf', str(crf)])
        
        cmd.extend([
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-an',
            output_path
        ])
        
        logger.info(f"Normalizing video (Async): {input_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Use asyncio.create_subprocess_exec to avoid blocking the event loop
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=3600  # 1 hour timeout
            )
        except asyncio.TimeoutError:
            logger.error("Video normalization timed out (exceeded 1 hour)")
            process.kill()
            await process.wait()
            return {
                'success': False,
                'error': 'Video normalization timed out (exceeded 1 hour)'
            }
        
        if process.returncode != 0:
            stderr_decoded = stderr.decode() if stderr else "Unknown error"
            logger.error(f"FFmpeg normalization failed: {stderr_decoded}")
            return {
                'success': False,
                'error': f'FFmpeg failed: {stderr_decoded[-500:] if len(stderr_decoded) > 500 else stderr_decoded}'
            }
            
        # Get output video info
        output_info = get_video_dimensions(output_path)
        output_duration = estimate_video_duration(output_path)
        
        if not output_info:
             return {
                'success': False,
                'error': 'Failed to verify output video'
            }
            
        output_width, output_height, _ = output_info
        frame_count = int(output_duration * target_fps) if output_duration else 0
        
        logger.info(f"Video normalized successfully: {output_width}x{output_height} @ {target_fps}fps")
        
        return {
            'success': True,
            'output_path': output_path,
            'duration': output_duration,
            'frame_count': frame_count,
            'fps': float(target_fps),
            'width': output_width,
            'height': output_height
        }

    except Exception as e:
        logger.error(f"Error normalizing video (Async): {e}")
        return {
            'success': False,
            'error': str(e)
        }

def normalize_video(
    input_path: str,
    output_path: str,
    target_resolution: Tuple[int, int] = (1280, 720),
    target_fps: int = 24,
    target_codec: str = 'libx264',
    crf: int = 23,
    preset: str = 'medium',
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Normalize a video to standard format for consistent processing.
    
    Converts video to:
    - Resolution: 720p (1280x720) by default
    - Codec: H.264 (libx264)
    - Frame rate: 24 fps
    - Audio: AAC or removed
    
    Args:
        input_path: Path to input video file
        output_path: Path for output normalized video
        target_resolution: Target resolution as (width, height)
        target_fps: Target frame rate
        target_codec: Video codec to use (libx264, h264_nvenc for GPU)
        crf: Constant Rate Factor for quality (lower = better, 18-28 typical)
        preset: Encoding speed preset (ultrafast, fast, medium, slow)
        progress_callback: Optional callback function(progress_percent: float)
    
    Returns:
        dict with normalization results:
        - success: bool
        - output_path: str
        - duration: float (seconds)
        - frame_count: int
        - fps: float
        - width: int
        - height: int
        - error: str (if failed)
    """
    try:
        # Get input video info
        input_info = get_video_dimensions(input_path)
        if not input_info:
            return {
                'success': False,
                'error': f'Could not read video info from {input_path}'
            }
        
        input_width, input_height, _ = input_info
        duration = estimate_video_duration(input_path)
        
        # Check if NVIDIA GPU is available for hardware encoding
        use_nvenc = _check_nvenc_available() and target_codec == 'libx264'
        if use_nvenc:
            target_codec = 'h264_nvenc'
            logger.info("Using NVIDIA NVENC for hardware-accelerated encoding")
        
        # Build FFmpeg command
        # Scale filter to target resolution, maintaining aspect ratio with padding
        scale_filter = (
            f"scale={target_resolution[0]}:{target_resolution[1]}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={target_resolution[0]}:{target_resolution[1]}:(ow-iw)/2:(oh-ih)/2:black"
        )
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', input_path,
            '-vf', scale_filter,
            '-r', str(target_fps),
            '-c:v', target_codec,
        ]
        
        # Add codec-specific options
        if target_codec == 'h264_nvenc':
            cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', str(crf)])
        else:
            cmd.extend(['-preset', preset, '-crf', str(crf)])
        
        # Add common options
        cmd.extend([
            '-pix_fmt', 'yuv420p',  # Compatibility
            '-movflags', '+faststart',  # Web streaming optimization
            '-an',  # Remove audio (not needed for ANPR)
            output_path
        ])
        
        logger.info(f"Normalizing video: {input_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for completion
        _, stderr = process.communicate(timeout=3600)  # 1 hour timeout
        
        if process.returncode != 0:
            logger.error(f"FFmpeg normalization failed: {stderr}")
            return {
                'success': False,
                'error': f'FFmpeg failed: {stderr[-500:] if len(stderr) > 500 else stderr}'
            }
        
        # Get output video info
        output_info = get_video_dimensions(output_path)
        output_duration = estimate_video_duration(output_path)
        
        if not output_info:
            return {
                'success': False,
                'error': 'Failed to verify output video'
            }
        
        output_width, output_height, _ = output_info
        frame_count = int(output_duration * target_fps) if output_duration else 0
        
        logger.info(f"Video normalized successfully: {output_width}x{output_height} @ {target_fps}fps")
        
        return {
            'success': True,
            'output_path': output_path,
            'duration': output_duration,
            'frame_count': frame_count,
            'fps': float(target_fps),
            'width': output_width,
            'height': output_height
        }
        
    except subprocess.TimeoutExpired:
        # Kill the process on timeout to prevent zombie processes
        process.kill()
        process.wait()
        logger.error("Video normalization timed out")
        return {
            'success': False,
            'error': 'Video normalization timed out (exceeded 1 hour)'
        }
    except Exception as e:
        logger.error(f"Error normalizing video: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def get_video_metadata(source: str) -> Optional[dict]:
    """
    Get comprehensive video metadata.
    
    Returns:
        dict with video metadata or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            source
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        
        import json
        info = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        fps = _parse_fps(video_stream.get('avg_frame_rate', '0'))
        
        # Get duration
        duration = float(info.get('format', {}).get('duration', 0))
        
        # Calculate frame count
        frame_count = int(video_stream.get('nb_frames', 0))
        if frame_count == 0 and duration > 0 and fps > 0:
            frame_count = int(duration * fps)
        
        return {
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count,
            'codec': video_stream.get('codec_name', 'unknown'),
            'bitrate': int(info.get('format', {}).get('bit_rate', 0)),
            'size_bytes': int(info.get('format', {}).get('size', 0))
        }
    except Exception as e:
        logger.error(f"Error getting video metadata: {e}")
        return None
