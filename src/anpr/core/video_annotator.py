"""Video Annotator - draws detection overlays on video frames."""

import cv2
import numpy as np
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnnotationStyle:
    """Style configuration for video annotations."""
    # Bounding box
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)
    bbox_thickness: int = 2
    bbox_corner_length: int = 15
    
    # Text background
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    text_bg_alpha: float = 0.7
    text_padding: int = 8
    
    # Text
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.7
    text_thickness: int = 2
    
    # Confidence indicator
    confidence_high_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    confidence_medium_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    confidence_low_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    confidence_high_threshold: float = 0.85
    confidence_medium_threshold: float = 0.70


class VideoAnnotator:
    """Draws bounding boxes and plate text on video frames with confidence-based coloring."""
    
    def __init__(self, style: Optional[AnnotationStyle] = None):
        self.style = style or AnnotationStyle()
        self._detection_cache: Dict[int, List[Dict]] = {}
        
    async def annotate_video_async(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_frame_number: bool = False,
        persist_frames: int = 15
    ) -> Dict[str, Any]:
        """Async wrapper - runs frame processing in thread pool."""
        try:
            # Run the heavy processing loop in a separate thread to avoid blocking the event loop
            # since cv2 operations are CPU bound
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._process_annotation_frames,
                input_path,
                output_path,
                detections,
                show_confidence,
                show_frame_number,
                persist_frames
            )
            return result
        except Exception as e:
            logger.error(f"Error annotating video (async): {e}")
            return {'success': False, 'error': str(e)}

    def _process_annotation_frames(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_frame_number: bool = False,
        persist_frames: int = 15
    ) -> Dict[str, Any]:
        """Internal synchronous method to process frames, moved from annotate_video."""
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return {'success': False, 'error': f'Could not open video: {input_path}'}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Annotating video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Build detection lookup by frame number
            self._build_detection_cache(detections, persist_frames, total_frames)
            
            # Create temporary raw output
            temp_output = str(Path(output_path).with_suffix('.temp.mp4'))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                cap.release()
                return {'success': False, 'error': 'Could not create video writer'}
            
            # Process frames
            frame_number = 0
            frames_processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get detections for this frame
                frame_detections = self._detection_cache.get(frame_number, [])
                
                # Annotate frame
                annotated_frame = self._annotate_frame(
                    frame, 
                    frame_detections,
                    show_confidence=show_confidence,
                    frame_number=frame_number if show_frame_number else None
                )
                
                # Write frame
                writer.write(annotated_frame)
                
                frame_number += 1
                frames_processed += 1
                
                # Log progress
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    logger.debug(f"Annotation progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            writer.release()
            
            # Re-encode with H.264
            success = self._reencode_video(temp_output, output_path, fps)
            
            # Remove temp file
            try:
                Path(temp_output).unlink()
            except Exception:
                pass
            
            if not success:
                return {'success': False, 'error': 'Failed to re-encode video'}
            
            logger.info(f"Annotation complete: {frames_processed} frames processed")
            
            return {
                'success': True,
                'output_path': output_path,
                'frames_processed': frames_processed
            }
            
        except Exception as e:
            logger.error(f"Error annotating video: {e}")
            return {'success': False, 'error': str(e)}

    def annotate_video(
        self,
        input_path: str,
        output_path: str,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
        show_frame_number: bool = False,
        persist_frames: int = 15
    ) -> Dict[str, Any]:
        """
        Create annotated video with detection overlays (Legacy synchronous wrapper).
        """
        return self._process_annotation_frames(
            input_path, output_path, detections, show_confidence, show_frame_number, persist_frames
        )

    def _build_detection_cache(
        self, 
        detections: List[Dict], 
        persist_frames: int,
        total_frames: int
    ):
        """Build frame-indexed detection cache with fade-out effect."""
        self._detection_cache = {}
        
        for det in detections:
            frame_num = det.get('frame_number', 0)
            
            # Add detection to its frame and subsequent frames with fading
            for offset in range(persist_frames + 1):
                target_frame = frame_num + offset
                if target_frame >= total_frames:
                    break
                
                # Calculate fade factor (1.0 at detection frame, fading out)
                fade = 1.0 - (offset / (persist_frames + 1))
                
                if target_frame not in self._detection_cache:
                    self._detection_cache[target_frame] = []
                
                # Add detection with fade info
                cached_det = det.copy()
                cached_det['fade'] = fade
                self._detection_cache[target_frame].append(cached_det)
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        frame_number: Optional[int] = None
    ) -> np.ndarray:
        """Draw detection overlays on a single frame."""
        annotated = frame.copy()
        
        for det in detections:
            fade = det.get('fade', 1.0)
            bbox = det.get('bbox', det.get('bounding_box'))
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det.get('confidence', 0.0)
            plate_text = det.get('plate_text', det.get('text', ''))
            
            # Get color based on confidence
            color = self._get_confidence_color(confidence)
            
            # Draw bounding box with corners
            self._draw_bbox(annotated, x1, y1, x2, y2, color, fade)
            
            # Build label text
            if plate_text:
                label = plate_text
                if show_confidence:
                    label += f" ({confidence:.0%})"
            elif show_confidence:
                label = f"{confidence:.0%}"
            else:
                label = ""
            
            # Draw label if we have one
            if label:
                self._draw_label(annotated, label, x1, y1, color, fade)
        
        # Draw frame number if requested
        if frame_number is not None:
            self._draw_frame_number(annotated, frame_number)
        
        return annotated
    
    def _draw_bbox(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: Tuple[int, int, int],
        alpha: float = 1.0
    ):
        """Draw stylized corner bounding box."""
        thickness = self.style.bbox_thickness
        corner_len = self.style.bbox_corner_length
        
        # Apply alpha to color
        if alpha < 1.0:
            color = tuple(int(c * alpha) for c in color)
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
        
        # Draw thin full rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level."""
        if confidence >= self.style.confidence_high_threshold:
            return self.style.confidence_high_color
        elif confidence >= self.style.confidence_medium_threshold:
            return self.style.confidence_medium_color
        else:
            return self.style.confidence_low_color
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int, y: int,
        color: Tuple[int, int, int],
        alpha: float = 1.0
    ):
        """Draw label with semi-transparent background."""
        font = self.style.text_font
        scale = self.style.text_scale
        thickness = self.style.text_thickness
        padding = self.style.text_padding
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Calculate background rectangle
        bg_x1 = x
        bg_y1 = y - text_h - 2 * padding
        bg_x2 = x + text_w + 2 * padding
        bg_y2 = y
        
        # Ensure background is within frame
        if bg_y1 < 0:
            bg_y1 = y
            bg_y2 = y + text_h + 2 * padding
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.style.text_bg_color, -1)
        
        bg_alpha = self.style.text_bg_alpha * alpha
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw text
        text_x = bg_x1 + padding
        text_y = bg_y2 - padding
        
        # Apply alpha to text color
        text_color = self.style.text_color
        if alpha < 1.0:
            text_color = tuple(int(c * alpha) for c in text_color)
        
        cv2.putText(frame, text, (text_x, text_y), font, scale, text_color, thickness)
        
        # Draw color indicator line under text
        indicator_y = bg_y2 - 2
        cv2.line(frame, (bg_x1, indicator_y), (bg_x2, indicator_y), color, 2)
    
    def _draw_frame_number(self, frame: np.ndarray, frame_number: int):
        """Draw frame number in corner."""
        text = f"Frame: {frame_number}"
        font = self.style.text_font
        scale = 0.5
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        x = 10
        y = frame.shape[0] - 10
        
        # Draw background
        cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)
    
    def _reencode_video(self, input_path: str, output_path: str, fps: float) -> bool:
        """Re-encode video with H.264 for web compatibility."""
        try:
            # Check if NVENC is available
            use_nvenc = self._check_nvenc_available()
            
            if use_nvenc:
                codec = 'h264_nvenc'
                codec_opts = ['-preset', 'p4', '-rc', 'vbr', '-cq', '23']
            else:
                codec = 'libx264'
                codec_opts = ['-preset', 'medium', '-crf', '23']
            
            cmd = [
                'ffmpeg',
                '-y',
                '-i', input_path,
                '-c:v', codec,
                *codec_opts,
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-r', str(fps),
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg re-encode failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error re-encoding video: {e}")
            return False
    
    def _check_nvenc_available(self) -> bool:
        """Check if NVIDIA NVENC encoder is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'h264_nvenc' in result.stdout
        except Exception:
            return False
    
    def annotate_frame_realtime(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Annotate single frame for live display."""
        return self._annotate_frame(
            frame, 
            [{'fade': 1.0, **d} for d in detections],
            show_confidence=True,
            frame_number=None
        )


# Factory function
def create_video_annotator(style: Optional[AnnotationStyle] = None) -> VideoAnnotator:
    """Create a VideoAnnotator instance."""
    return VideoAnnotator(style)
