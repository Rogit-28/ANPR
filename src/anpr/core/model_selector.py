import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Handles the selection between day and night models based on lighting conditions.
    Uses YOLOv11 license plate detection model with optional night mode processing.
    """
    
    def __init__(self, day_model_path: Optional[str] = None, night_model_path: Optional[str] = None):
        """
        Initialize the ModelSelector with paths to day and night models.
        
        If no paths are provided, uses the default YOLOv11 license plate model from config.
        
        Args:
            day_model_path: Path to the day model (if None, uses config default)
            night_model_path: Path to the night model (if None, uses config default)
        """
        self.day_model_path = day_model_path or "models/yolov11s-license-plate.pt"
        self.night_model_path = night_model_path or "models/yolov11s-license-plate.pt"
        
        # Initialize model detectors (will be loaded lazily)
        self.day_detector = None
        self.night_detector = None
    
    def select_model(self, frame) -> Tuple[str, object]:
        """
        Select the appropriate model based on lighting conditions.
        
        Args:
            frame: Input image frame to analyze for lighting conditions
            
        Returns:
            Tuple of (model_type, detector_object) where model_type is 'day' or 'night'
        """
        is_low_light = self._detect_low_light_conditions(frame)
        
        if is_low_light:
            logger.info("Low light conditions detected, selecting night model")
            return 'night', self.night_detector
        else:
            logger.info("Normal light conditions detected, selecting day model")
            return 'day', self.day_detector
    
    def _detect_low_light_conditions(self, frame) -> bool:
        """Detect if frame is in low light conditions based on brightness."""
        import cv2
        import numpy as np
        
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Threshold for low light detection
        LOW_LIGHT_THRESHOLD = 50
        return mean_brightness < LOW_LIGHT_THRESHOLD
    
    def process_frame_for_model_selection(self, frame) -> Tuple[str, object, dict]:
        """
        Process a frame for model selection, including preprocessing if needed.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (model_type, detector_object, processing_metadata)
        """
        # Get brightness metrics for logging and decision making
        brightness_metrics = self._calculate_brightness_metrics(frame)
        
        # Select the appropriate model
        model_type, detector = self.select_model(frame)
        
        metadata = {
            'brightness_metrics': brightness_metrics,
            'selected_model_type': model_type,
            'night_mode_activated': model_type == 'night'
        }
        
        return model_type, detector, metadata
    
    def _calculate_brightness_metrics(self, frame) -> dict:
        """Calculate brightness metrics for a frame."""
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return {
            'mean_brightness': float(np.mean(gray)),
            'std_brightness': float(np.std(gray)),
            'min_brightness': int(np.min(gray)),
            'max_brightness': int(np.max(gray))
        }
    
    def get_brightness_analysis(self, frame) -> dict:
        """
        Get detailed brightness analysis of a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing brightness analysis
        """
        return self._calculate_brightness_metrics(frame)
