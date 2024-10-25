import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NightVisionProcessor:
    """
    Implements the night vision preprocessing pipeline for the ANPR system.
    Based on sections 4.1 and 7 of the specification document.
    """
    
    def __init__(self):
        """
        Initialize the Night Vision Processor with default parameters.
        """
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Bilateral filter parameters
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75
        
        # Unsharp masking parameters
        self.unsharp_kernel_size = 5
        self.unsharp_amount = 1.5
        
        # Brightness threshold for night mode activation
        self.brightness_threshold = 0.3  # 30% pixels above mid-tone
        
    def detect_low_light_conditions(self, frame: np.ndarray) -> bool:
        """
        Detect low light conditions based on frame brightness histogram.
        
        Args:
            frame: Input image frame
            
        Returns:
            True if low light conditions are detected, False otherwise
        """
        try:
            # Convert to grayscale if the frame is in color
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            # Calculate histogram
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            
            # Normalize histogram
            hist = hist.flatten() / hist.sum()
            
            # Calculate percentage of pixels above mid-tone (127)
            pixels_above_mid_tone = np.sum(hist[128:])
            
            # If less than 30% of pixels are above mid-tone, activate night mode
            return pixels_above_mid_tone < self.brightness_threshold
            
        except Exception as e:
            logger.error(f"Error detecting low light conditions: {e}")
            return False
    
    def calculate_brightness_metrics(self, frame: np.ndarray) -> dict:
        """
        Calculate various brightness metrics for the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing brightness metrics
        """
        try:
            # Convert to grayscale if the frame is in color
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            # Calculate mean brightness
            mean_brightness = float(np.mean(gray_frame))
            
            # Calculate median brightness
            median_brightness = float(np.median(gray_frame))
            
            # Calculate standard deviation of brightness
            std_brightness = float(np.std(gray_frame))
            
            # Calculate histogram-based metrics
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            pixels_below_50 = float(np.sum(hist[:51]))  # Very dark pixels
            pixels_above_200 = float(np.sum(hist[201:]))  # Very bright pixels
            pixels_above_mid_tone = float(np.sum(hist[128:]))  # Above mid-tone
            
            return {
                'mean_brightness': mean_brightness,
                'median_brightness': median_brightness,
                'std_brightness': std_brightness,
                'pixels_below_50': pixels_below_50,
                'pixels_above_200': pixels_above_200,
                'pixels_above_mid_tone': pixels_above_mid_tone
            }
            
        except Exception as e:
            logger.error(f"Error calculating brightness metrics: {e}")
            return {}
