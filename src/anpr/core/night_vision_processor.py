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
    
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with CLAHE applied
        """
        try:
            if len(frame.shape) == 3:
                # Apply CLAHE to each channel separately for color images
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                
                # Apply CLAHE to the L channel
                l_channel = self.clahe.apply(l_channel)
                
                # Merge channels back
                lab = cv2.merge((l_channel, a, b))
                enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Apply CLAHE to grayscale image
                enhanced_frame = self.clahe.apply(frame)
            
            return enhanced_frame
            
        except Exception as e:
            logger.error(f"Error applying CLAHE: {e}")
            return frame
    
    def apply_bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering to preserve edges while smoothing noise.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with bilateral filter applied
        """
        try:
            filtered_frame = cv2.bilateralFilter(
                frame,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
            return filtered_frame
            
        except Exception as e:
            logger.error(f"Error applying bilateral filter: {e}")
            return frame
    
    def calculate_dynamic_gamma(self, frame: np.ndarray) -> float:
        """
        Calculate dynamic gamma based on mean brightness.
        
        Args:
            frame: Input image frame
            
        Returns:
            Calculated gamma value in range 1.5-2.5
        """
        try:
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            mean_brightness = np.mean(gray_frame)
            
            # Map mean brightness (0-255) to gamma range (1.5-2.5)
            # Lower brightness -> higher gamma (brighter image)
            gamma = 2.5 - ((mean_brightness / 255.0) * 1.0)
            
            # Clamp gamma to the required range
            gamma = max(1.5, min(2.5, gamma))
            
            return gamma
            
        except Exception as e:
            logger.error(f"Error calculating dynamic gamma: {e}")
            return 2.0 # Default gamma value
    
    def apply_gamma_correction(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to the frame.
        
        Args:
            frame: Input image frame
            gamma: Gamma value to apply
            
        Returns:
            Frame with gamma correction applied
        """
        try:
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            
            corrected_frame = cv2.LUT(frame, table)
            return corrected_frame
            
        except Exception as e:
            logger.error(f"Error applying gamma correction: {e}")
            return frame
