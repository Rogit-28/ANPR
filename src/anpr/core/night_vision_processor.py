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
    
    def detect_motion_blur(self, frame: np.ndarray) -> bool:
        """
        Detect motion blur in the frame using Laplacian variance.
        
        Args:
            frame: Input image frame
            
        Returns:
            True if motion blur is detected, False otherwise
        """
        try:
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            # Calculate Laplacian variance - lower values indicate blur
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Threshold for blur detection (this value may need tuning)
            blur_threshold = 100.0
            
            return laplacian_var < blur_threshold
            
        except Exception as e:
            logger.error(f"Error detecting motion blur: {e}")
            return False
    
    def apply_unsharp_masking(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to enhance details.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with unsharp masking applied
        """
        try:
            # Create Gaussian blurred version
            blurred = cv2.GaussianBlur(frame, (self.unsharp_kernel_size, self.unsharp_kernel_size), 0)
            
            # Create unsharp mask
            mask = cv2.subtract(frame, blurred)
            
            # Add weighted mask to original
            sharpened = cv2.addWeighted(frame, 1.0, mask, self.unsharp_amount, 0)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error applying unsharp masking: {e}")
            return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Main preprocessing function that applies all night vision enhancements.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (processed frame, processing metadata)
        """
        try:
            processing_metadata = {
                'night_mode_activated': False,
                'clahe_applied': False,
                'bilateral_filter_applied': False,
                'gamma_correction_applied': False,
                'unsharp_masking_applied': False,
                'dynamic_gamma_value': 0.0,
                'motion_blur_detected': False
            }
            
            # Check if we need to apply night vision preprocessing
            if self.detect_low_light_conditions(frame):
                processing_metadata['night_mode_activated'] = True
                
                # Apply CLAHE for contrast enhancement
                frame = self.safe_apply_clahe(frame)
                processing_metadata['clahe_applied'] = True
                
                # Apply bilateral filtering to reduce noise while preserving edges
                frame = self.safe_apply_bilateral_filter(frame)
                processing_metadata['bilateral_filter_applied'] = True
                
                # Calculate and apply dynamic gamma correction
                gamma = self.calculate_dynamic_gamma(frame)
                processing_metadata['dynamic_gamma_value'] = gamma
                frame = self.safe_apply_gamma_correction(frame, gamma)
                processing_metadata['gamma_correction_applied'] = True
                
                # Detect motion blur and apply unsharp masking if needed
                if self.detect_motion_blur(frame):
                    processing_metadata['motion_blur_detected'] = True
                    frame = self.safe_apply_unsharp_masking(frame)
                    processing_metadata['unsharp_masking_applied'] = True
            
            return frame, processing_metadata
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            # Return original frame if preprocessing fails
            return frame, {'error': str(e)}
    
    def process_with_fallback(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process frame with fallback mechanism in case of errors.
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (processed frame, processing metadata)
        """
        try:
            # Make a copy of the original frame for fallback
            original_frame = frame.copy()
            
            processed_frame, metadata = self.preprocess_frame(frame)
            
            # Verify that the processed frame is valid
            if processed_frame is None or processed_frame.size != original_frame.size:
                logger.warning("Processed frame validation failed, returning original frame")
                return original_frame, {'fallback_used': True, 'original_returned': True}
            
            # Additional check: ensure pixel values are in valid range
            if np.any(processed_frame > 255) or np.any(processed_frame < 0):
                logger.warning("Processed frame has invalid pixel values, returning original frame")
                return original_frame, {'fallback_used': True, 'invalid_pixels': True}
            
            return processed_frame, metadata
            
        except Exception as e:
            logger.error(f"Critical error in preprocessing with fallback: {e}")
            # Return original frame if anything goes wrong
            return frame, {'fallback_used': True, 'exception': str(e)}
    
    def safe_apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Safely apply CLAHE with error handling.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with CLAHE applied or original frame if error occurs
        """
        try:
            return self.apply_clahe(frame)
        except Exception as e:
            logger.error(f"Error in safe CLAHE application: {e}")
            return frame
    
    def safe_apply_bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Safely apply bilateral filter with error handling.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with bilateral filter applied or original frame if error occurs
        """
        try:
            return self.apply_bilateral_filter(frame)
        except Exception as e:
            logger.error(f"Error in safe bilateral filter application: {e}")
            return frame
    
    def safe_apply_gamma_correction(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        """
        Safely apply gamma correction with error handling.
        
        Args:
            frame: Input image frame
            gamma: Gamma value to apply
            
        Returns:
            Frame with gamma correction applied or original frame if error occurs
        """
        try:
            return self.apply_gamma_correction(frame, gamma)
        except Exception as e:
            logger.error(f"Error in safe gamma correction application: {e}")
            return frame
    
    def safe_apply_unsharp_masking(self, frame: np.ndarray) -> np.ndarray:
        """
        Safely apply unsharp masking with error handling.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with unsharp masking applied or original frame if error occurs
        """
        try:
            return self.apply_unsharp_masking(frame)
        except Exception as e:
            logger.error(f"Error in safe unsharp masking application: {e}")
            return frame