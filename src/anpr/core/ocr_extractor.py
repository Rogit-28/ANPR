"""OCR Extractor using RapidOCR, optimized for Indian license plates."""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Dict, List
import logging
import time

logger = logging.getLogger(__name__)


class OCRExtractor:
    """RapidOCR-based text extractor for Indian license plates."""
    
    def __init__(self, device: str = None, use_gpu: bool = True):
        """Initialize with optional GPU acceleration."""
        self.ocr = None
        self.use_gpu = use_gpu
        self._load_rapidocr_model()
        
        # Indian plate format patterns
        # Standard format: XX00XX0000 or XX00XXX0000
        self.indian_plate_pattern1 = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
        self.indian_plate_pattern2 = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{4}$')
        
        # Valid Indian state codes
        self.valid_state_codes = {
            'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
            'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
            'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY',
            'RJ', 'SK', 'TG', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
        }
        
        # Character confusion mapping for correction
        self.char_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2'}
        self.digit_to_char = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z'}
        
        # Confidence threshold for database writes
        self.confidence_threshold = 0.7
        
        # Reusable CLAHE object for preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info("OCRExtractor initialized with RapidOCR")

    def _load_rapidocr_model(self):
        """Load RapidOCR model with GPU acceleration if available."""
        try:
            # Add PyTorch's CUDA libraries to PATH for ONNX Runtime
            # This allows ONNX Runtime to find cuBLAS, cuDNN, etc.
            import torch
            import os
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib_path):
                current_path = os.environ.get('PATH', '')
                if torch_lib_path not in current_path:
                    os.environ['PATH'] = torch_lib_path + os.pathsep + current_path
                    logger.info(f"Added PyTorch CUDA libs to PATH: {torch_lib_path}")
            
            from rapidocr_onnxruntime import RapidOCR
            import onnxruntime as ort
            
            # Check available providers
            available_providers = ort.get_available_providers()
            logger.info(f"ONNX Runtime available providers: {available_providers}")
            
            # Determine if we can use GPU
            use_cuda = False
            use_dml = False
            
            if self.use_gpu:
                if 'CUDAExecutionProvider' in available_providers:
                    use_cuda = True
                    logger.info("CUDA provider available for ONNX Runtime")
                elif 'DmlExecutionProvider' in available_providers:
                    use_dml = True
                    logger.info("DirectML provider available for ONNX Runtime")
            
            # Initialize RapidOCR with GPU flags
            # RapidOCR will gracefully fall back to CPU if GPU fails
            if use_cuda:
                logger.info("Initializing RapidOCR with CUDA GPU acceleration")
                self.ocr = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
            elif use_dml:
                logger.info("Initializing RapidOCR with DirectML GPU acceleration")
                self.ocr = RapidOCR(det_use_dml=True, rec_use_dml=True, cls_use_dml=True)
            else:
                logger.info("Initializing RapidOCR with CPU (GPU not available or disabled)")
                self.ocr = RapidOCR()
            
            # Log actual providers being used
            try:
                det_providers = self.ocr.text_det.infer.session.get_providers()
                logger.info(f"RapidOCR text detection actually using: {det_providers}")
            except Exception:
                pass
            
            logger.info("RapidOCR model loaded successfully")
            
        except ImportError as e:
            logger.error(f"RapidOCR not installed: {e}")
            raise RuntimeError("RapidOCR is not installed. Run: pip install rapidocr-onnxruntime")
        except Exception as e:
            logger.error(f"Failed to load RapidOCR model: {e}")
            raise RuntimeError(f"RapidOCR initialization failed: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE and denoising for better OCR accuracy."""
        if image is None or image.size == 0:
            return image
            
        # Make a copy to avoid modifying original
        img = image.copy()
        
        # Resize if too small (OCR works better with larger images)
        h, w = img.shape[:2]
        if w < 200:
            scale = 200 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = self._clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Convert back to BGR for RapidOCR
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return result

    def _clean_plate_text(self, text: str) -> str:
        """Normalize to uppercase alphanumeric only."""
        if not text:
            return ""
        
        # Convert to uppercase
        text = text.upper()
        
        # Remove all spaces, hyphens, dots, and other separators
        text = re.sub(r'[\s\-\.\,\_\|]', '', text)
        
        # Keep only alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text

    def _extract_plate_from_text(self, text: str) -> str:
        """Extract plate number from text that may contain logos/watermarks."""
        if not text or len(text) < 9:
            return text
        
        # Pattern to find Indian plate format anywhere in the text
        # Matches: XX00X0000, XX00XX0000, XX00XXX0000
        plate_pattern = re.compile(r'([A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4})')
        
        match = plate_pattern.search(text)
        if match:
            return match.group(1)
        
        # Try to find plate by looking for valid state codes
        for state_code in self.valid_state_codes:
            if state_code in text:
                # Find position of state code
                pos = text.find(state_code)
                
                # Extract potential plate starting from state code
                # Plate is typically 9-11 characters
                for length in [11, 10, 9]:
                    if pos + length <= len(text):
                        candidate = text[pos:pos + length]
                        
                        # Check if it matches plate format
                        if self.validate_plate_format(candidate):
                            return candidate
        
        # If no valid plate found, try trimming from both ends
        # Sometimes extra characters are at the beginning or end
        for start in range(min(5, len(text) - 9)):
            for end in range(len(text), max(8, len(text) - 5), -1):
                candidate = text[start:end]
                if len(candidate) >= 9 and len(candidate) <= 11:
                    if self.validate_plate_format(candidate):
                        return candidate
        
        return text

    def _correct_character_confusion(self, text: str) -> str:
        """Fix OCR confusions (O/0, I/1, etc.) based on expected position in Indian plates."""
        if len(text) < 9:
            return text
        
        chars = list(text)
        
        # Positions 0-1 should be letters (state code)
        for i in range(min(2, len(chars))):
            if chars[i] in self.digit_to_char:
                chars[i] = self.digit_to_char[chars[i]]
        
        # Positions 2-3 should be digits (district code)
        for i in range(2, min(4, len(chars))):
            if chars[i] in self.char_to_digit:
                chars[i] = self.char_to_digit[chars[i]]
        
        # Determine series length (1-3 letters)
        # Standard: 10 chars total = 2 state + 2 district + 2 series + 4 number
        # New: 11 chars total = 2 state + 2 district + 3 series + 4 number
        remaining = len(chars) - 4
        if remaining >= 7:  # 3 letter series + 4 digits
            series_len = 3
        elif remaining >= 6:  # 2 letter series + 4 digits
            series_len = 2
        else:
            series_len = 1
        
        # Series positions should be letters
        series_end = 4 + series_len
        for i in range(4, min(series_end, len(chars))):
            if chars[i] in self.digit_to_char:
                chars[i] = self.digit_to_char[chars[i]]
        
        # Remaining positions should be digits (registration number)
        for i in range(series_end, len(chars)):
            if chars[i] in self.char_to_digit:
                chars[i] = self.char_to_digit[chars[i]]
        
        return ''.join(chars)

    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract and clean plate text, returning (text, confidence)."""
        if image is None or image.size == 0:
            return "", 0.0
        
        try:
            # Preprocess the image
            processed = self.preprocess_image(image)
            
            # Run RapidOCR
            # RapidOCR returns: (result, elapse)
            # result is a list of [box, text, confidence] or None
            result, elapse = self.ocr(processed)
            
            if result is None or len(result) == 0:
                return "", 0.0
            
            all_texts = []
            all_scores = []
            
            # Parse RapidOCR result format
            # Each item is [box_coords, text, confidence]
            for item in result:
                if len(item) >= 3:
                    text = item[1]
                    confidence = item[2]
                    if text:
                        all_texts.append(text)
                        all_scores.append(confidence)
            
            if not all_texts:
                return "", 0.0
            
            # Combine all detected text segments
            combined_text = ''.join(all_texts)
            avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            # Clean the text
            cleaned_text = self._clean_plate_text(combined_text)
            
            # Extract plate number from text (handles extra characters from logos/watermarks)
            extracted_plate = self._extract_plate_from_text(cleaned_text)
            
            # Apply character confusion correction
            corrected_text = self._correct_character_confusion(extracted_plate)
            
            return corrected_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return "", 0.0

    def validate_plate_format(self, text: str) -> bool:
        """Check if text matches Indian plate format (XX00XX0000 or XX00XXX0000)."""
        if not text:
            return False
        
        return bool(
            self.indian_plate_pattern1.match(text) or 
            self.indian_plate_pattern2.match(text)
        )

    def validate_state_code(self, text: str) -> bool:
        """Check if first 2 characters are a valid Indian state code."""
        if len(text) < 2:
            return False
        
        return text[:2] in self.valid_state_codes

    def calculate_aggregate_confidence(self, char_confidences: List[float]) -> float:
        """Geometric mean of per-character confidences."""
        if not char_confidences:
            return 0.0
        
        product = 1.0
        for conf in char_confidences:
            product *= max(conf, 0.01)
        
        return product ** (1.0 / len(char_confidences))

    def calculate_per_char_confidence(self, text: str, base_confidence: float = 0.85) -> List[float]:
        """Estimate per-character confidence, penalizing commonly confused chars."""
        if not text:
            return []
        
        char_confidences = []
        
        for i, char in enumerate(text):
            conf = base_confidence
            
            # Adjust based on character type and position
            if i < 2:  # State code
                conf *= 0.95 if char.isalpha() else 0.7
            elif i < 4:  # District code
                conf *= 0.95 if char.isdigit() else 0.7
            elif i < 7:  # Series
                conf *= 0.9 if char.isalpha() else 0.75
            else:  # Registration number
                conf *= 0.95 if char.isdigit() else 0.7
            
            # Commonly confused characters get lower confidence
            if char in 'O0I1L8B5S':
                conf *= 0.85
            
            char_confidences.append(conf)
        
        return char_confidences

    def extract_with_validation(self, plate_image: np.ndarray) -> Dict:
        """Full extraction pipeline returning text, confidence, and validation flags."""
        start_time = time.time()
        
        try:
            # Validate input
            if plate_image is None or plate_image.size == 0:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'per_char_confidences': [],
                    'aggregate_confidence': 0.0,
                    'format_valid': False,
                    'state_valid': False,
                    'meets_confidence_threshold': False,
                    'should_write_to_db': False,
                    'processing_time': time.time() - start_time,
                    'error': 'Input image is empty or None'
                }
            
            # Extract text
            text, confidence = self.extract_text(plate_image)
            
            # Validate format and state
            format_valid = self.validate_plate_format(text)
            state_valid = self.validate_state_code(text) if format_valid else False
            
            # Calculate per-character confidences
            char_confidences = self.calculate_per_char_confidence(text, confidence)
            aggregate_confidence = self.calculate_aggregate_confidence(char_confidences)
            
            # Check threshold
            meets_threshold = aggregate_confidence >= self.confidence_threshold
            
            processing_time = time.time() - start_time
            
            return {
                'text': text,
                'confidence': confidence,
                'per_char_confidences': char_confidences,
                'aggregate_confidence': aggregate_confidence,
                'format_valid': format_valid,
                'state_valid': state_valid,
                'meets_confidence_threshold': meets_threshold,
                'should_write_to_db': meets_threshold and format_valid and state_valid,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in extraction pipeline: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'per_char_confidences': [],
                'aggregate_confidence': 0.0,
                'format_valid': False,
                'state_valid': False,
                'meets_confidence_threshold': False,
                'should_write_to_db': False,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def crop_plate_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                          padding_percent: float = 0.1) -> np.ndarray:
        """Crop plate region from image with padding."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate padding
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_percent)
        pad_y = int(height * padding_percent)
        
        # Apply padding with bounds checking
        h, w = image.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return image[y1:y2, x1:x2]

    def adjust_confidence_threshold(self, threshold: float):
        """Set minimum confidence for database writes (0.0-1.0)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold

    def cleanup(self):
        """Clean up resources."""
        if self.ocr is not None:
            del self.ocr
            self.ocr = None


def create_ocr_extractor(device: str = None, use_gpu: bool = True) -> OCRExtractor:
    """Factory function for OCRExtractor."""
    return OCRExtractor(device=device, use_gpu=use_gpu)
