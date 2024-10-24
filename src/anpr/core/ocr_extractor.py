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
