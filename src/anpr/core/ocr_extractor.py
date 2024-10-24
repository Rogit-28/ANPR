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
        
        logger.info("OCRExtractor initialized")
