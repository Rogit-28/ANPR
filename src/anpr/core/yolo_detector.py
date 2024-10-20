import torch
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, Union, List
import cv2
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv11-based license plate detector using morsetechlab/yolov11-license-plate-detection."""
    
    # License plate class names the model might use
    PLATE_CLASS_NAMES = {'license-plate', 'license_plate', 'plate', 'number_plate', 'License Plate', 'License-Plate'}
    
    def __init__(
        self,
        model_path: str = "models/yolov11s-license-plate.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 1280,
        device: Optional[str] = None,
        fp16: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.fp16 = fp16
        
        # Determine device automatically if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'  # Explicitly use first GPU
                # Check CUDA capability for FP16 support (compute capability >= 5.3)
                cuda_device = torch.cuda.get_device_capability()
                if cuda_device[0] >= 5 and cuda_device[1] >= 3:
                    self.fp16 = fp16
                else:
                    self.fp16 = False  # Disable FP16 if compute capability is insufficient
            else:
                self.device = 'cpu'
                self.fp16 = False
        else:
            self.device = device
            
        # Initialize model variables
        self.model = None
        self.batch_size = 1  # As specified in requirements
        self._model_loaded = False
        self._fallback_attempted = False  # Prevent infinite recursion on fallback
        
        # Load the model
        self.load_model()
