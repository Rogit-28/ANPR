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
        
    def load_model(self) -> bool:
        """Load the YOLO model with GPU memory management and FP16 support."""
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            logger.info(f"Target device: {self.device}, FP16: {self.fp16}")
            
            # Validate model file exists before loading
            model_path = Path(self.model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not self._validate_model_file(self.model_path):
                raise FileNotFoundError(f"Invalid model file: {self.model_path}")
            
            # Load the model
            self.model = YOLO(self.model_path)
            
            # GPU setup - must happen before any inference
            if 'cuda' in self.device:
                # Check available GPU memory
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
                
                # Check if there's enough memory for the model
                if not self._check_gpu_memory_sufficiency(gpu_memory_gb):
                    logger.warning("Insufficient GPU memory, falling back to CPU")
                    self.device = 'cpu'
                    self.fp16 = False
                    return self.load_model()
                
                # Explicitly move model to GPU
                self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")
                
                # Attempt to enable FP16 if requested and available
                if self.fp16:
                    try:
                        # Check if GPU supports FP16
                        if torch.cuda.get_device_capability(0)[0] >= 5:
                            self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers for speed
                            logger.info("Model fused for better performance")
                        else:
                            logger.warning("GPU does not support FP16, disabling...")
                            self.fp16 = False
                    except Exception as e:
                        logger.warning(f"Could not optimize model for FP16: {e}")
                        self.fp16 = False
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Test inference to ensure model is loaded properly on the target device
            dummy_input = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            _ = self.model(
                dummy_input, 
                imgsz=self.input_size, 
                conf=0.1, 
                iou=0.1, 
                verbose=False,
                device=self.device,
                half=self.fp16 if 'cuda' in self.device else False
            )
            
            self._model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except MemoryError as e:
            logger.error(f"GPU memory insufficient: {e}")
            # Try fallback to CPU only once
            if self.device != 'cpu' and not self._fallback_attempted:
                logger.info("Attempting fallback to CPU...")
                self._fallback_attempted = True
                self.device = 'cpu'
                self.fp16 = False
                return self.load_model()
            return False
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try fallback to CPU only once
            if self.device != 'cpu' and not self._fallback_attempted:
                logger.info("Attempting fallback to CPU...")
                self._fallback_attempted = True
                self.device = 'cpu'
                self.fp16 = False
                return self.load_model()
            return False
    
    def is_model_loaded(self) -> bool:
        return self._model_loaded
    
    def _validate_model_file(self, model_path: str) -> bool:
        """Check model file exists and has valid extension/size."""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error(f"Model file does not exist: {model_path}")
                return False
            
            if not path.is_file():
                logger.error(f"Model path is not a file: {model_path}")
                return False
            
            # Check if file has appropriate extension
            valid_extensions = ['.pt', '.onnx', '.engine', '.trt', '.safetensors']
            if path.suffix.lower() not in valid_extensions:
                logger.warning(f"Model file has unexpected extension: {path.suffix}")
            
            # Check file size (should be reasonably large for a model)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.1:  # Less than 0.1 MB is suspiciously small
                logger.warning(f"Model file seems unusually small: {file_size_mb:.2f} MB")
            
            return True
        except Exception as e:
            logger.error(f"Error validating model file: {e}")
            return False
    
    def _check_gpu_memory_sufficiency(self, gpu_memory_gb: float) -> bool:
        """Check if GPU has enough VRAM for the model variant (YOLOv8/v11 n/s/m/l/x)."""
        model_path_lower = self.model_path.lower()
        
        # YOLOv11 model memory requirements (fine-tuned for license plates)
        if 'yolov11n' in model_path_lower or 'yolo11n' in model_path_lower:
            required_memory_gb = 1.0  # YOLOv11n nano
        elif 'yolov11s' in model_path_lower or 'yolo11s' in model_path_lower:
            required_memory_gb = 1.5  # YOLOv11s small (our default)
        elif 'yolov11m' in model_path_lower or 'yolo11m' in model_path_lower:
            required_memory_gb = 2.5  # YOLOv11m medium
        elif 'yolov11l' in model_path_lower or 'yolo11l' in model_path_lower:
            required_memory_gb = 3.5  # YOLOv11l large
        elif 'yolov11x' in model_path_lower or 'yolo11x' in model_path_lower:
            required_memory_gb = 5.0  # YOLOv11x extra-large
        else:
            # Default conservative estimate for unknown models
            required_memory_gb = 1.5
        
        if gpu_memory_gb < required_memory_gb:
            logger.warning(f"Insufficient GPU memory: {gpu_memory_gb:.2f}GB available, "
                          f"{required_memory_gb}GB required")
            return False
        
        logger.info(f"GPU memory check passed: {gpu_memory_gb:.2f}GB available, "
                   f"{required_memory_gb}GB required for {self.model_path}")
        return True
    
    def detect(
        self,
        image: Union[np.ndarray, str],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> List[dict]:
        """Run detection on image, returning list of {bbox, confidence, class_id, class_name}."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use provided thresholds or defaults
        conf_thresh = conf_threshold if conf_threshold is not None else self.confidence_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        try:
            # Validate input image
            if isinstance(image, str):
                if not Path(image).exists():
                    raise FileNotFoundError(f"Image file does not exist: {image}")
                # Load image to validate
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"Could not load image from path: {image}")
            elif not isinstance(image, np.ndarray):
                raise TypeError(f"Image must be numpy array or string path, got {type(image)}")
            
            # Validate image dimensions
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape: {image.shape}")
            
            # Run inference
            results = self.model(
                image,
                imgsz=self.input_size,
                conf=conf_thresh,
                iou=iou_thresh,
                half=self.fp16 if 'cuda' in self.device else False,
                verbose=False,
                device=self.device
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2]
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': xyxy.tolist(),
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': result.names[cls] if hasattr(result, 'names') and result.names else f'class_{cls}'
                        }
                        detections.append(detection)
            
            return detections
            
        except MemoryError as e:
            logger.error(f"GPU memory error during detection: {e}")
            # Clear cache and try again on CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
