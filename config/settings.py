# Video Processing
VIDEO_FPS_EXTRACT = 5  # Extract every Nth frame
VIDEO_CODEC = "h264"
VIDEO_BUFFER_SIZE = 30  # Frame buffer

# Detection
DETECTION_CONF_THRESHOLD = 0.5  # YOLO confidence
DETECTION_IOU_THRESHOLD = 0.45
MODEL_INPUT_SIZE = 640  # YOLO input resolution
USE_GPU = True  # Auto-detect if available

# OCR
OCR_ENGINE = "easyocr"  # Options: easyocr, paddleocr
OCR_LANGUAGES = ["en"]  # Expandable
OCR_MIN_CONFIDENCE = 0.93

# Tracking
DUPLICATE_TIME_WINDOW = 3.0  # Seconds
SPATIAL_THRESHOLD = 50  # Pixels

# Storage
DB_URL = "postgresql://user:pass@localhost:5432/anpr"
IMAGE_STORAGE_PATH = "./data/output/images"
LOG_RETENTION_DAYS = 30
IMAGE_COMPRESSION_DAYS = 7

# Performance
MAX_LATENCY_MS = 500
ENABLE_METRICS = True
