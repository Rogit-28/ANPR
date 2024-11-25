"""
Integration test script for ANPR system with YOLOv11 license plate detection.

Tests:
1. Configuration loading (YOLOv11 defaults)
2. YOLODetector initialization and inference
3. OCR extraction
4. Full detection pipeline on sample images
5. API service startup check

Usage:
    python test_integration.py
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Default model path for tests (use project-local model)
DEFAULT_MODEL_PATH = "models/yolov11s-license-plate.pt"


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"{status} {name}")
    if details:
        print(f"       {details}")


def test_configuration():
    """Test configuration loading with YOLOv11 defaults."""
    print_header("1. Configuration Test")
    
    try:
        from anpr.config.config import get_config, ConfigManager, ModelsConfig
        
        # Test default ModelsConfig (not affected by user config file)
        default_models = ModelsConfig()
        
        print("   Default ModelsConfig values:")
        print(f"     yolo_model: {default_models.yolo_model}")
        print(f"     yolo_weights: {default_models.yolo_weights}")
        print(f"     yolo_confidence: {default_models.yolo_confidence}")
        print(f"     yolo_fp16: {default_models.yolo_fp16}")
        
        # Check that defaults are correct for YOLOv11
        checks = [
            ("Default YOLO model variant", default_models.yolo_model, "yolov11s"),
            ("Default YOLO weights path", "yolov11" in default_models.yolo_weights.lower(), True),
            ("Default confidence threshold", 0.3 <= default_models.yolo_confidence <= 0.6, True),
            ("Default FP16 enabled", default_models.yolo_fp16, True),
        ]
        
        all_passed = True
        for name, actual, expected in checks:
            if isinstance(expected, bool):
                passed = actual == expected
            else:
                passed = actual == expected
            print_result(name, passed, f"Expected: {expected}, Got: {actual}")
            all_passed = all_passed and passed
        
        # Also check if model file exists
        model_exists = Path(DEFAULT_MODEL_PATH).exists()
        print_result("Model file exists", model_exists, DEFAULT_MODEL_PATH)
        all_passed = all_passed and model_exists
        
        return all_passed
        
    except Exception as e:
        print_result("Configuration loading", False, str(e))
        return False


def test_yolo_detector():
    """Test YOLODetector initialization and inference."""
    print_header("2. YOLODetector Test")
    
    try:
        from anpr.core.yolo_detector import YOLODetector
        import numpy as np
        
        # Initialize detector with explicit model path (not from user config)
        print("   Initializing YOLODetector...")
        start = time.time()
        detector = YOLODetector(
            model_path=DEFAULT_MODEL_PATH,
            confidence_threshold=0.4,
            iou_threshold=0.45,
            fp16=True
        )
        init_time = time.time() - start
        
        print_result("Detector initialization", detector._model_loaded, f"Time: {init_time:.2f}s")
        
        if not detector._model_loaded:
            print("   ERROR: Model failed to load")
            return False
        
        # Check model info
        info = detector.get_model_info()
        print(f"\n   Model Info:")
        print(f"     Path: {info['model_path']}")
        print(f"     Device: {info['device']}")
        print(f"     FP16: {info['fp16_enabled']}")
        
        # Test with dummy image
        print("\n   Testing inference on dummy image...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        start = time.time()
        detections = detector.detect(dummy_img)
        infer_time = time.time() - start
        
        print_result("Dummy inference", True, f"Time: {infer_time*1000:.1f}ms, Detections: {len(detections)}")
        
        # Test on sample image if available
        sample_images = list(Path("data/samples/plates").glob("*.jpg"))
        if sample_images:
            print(f"\n   Testing on {len(sample_images)} sample images...")
            total_plates = 0
            total_time = 0
            
            for img_path in sample_images:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    start = time.time()
                    plates = detector.detect_license_plate(img)
                    total_time += time.time() - start
                    total_plates += len(plates)
            
            avg_time = (total_time / len(sample_images)) * 1000
            print_result(
                "Sample image detection", 
                total_plates > 0, 
                f"Plates: {total_plates}/{len(sample_images)} images, Avg: {avg_time:.1f}ms"
            )
        
        # Benchmark
        print("\n   Running performance benchmark...")
        benchmark = detector.benchmark_performance()
        print(f"     FPS: {benchmark['fps']}")
        print(f"     Avg inference: {benchmark['avg_inference_time']*1000:.1f}ms")
        print(f"     Memory: {benchmark['memory_peak_mb']:.0f}MB")
        
        return True
        
    except Exception as e:
        import traceback
        print_result("YOLODetector test", False, str(e))
        traceback.print_exc()
        return False


def test_ocr_extractor():
    """Test OCR extraction."""
    print_header("3. OCR Extractor Test")
    
    try:
        from anpr.core.ocr_extractor import OCRExtractor
        import numpy as np
        import cv2
        
        # Initialize OCR
        print("   Initializing OCRExtractor...")
        start = time.time()
        ocr = OCRExtractor(use_gpu=True)
        init_time = time.time() - start
        
        print_result("OCR initialization", True, f"Time: {init_time:.2f}s")
        
        # Test on sample plate images
        sample_images = list(Path("data/samples/plates").glob("*.jpg"))
        if sample_images:
            print(f"\n   Testing OCR on {len(sample_images)} sample images...")
            
            for img_path in sample_images[:3]:  # Test first 3
                img = cv2.imread(str(img_path))
                if img is not None:
                    result = ocr.extract_with_validation(img)
                    text = result.get('text', '')
                    conf = result.get('aggregate_confidence', 0)
                    is_valid = result.get('is_valid_indian_plate', False)
                    
                    print(f"     {img_path.name}: '{text}' (conf: {conf:.2f}, valid: {is_valid})")
        
        return True
        
    except Exception as e:
        import traceback
        print_result("OCR test", False, str(e))
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test full detection pipeline on a sample image."""
    print_header("4. Full Pipeline Test")
    
    try:
        from anpr.core.yolo_detector import YOLODetector
        from anpr.core.ocr_extractor import OCRExtractor
        import cv2
        
        # Initialize components with explicit paths (not from user config)
        print("   Initializing pipeline components...")
        detector = YOLODetector(
            model_path=DEFAULT_MODEL_PATH,
            confidence_threshold=0.4,
        )
        ocr = OCRExtractor(use_gpu=True)
        
        if not detector._model_loaded:
            print_result("Pipeline test", False, "Model failed to load")
            return False
        
        # Find sample images
        sample_images = list(Path("data/samples/plates").glob("*.jpg"))
        if not sample_images:
            print_result("Pipeline test", False, "No sample images found")
            return False
        
        print(f"\n   Running full pipeline on {len(sample_images)} images...")
        results = []
        
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            start = time.time()
            
            # Step 1: Detect plates
            plates = detector.detect_license_plate(img)
            
            for plate in plates:
                # Step 2: Crop plate region
                x1, y1, x2, y2 = map(int, plate['bbox'])
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                plate_crop = img[y1:y2, x1:x2]
                
                # Step 3: OCR
                ocr_result = ocr.extract_with_validation(plate_crop)
                
                pipeline_time = time.time() - start
                
                results.append({
                    'image': img_path.name,
                    'text': ocr_result.get('text', ''),
                    'det_conf': plate.get('confidence', 0),
                    'ocr_conf': ocr_result.get('aggregate_confidence', 0),
                    'time_ms': pipeline_time * 1000
                })
        
        # Print results
        print("\n   Pipeline Results:")
        print("   " + "-" * 56)
        print(f"   {'Image':<20} {'Plate Text':<15} {'Det%':>6} {'OCR%':>6} {'Time':>8}")
        print("   " + "-" * 56)
        
        for r in results:
            print(f"   {r['image']:<20} {r['text']:<15} {r['det_conf']*100:>5.1f}% {r['ocr_conf']*100:>5.1f}% {r['time_ms']:>6.1f}ms")
        
        print("   " + "-" * 56)
        
        success_rate = len([r for r in results if r['text']]) / len(results) if results else 0
        print_result(
            "Full pipeline", 
            success_rate > 0.5, 
            f"Success rate: {success_rate*100:.0f}% ({len([r for r in results if r['text']])}/{len(results)})"
        )
        
        return success_rate > 0.5
        
    except Exception as e:
        import traceback
        print_result("Pipeline test", False, str(e))
        traceback.print_exc()
        return False


def test_api_imports():
    """Test API service can be imported."""
    print_header("5. API Service Import Test")
    
    try:
        from anpr.api.service import app
        from anpr.api.models import DetectionResponse, VideoProcessRequest
        
        print_result("API service import", True, "FastAPI app loaded")
        print_result("API models import", True, "Pydantic models loaded")
        
        # Check endpoints
        routes = [route.path for route in app.routes]
        print(f"\n   Available endpoints: {len(routes)}")
        for route in routes[:10]:  # Show first 10
            print(f"     {route}")
        
        return True
        
    except Exception as e:
        import traceback
        print_result("API import test", False, str(e))
        traceback.print_exc()
        return False


def test_database():
    """Test database connection."""
    print_header("6. Database Test")
    
    try:
        from anpr.database.manager import DatabaseManager
        
        db = DatabaseManager()
        connected = db.test_connection()
        
        print_result("Database connection", connected)
        
        return connected
        
    except Exception as e:
        import traceback
        print_result("Database test", False, str(e))
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  ANPR System Integration Tests (YOLOv11)".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    results = {}
    
    # Run tests
    results['configuration'] = test_configuration()
    results['yolo_detector'] = test_yolo_detector()
    results['ocr_extractor'] = test_ocr_extractor()
    results['full_pipeline'] = test_full_pipeline()
    results['api_imports'] = test_api_imports()
    results['database'] = test_database()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  All tests passed! System is ready.")
        return 0
    else:
        print(f"\n  {total - passed} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
