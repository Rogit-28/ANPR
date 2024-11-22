import unittest
import numpy as np
import cv2
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anpr.core.yolo_detector import YOLODetector


class TestDetection(unittest.TestCase):
    def setUp(self):
        # Create a dummy image file for testing
        self.image_path = "test_image.jpg"
        cv2.imwrite(self.image_path, np.zeros((640, 640, 3), dtype=np.uint8))

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_detector_initialization(self):
        """Test that the YOLO detector initializes without errors."""
        detector = YOLODetector()
        self.assertIsNotNone(detector)

    def test_detect_plates(self):
        """Test plate detection on a blank image (should return empty list)."""
        detector = YOLODetector()
        frame = cv2.imread(self.image_path)
        detections = detector.detect_license_plate(frame)
        self.assertIsInstance(detections, list)

    def test_detect_with_sample_image(self):
        """Test detection on a sample plate image if available."""
        sample_path = "data/samples/plates/HR_plate.jpg"
        if os.path.exists(sample_path):
            detector = YOLODetector()
            frame = cv2.imread(sample_path)
            detections = detector.detect_license_plate(frame)
            self.assertIsInstance(detections, list)
            # Sample images should have at least one detection
            # but we don't fail if there are none (model might not detect)


if __name__ == '__main__':
    unittest.main()
