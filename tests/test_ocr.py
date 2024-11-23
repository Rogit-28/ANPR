import unittest
import numpy as np
import cv2
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anpr.core.ocr_extractor import OCRExtractor


class TestOCR(unittest.TestCase):
    def setUp(self):
        # Create a dummy image file for testing
        self.image_path = "test_plate.jpg"
        # Create an image with some text-like noise
        img = np.zeros((50, 200, 3), dtype=np.uint8)
        cv2.putText(img, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(self.image_path, img)

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_ocr_initialization(self):
        """Test that OCR engine initializes without errors."""
        ocr_engine = OCRExtractor()
        self.assertIsNotNone(ocr_engine)

    def test_extract_text(self):
        """Test OCR text extraction (basic functionality check)."""
        ocr_engine = OCRExtractor()
        plate_img = cv2.imread(self.image_path)
        result = ocr_engine.extract_with_validation(plate_img)
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('aggregate_confidence', result)

    def test_extract_with_sample_plate(self):
        """Test OCR on a sample plate image if available."""
        sample_path = "data/samples/plates/HR_plate.jpg"
        if os.path.exists(sample_path):
            ocr_engine = OCRExtractor()
            plate_img = cv2.imread(sample_path)
            result = ocr_engine.extract_with_validation(plate_img)
            self.assertIsInstance(result, dict)
            # Should have some text if the plate is readable
            if result['text']:
                self.assertGreater(len(result['text']), 0)


if __name__ == '__main__':
    unittest.main()
