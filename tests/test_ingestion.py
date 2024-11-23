import unittest
import os
import cv2
import numpy as np
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anpr.core.video_handler import get_video_metadata, VideoHandler


class TestIngestion(unittest.TestCase):
    def setUp(self):
        # Create a dummy video file for testing
        self.video_path = "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 1, (10, 10))
        for _ in range(5):
            out.write(np.zeros((10, 10, 3), dtype=np.uint8))
        out.release()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)

    def test_get_video_metadata(self):
        """Test getting video metadata."""
        metadata = get_video_metadata(self.video_path)
        self.assertIsNotNone(metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
        self.assertIn('fps', metadata)
        self.assertIn('frame_count', metadata)
        self.assertEqual(metadata['width'], 10)
        self.assertEqual(metadata['height'], 10)

    def test_video_handler_initialization(self):
        """Test VideoHandler initialization."""
        handler = VideoHandler()
        self.assertIsNotNone(handler)

    def test_open_video(self):
        """Test opening a video file."""
        cap = cv2.VideoCapture(self.video_path)
        self.assertIsNotNone(cap)
        self.assertTrue(cap.isOpened())
        cap.release()

    def test_read_frames(self):
        """Test reading frames from video."""
        cap = cv2.VideoCapture(self.video_path)
        frames_read = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
        cap.release()
        self.assertEqual(frames_read, 5)


if __name__ == '__main__':
    unittest.main()
