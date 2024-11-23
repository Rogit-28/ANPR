import unittest
import os
import cv2
import numpy as np
import sys
import asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anpr.core.pipeline_manager import PipelineManager
from src.anpr.database.manager import DatabaseManager


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy video file for testing
        self.video_path = "pipeline_test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 1, (640, 480))
        # Create a few frames with some noise to ensure the pipeline runs
        for _ in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # Use an in-memory SQLite database for testing
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.init_database()

    def tearDown(self):
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        self.db_manager.close()

    def test_pipeline_manager_initialization(self):
        """Test that PipelineManager initializes without errors."""
        manager = PipelineManager()
        self.assertIsNotNone(manager)

    def test_pipeline_components_initialization(self):
        """Test initializing pipeline components."""
        async def run_test():
            manager = PipelineManager()
            await manager.initialize_components()
            self.assertIsNotNone(manager.yolo_detector)
            self.assertIsNotNone(manager.ocr_extractor)
            await manager.cleanup()
        
        asyncio.run(run_test())

    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        manager = PipelineManager()
        metrics = manager.get_performance_metrics()
        self.assertIsInstance(metrics, dict)

    def test_get_memory_metrics(self):
        """Test getting memory metrics."""
        manager = PipelineManager()
        metrics = manager.get_memory_metrics()
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    unittest.main()
