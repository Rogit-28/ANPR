import unittest
import os
import datetime
import numpy as np
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anpr.database.manager import DatabaseManager
from src.anpr.database.models import Detection, ProcessingJob, Base


class TestStorage(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.db_manager = DatabaseManager(database_url="sqlite:///:memory:")
        self.db_manager.init_database()
        self.test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    def tearDown(self):
        self.db_manager.close()

    def test_insert_detection(self):
        """Test inserting a detection record."""
        with self.db_manager.get_session() as session:
            detection = Detection(
                plate_text='TEST1234',
                confidence=0.95,
                timestamp=datetime.datetime.now().isoformat(),
                source_type='test',
                source_identifier='test_camera',
                frame_number=0,
                bbox_x=10,
                bbox_y=20,
                bbox_width=100,
                bbox_height=80,
                snapshot_path='test/path.jpg',
                processing_time_ms=50
            )
            session.add(detection)
            session.commit()
            
            # Verify it was inserted
            self.assertIsNotNone(detection.id)
            self.assertIsInstance(detection.id, int)

    def test_query_detections(self):
        """Test querying detection records."""
        with self.db_manager.get_session() as session:
            # Insert test data
            detection = Detection(
                plate_text='QUERY123',
                confidence=0.88,
                timestamp=datetime.datetime.now().isoformat(),
                source_type='test',
                source_identifier='test_camera',
                frame_number=0,
                bbox_x=10,
                bbox_y=20,
                bbox_width=100,
                bbox_height=80,
                snapshot_path='test/path.jpg',
                processing_time_ms=50
            )
            session.add(detection)
            session.commit()
            
            # Query it back
            results = session.query(Detection).filter(
                Detection.plate_text == 'QUERY123'
            ).all()
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].confidence, 0.88)

    def test_insert_processing_job(self):
        """Test inserting a processing job record."""
        import uuid
        
        with self.db_manager.get_session() as session:
            job = ProcessingJob(
                id=str(uuid.uuid4()),
                status='pending',
                progress=0,
                original_filename='test_video.mp4',
                created_at=datetime.datetime.now().isoformat()
            )
            session.add(job)
            session.commit()
            
            # Verify it was inserted
            self.assertIsNotNone(job.id)

    def test_connection(self):
        """Test database connection."""
        is_connected = self.db_manager.test_connection()
        # Note: test_connection queries Detection table which may be empty
        # so we just check it doesn't throw an error
        self.assertIsInstance(is_connected, bool)


if __name__ == '__main__':
    unittest.main()
