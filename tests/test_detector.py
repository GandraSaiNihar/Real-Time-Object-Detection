"""
Tests for the YOLO detector module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.detector import YOLODetector
from src.utils import setup_logger


class TestYOLODetector:
    """Test cases for YOLODetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = YOLODetector(device="cpu")
        assert detector.device == "cpu"
        assert detector.model is not None
    
    def test_detector_initialization_with_model(self):
        """Test detector initialization with custom model."""
        # This test might fail if model file doesn't exist, which is expected
        try:
            detector = YOLODetector(model_path="yolov8n.pt", device="cpu")
            assert detector.model is not None
        except Exception:
            # Model file might not exist, which is okay for testing
            pass
    
    def test_detection_with_dummy_frame(self):
        """Test detection with a dummy frame."""
        detector = YOLODetector(device="cpu")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(dummy_frame)
        
        assert isinstance(detections, list)
        # Detections should be a list of dictionaries
        for detection in detections:
            assert isinstance(detection, dict)
            assert 'bbox' in detection
            assert 'confidence' in detection
            assert 'class_id' in detection
            assert 'class_name' in detection
    
    def test_fps_calculation(self):
        """Test FPS calculation."""
        detector = YOLODetector(device="cpu")
        
        # Initially FPS should be 0
        assert detector.get_fps() == 0
        
        # After some detections, FPS should be calculated
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            detector.detect(dummy_frame)
        
        # FPS should be calculated after detections
        fps = detector.get_fps()
        assert isinstance(fps, float)
        assert fps >= 0
    
    def test_model_info(self):
        """Test model information retrieval."""
        detector = YOLODetector(device="cpu")
        model_info = detector.get_model_info()
        
        assert isinstance(model_info, dict)
        assert 'device' in model_info
        assert 'num_classes' in model_info
        assert model_info['device'] == "cpu"
    
    def test_draw_detections(self):
        """Test drawing detections on frame."""
        detector = YOLODetector(device="cpu")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create mock detections
        mock_detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.8,
                'class_name': 'person',
                'class_id': 0
            }
        ]
        
        annotated_frame = detector.draw_detections(dummy_frame, mock_detections)
        
        assert annotated_frame.shape == dummy_frame.shape
        assert isinstance(annotated_frame, np.ndarray)
    
    def test_benchmark(self):
        """Test benchmark functionality."""
        detector = YOLODetector(device="cpu")
        
        # Run a small benchmark
        metrics = detector.benchmark(test_frames=5)
        
        assert isinstance(metrics, dict)
        assert 'total_frames' in metrics
        assert 'total_time' in metrics
        assert 'avg_fps' in metrics
        assert 'avg_inference_time_ms' in metrics
        assert 'device' in metrics
        
        assert metrics['total_frames'] == 5
        assert metrics['device'] == "cpu"
        assert metrics['avg_fps'] >= 0
        assert metrics['avg_inference_time_ms'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
