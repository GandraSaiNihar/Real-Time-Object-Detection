"""
Tests for utility functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import (
    calculate_iou,
    calculate_center_distance,
    calculate_fps,
    validate_bbox,
    non_max_suppression
)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Test identical boxes
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        iou = calculate_iou(box1, box2)
        assert iou == 1.0
        
        # Test non-overlapping boxes
        box1 = [100, 100, 200, 200]
        box2 = [300, 300, 400, 400]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0
        
        # Test partially overlapping boxes
        box1 = [100, 100, 200, 200]
        box2 = [150, 150, 250, 250]
        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1
        
        # Test edge case: zero area boxes
        box1 = [100, 100, 100, 100]
        box2 = [100, 100, 100, 100]
        iou = calculate_iou(box1, box2)
        assert iou == 0.0  # Zero area boxes should have IoU of 0
    
    def test_calculate_center_distance(self):
        """Test center distance calculation."""
        # Test identical centers
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        distance = calculate_center_distance(box1, box2)
        assert distance == 0.0
        
        # Test different centers
        box1 = [0, 0, 100, 100]
        box2 = [100, 100, 200, 200]
        distance = calculate_center_distance(box1, box2)
        expected_distance = np.sqrt(100**2 + 100**2)  # Distance between (50,50) and (150,150)
        assert abs(distance - expected_distance) < 1e-6
    
    def test_calculate_fps(self):
        """Test FPS calculation."""
        import time
        
        # Test with zero elapsed time
        fps = calculate_fps(time.time(), 0)
        assert fps == 0.0
        
        # Test with some frames
        start_time = time.time() - 1.0  # 1 second ago
        fps = calculate_fps(start_time, 30)  # 30 frames in 1 second
        assert abs(fps - 30.0) < 0.1
    
    def test_validate_bbox(self):
        """Test bounding box validation."""
        frame_shape = (480, 640, 3)
        
        # Test valid bbox
        valid_bbox = [100, 100, 200, 200]
        assert validate_bbox(valid_bbox, frame_shape) == True
        
        # Test invalid bbox (out of bounds)
        invalid_bbox = [100, 100, 700, 500]  # x2 > width, y2 > height
        assert validate_bbox(invalid_bbox, frame_shape) == False
        
        # Test invalid bbox (negative coordinates)
        invalid_bbox = [-10, 100, 200, 200]
        assert validate_bbox(invalid_bbox, frame_shape) == False
        
        # Test invalid bbox (wrong size)
        invalid_bbox = [100, 100, 200, 200]  # Valid size
        assert validate_bbox(invalid_bbox, frame_shape) == True
        
        # Test invalid bbox (too small)
        invalid_bbox = [100, 100, 102, 102]  # Too small
        assert validate_bbox(invalid_bbox, frame_shape) == False
        
        # Test invalid bbox (wrong order)
        invalid_bbox = [200, 200, 100, 100]  # x2 < x1, y2 < y1
        assert validate_bbox(invalid_bbox, frame_shape) == False
        
        # Test invalid bbox (wrong length)
        invalid_bbox = [100, 100, 200]  # Only 3 elements
        assert validate_bbox(invalid_bbox, frame_shape) == False
    
    def test_non_max_suppression(self):
        """Test non-maximum suppression."""
        # Test with no boxes
        boxes = []
        scores = []
        keep_indices = non_max_suppression(boxes, scores)
        assert keep_indices == []
        
        # Test with single box
        boxes = [[100, 100, 200, 200]]
        scores = [0.9]
        keep_indices = non_max_suppression(boxes, scores)
        assert keep_indices == [0]
        
        # Test with non-overlapping boxes
        boxes = [
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ]
        scores = [0.9, 0.8]
        keep_indices = non_max_suppression(boxes, scores)
        assert set(keep_indices) == {0, 1}
        
        # Test with overlapping boxes (should keep highest score)
        boxes = [
            [100, 100, 200, 200],  # Higher score
            [120, 120, 220, 220]   # Lower score, overlaps more
        ]
        scores = [0.9, 0.7]
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.3)
        assert keep_indices == [0]  # Should keep only the first one
        
        # Test with multiple overlapping boxes
        boxes = [
            [100, 100, 200, 200],  # Highest score
            [120, 120, 220, 220],  # Medium score, overlaps with first
            [300, 300, 400, 400]   # Lower score, no overlap
        ]
        scores = [0.9, 0.7, 0.6]
        keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.3)
        assert set(keep_indices) == {0, 2}  # Should keep first and third


if __name__ == "__main__":
    pytest.main([__file__])
