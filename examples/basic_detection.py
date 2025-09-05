#!/usr/bin/env python3
"""
Basic Object Detection Example
Simple example showing how to use the YOLO detector for object detection.
"""

import sys
import cv2
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.detector import YOLODetector
from src.utils import setup_logger


def main():
    """Run basic detection example."""
    logger = setup_logger("BasicDetection")
    
    # Initialize detector
    logger.info("Initializing YOLO detector...")
    detector = YOLODetector(device="cpu")  # Use CPU for compatibility
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Starting detection. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            continue
        
        # Perform detection
        detections = detector.detect(frame)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Add FPS info
        fps = detector.get_fps()
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Basic Object Detection", annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Detection completed")


if __name__ == "__main__":
    main()
