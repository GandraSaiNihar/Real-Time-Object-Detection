#!/usr/bin/env python3
"""
Object Tracking Demo
Example showing object detection and tracking capabilities.
"""

import sys
import cv2
import time
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.utils import setup_logger


def main():
    """Run tracking demo."""
    logger = setup_logger("TrackingDemo")
    
    # Initialize detector and tracker
    logger.info("Initializing detector and tracker...")
    detector = YOLODetector(device="cpu")
    tracker = MultiObjectTracker(detector, tracker_type="bytetrack")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Starting tracking demo. Press 'q' to quit, 'r' to reset tracker.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            continue
        
        frame_count += 1
        
        # Perform detection and tracking
        annotated_frame, tracked_objects = tracker.track(frame)
        
        # Add information overlay
        fps = detector.get_fps()
        tracking_fps = tracker.tracker.tracking_fps
        
        cv2.putText(annotated_frame, f"Detection FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Tracking FPS: {tracking_fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Tracked Objects: {len(tracked_objects)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add tracking statistics
        stats = tracker.tracker.get_tracking_stats()
        cv2.putText(annotated_frame, f"Active Tracks: {stats['active_tracks']}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Total Tracks: {stats['total_tracks_created']}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Object Tracking Demo", annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.tracker.reset()
            logger.info("Tracker reset")
        
        # Log progress every 100 frames
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / elapsed_time
            logger.info(f"Processed {frame_count} frames, avg FPS: {avg_fps:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    logger.info(f"Demo completed. Processed {frame_count} frames in {elapsed_time:.2f}s (avg FPS: {avg_fps:.2f})")


if __name__ == "__main__":
    main()
