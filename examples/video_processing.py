#!/usr/bin/env python3
"""
Video Processing Example
Example showing how to process video files with detection and tracking.
"""

import sys
import cv2
import argparse
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.utils import setup_logger, create_video_writer


def process_video(input_path: str, output_path: str, model_path: str = None, device: str = "cpu"):
    """
    Process video file with object detection and tracking.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        model_path: Path to YOLO model
        device: Device to run on
    """
    logger = setup_logger("VideoProcessing")
    
    # Initialize detector and tracker
    logger.info("Initializing detector and tracker...")
    detector = YOLODetector(model_path=model_path, device=device)
    tracker = MultiObjectTracker(detector, tracker_type="bytetrack")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open input video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Input video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    writer = create_video_writer(output_path, fps, width, height)
    if not writer.isOpened():
        logger.error(f"Failed to create output video writer: {output_path}")
        return
    
    logger.info("Starting video processing...")
    
    frame_count = 0
    processed_detections = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        annotated_frame, tracked_objects = tracker.track(frame)
        
        # Add progress info
        progress = (frame_count / total_frames) * 100
        cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Objects: {len(tracked_objects)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame
        writer.write(annotated_frame)
        processed_detections += len(tracked_objects)
        
        # Log progress
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    writer.release()
    
    logger.info(f"Video processing completed!")
    logger.info(f"Processed {frame_count} frames")
    logger.info(f"Total detections: {processed_detections}")
    logger.info(f"Output saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process video with object detection and tracking")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--model", "-m", help="YOLO model path")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    process_video(args.input, args.output, args.model, args.device)


if __name__ == "__main__":
    main()
