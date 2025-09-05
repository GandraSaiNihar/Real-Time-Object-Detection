#!/usr/bin/env python3
"""
Real-Time Object Detection & Tracking System
Main application script for live object detection and tracking using YOLOv8 and OpenCV.
"""

import cv2
import argparse
import time
import sys
from pathlib import Path
import signal
import threading
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.utils import setup_logger, PerformanceMonitor, create_performance_plot
from src.config import VIDEO_CONFIG, YOLO_CONFIG, TRACKING_CONFIG


class RealTimeDetectionApp:
    """
    Main application class for real-time object detection and tracking.
    """
    
    def __init__(self, args):
        """
        Initialize the application.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.logger = setup_logger("RealTimeDetectionApp")
        
        # Initialize components
        self.detector = None
        self.tracker = None
        self.performance_monitor = PerformanceMonitor()
        
        # Video capture and writer
        self.cap = None
        self.video_writer = None
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Real-Time Object Detection & Tracking App initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _setup_video_capture(self):
        """Setup video capture from input source."""
        input_source = self.args.input if self.args.input else VIDEO_CONFIG["input_source"]
        
        self.logger.info(f"Setting up video capture from: {input_source}")
        
        try:
            self.cap = cv2.VideoCapture(input_source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {input_source}")
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG["display_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG["display_height"])
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_CONFIG["output_fps"])
            
            # Get actual properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Video capture setup: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
            
        except Exception as e:
            self.logger.error(f"Failed to setup video capture: {e}")
            raise
    
    def _setup_video_writer(self):
        """Setup video writer for output."""
        if not self.args.save_output:
            return
        
        output_path = self.args.output if self.args.output else "output/detection_output.mp4"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Setting up video writer: {output_path}")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["output_codec"])
            self.video_writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                VIDEO_CONFIG["output_fps"], 
                (self.frame_width, self.frame_height)
            )
            
            if not self.video_writer.isOpened():
                raise RuntimeError(f"Failed to create video writer: {output_path}")
            
            self.logger.info("Video writer setup successful")
            
        except Exception as e:
            self.logger.error(f"Failed to setup video writer: {e}")
            raise
    
    def _setup_detector(self):
        """Setup YOLO detector."""
        model_path = self.args.model if self.args.model else YOLO_CONFIG["model_size"]
        device = self.args.device if self.args.device else YOLO_CONFIG["device"]
        
        self.logger.info(f"Setting up detector: {model_path} on {device}")
        
        try:
            self.detector = YOLODetector(model_path=model_path, device=device)
            self.logger.info("Detector setup successful")
            
        except Exception as e:
            self.logger.error(f"Failed to setup detector: {e}")
            raise
    
    def _setup_tracker(self):
        """Setup object tracker."""
        tracker_type = self.args.tracker if self.args.tracker else TRACKING_CONFIG["tracker_type"]
        
        self.logger.info(f"Setting up tracker: {tracker_type}")
        
        try:
            self.tracker = MultiObjectTracker(self.detector, tracker_type=tracker_type)
            self.logger.info("Tracker setup successful")
            
        except Exception as e:
            self.logger.error(f"Failed to setup tracker: {e}")
            raise
    
    def _draw_info_overlay(self, frame, tracked_objects):
        """Draw information overlay on frame."""
        # FPS
        if VIDEO_CONFIG["show_fps"]:
            fps = self.detector.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Object count
        cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Performance info
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time
            cv2.putText(frame, f"Avg FPS: {avg_fps:.1f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        if self.paused:
            cv2.putText(frame, "PAUSED - Press SPACE to resume", 
                       (self.frame_width // 2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to pause, Q to quit", 
                       (self.frame_width // 2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input."""
        if key == ord('q') or key == ord('Q'):
            self.running = False
        elif key == ord(' '):  # Space bar
            self.paused = not self.paused
            self.logger.info(f"Video {'paused' if self.paused else 'resumed'}")
        elif key == ord('r') or key == ord('R'):
            if self.tracker:
                self.tracker.tracker.reset()
                self.logger.info("Tracker reset")
        elif key == ord('s') or key == ord('S'):
            self._save_screenshot()
    
    def _save_screenshot(self):
        """Save current frame as screenshot."""
        if hasattr(self, 'current_frame'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"output/screenshot_{timestamp}.jpg"
            Path("output").mkdir(exist_ok=True)
            cv2.imwrite(screenshot_path, self.current_frame)
            self.logger.info(f"Screenshot saved: {screenshot_path}")
    
    def _process_frame(self, frame):
        """Process a single frame."""
        if not self.tracker:
            return frame, []
        
        # Perform detection and tracking
        annotated_frame, tracked_objects = self.tracker.track(frame)
        
        # Update performance monitor
        detection_count = len(tracked_objects)
        inference_time = 1000 / self.detector.get_fps() if self.detector.get_fps() > 0 else 0
        self.performance_monitor.update(
            fps=self.detector.get_fps(),
            detection_count=detection_count,
            inference_time=inference_time,
            detections=tracked_objects
        )
        
        return annotated_frame, tracked_objects
    
    def run(self):
        """Run the main application loop."""
        try:
            # Setup components
            self._setup_video_capture()
            self._setup_video_writer()
            self._setup_detector()
            self._setup_tracker()
            
            self.logger.info("Starting real-time detection and tracking...")
            self.running = True
            self.start_time = time.time()
            
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame, trying to reconnect...")
                    time.sleep(1)
                    continue
                
                self.current_frame = frame.copy()
                
                if not self.paused:
                    # Process frame
                    processed_frame, tracked_objects = self._process_frame(frame)
                    
                    # Draw info overlay
                    self._draw_info_overlay(processed_frame, tracked_objects)
                    
                    # Save frame if writer is available
                    if self.video_writer:
                        self.video_writer.write(processed_frame)
                    
                    self.frame_count += 1
                else:
                    processed_frame = frame.copy()
                    self._draw_info_overlay(processed_frame, [])
                
                # Display frame
                cv2.imshow("Real-Time Object Detection & Tracking", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    self._handle_keyboard_input(key)
                
                # Log progress
                if self.frame_count % 100 == 0:
                    elapsed_time = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed_time
                    self.logger.info(f"Processed {self.frame_count} frames, avg FPS: {avg_fps:.2f}")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Save performance metrics
        if self.frame_count > 0:
            self._save_performance_metrics()
        
        self.logger.info("Cleanup completed")
    
    def _save_performance_metrics(self):
        """Save performance metrics to file."""
        try:
            metrics = self.performance_monitor.get_metrics()
            
            # Add summary statistics
            elapsed_time = time.time() - self.start_time
            metrics['summary']['total_elapsed_time'] = elapsed_time
            metrics['summary']['total_frames_processed'] = self.frame_count
            
            # Save to JSON
            import json
            metrics_path = "output/performance_metrics.json"
            Path("output").mkdir(exist_ok=True)
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Create performance plot
            plot_path = "output/performance_plot.png"
            create_performance_plot(metrics, plot_path)
            
            self.logger.info(f"Performance metrics saved: {metrics_path}")
            self.logger.info(f"Performance plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance metrics: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection & Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use webcam
  python main.py --input video.mp4                  # Process video file
  python main.py --model yolov8s.pt                # Use different model
  python main.py --device cuda                     # Use GPU
  python main.py --save-output --output result.mp4 # Save output video
  python main.py --tracker bytetrack               # Use specific tracker
        """
    )
    
    # Input/Output options
    parser.add_argument("--input", "-i", type=str, 
                       help="Input video source (0 for webcam, or path to video file)")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output video path (default: output/detection_output.mp4)")
    parser.add_argument("--save-output", action="store_true", 
                       help="Save output video")
    
    # Model options
    parser.add_argument("--model", "-m", type=str, 
                       help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--device", "-d", type=str, 
                       help="Device to run on (cpu, cuda, 0, 1, etc.)")
    
    # Tracker options
    parser.add_argument("--tracker", "-t", type=str, 
                       choices=["bytetrack", "botsort", "strongsort"],
                       help="Tracker type (default: bytetrack)")
    
    # Performance options
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run benchmark test")
    parser.add_argument("--benchmark-frames", type=int, default=100, 
                       help="Number of frames for benchmark (default: 100)")
    
    # Display options
    parser.add_argument("--no-display", action="store_true", 
                       help="Run without display (headless mode)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run application
    app = RealTimeDetectionApp(args)
    
    if args.benchmark:
        # Run benchmark
        app._setup_detector()
        metrics = app.detector.benchmark(args.benchmark_frames)
        print(f"Benchmark Results:")
        print(f"  Average FPS: {metrics['avg_fps']:.2f}")
        print(f"  Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Device: {metrics['device']}")
    else:
        # Run main application
        app.run()


if __name__ == "__main__":
    main()
