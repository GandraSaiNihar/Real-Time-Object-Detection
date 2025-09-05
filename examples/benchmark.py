#!/usr/bin/env python3
"""
Performance Benchmark Example
Example showing how to benchmark the detection and tracking performance.
"""

import sys
import time
import argparse
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.utils import setup_logger


def benchmark_detector(model_path: str = None, device: str = "cpu", 
                      test_frames: int = 100, warmup_frames: int = 10):
    """
    Benchmark detector performance.
    
    Args:
        model_path: Path to YOLO model
        device: Device to run on
        test_frames: Number of test frames
        warmup_frames: Number of warmup frames
    """
    logger = setup_logger("DetectorBenchmark")
    
    logger.info("Initializing detector...")
    detector = YOLODetector(model_path=model_path, device=device)
    
    logger.info(f"Running benchmark: {test_frames} frames on {device}")
    
    # Run benchmark
    metrics = detector.benchmark(test_frames)
    
    # Print results
    print("\n" + "="*50)
    print("DETECTOR BENCHMARK RESULTS")
    print("="*50)
    print(f"Model: {model_path or 'Default YOLOv8n'}")
    print(f"Device: {metrics['device']}")
    print(f"Test Frames: {metrics['total_frames']}")
    print(f"Total Time: {metrics['total_time']:.2f} seconds")
    print(f"Average FPS: {metrics['avg_fps']:.2f}")
    print(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    print("="*50)
    
    return metrics


def benchmark_tracker(model_path: str = None, device: str = "cpu", 
                     test_frames: int = 100, tracker_type: str = "bytetrack"):
    """
    Benchmark tracker performance.
    
    Args:
        model_path: Path to YOLO model
        device: Device to run on
        test_frames: Number of test frames
        tracker_type: Type of tracker
    """
    logger = setup_logger("TrackerBenchmark")
    
    logger.info("Initializing detector and tracker...")
    detector = YOLODetector(model_path=model_path, device=device)
    tracker = MultiObjectTracker(detector, tracker_type=tracker_type)
    
    logger.info(f"Running tracker benchmark: {test_frames} frames")
    
    # Create dummy frame with some objects
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some dummy detections
    for i in range(3):
        x1, y1 = np.random.randint(0, 500, 2)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        cv2.rectangle(dummy_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Warmup
    for _ in range(10):
        tracker.track(dummy_frame)
    
    # Benchmark
    start_time = time.time()
    total_detections = 0
    total_tracks = 0
    
    for i in range(test_frames):
        # Vary the frame slightly to simulate real conditions
        frame = dummy_frame.copy()
        if i % 10 == 0:
            # Add some noise
            noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        annotated_frame, tracked_objects = tracker.track(frame)
        total_detections += len(tracked_objects)
        total_tracks += len(tracker.tracker.tracks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_fps = test_frames / total_time
    avg_inference_time = total_time / test_frames * 1000
    avg_detections_per_frame = total_detections / test_frames
    avg_tracks_per_frame = total_tracks / test_frames
    
    # Get tracker stats
    stats = tracker.tracker.get_tracking_stats()
    
    # Print results
    print("\n" + "="*50)
    print("TRACKER BENCHMARK RESULTS")
    print("="*50)
    print(f"Tracker Type: {tracker_type}")
    print(f"Device: {device}")
    print(f"Test Frames: {test_frames}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Average Detections/Frame: {avg_detections_per_frame:.2f}")
    print(f"Average Tracks/Frame: {avg_tracks_per_frame:.2f}")
    print(f"Total Tracks Created: {stats['total_tracks_created']}")
    print("="*50)
    
    return {
        'total_frames': test_frames,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'avg_inference_time_ms': avg_inference_time,
        'avg_detections_per_frame': avg_detections_per_frame,
        'avg_tracks_per_frame': avg_tracks_per_frame,
        'tracker_stats': stats
    }


def compare_models():
    """Compare different YOLO model sizes."""
    models = [
        ("yolov8n.pt", "YOLOv8 Nano"),
        ("yolov8s.pt", "YOLOv8 Small"),
        ("yolov8m.pt", "YOLOv8 Medium"),
    ]
    
    print("\n" + "="*60)
    print("MODEL COMPARISON BENCHMARK")
    print("="*60)
    
    results = []
    
    for model_path, model_name in models:
        try:
            print(f"\nTesting {model_name}...")
            metrics = benchmark_detector(model_path, device="cpu", test_frames=50)
            results.append((model_name, metrics))
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'FPS':<10} {'Inference Time (ms)':<20}")
    print("-" * 60)
    
    for model_name, metrics in results:
        print(f"{model_name:<20} {metrics['avg_fps']:<10.2f} {metrics['avg_inference_time_ms']:<20.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark detection and tracking performance")
    parser.add_argument("--model", "-m", help="YOLO model path")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run on")
    parser.add_argument("--frames", "-f", type=int, default=100, help="Number of test frames")
    parser.add_argument("--tracker", "-t", default="bytetrack", help="Tracker type")
    parser.add_argument("--compare-models", action="store_true", help="Compare different models")
    parser.add_argument("--detector-only", action="store_true", help="Benchmark detector only")
    parser.add_argument("--tracker-only", action="store_true", help="Benchmark tracker only")
    
    args = parser.parse_args()
    
    if args.compare_models:
        compare_models()
    elif args.detector_only:
        benchmark_detector(args.model, args.device, args.frames)
    elif args.tracker_only:
        benchmark_tracker(args.model, args.device, args.frames, args.tracker)
    else:
        # Run both benchmarks
        benchmark_detector(args.model, args.device, args.frames)
        benchmark_tracker(args.model, args.device, args.frames, args.tracker)


if __name__ == "__main__":
    main()
