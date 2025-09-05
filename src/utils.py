"""
Utility functions for the Real-Time Object Detection & Tracking system.
"""

import cv2
import numpy as np
import logging
import time
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path("logs") / f"{name}.log"
    log_file.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """
    Calculate center distance between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Euclidean distance between centers
    """
    # Calculate centers
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    
    # Calculate distance
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    return distance


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate FPS from start time and frame count.
    
    Args:
        start_time: Start time
        frame_count: Number of frames processed
        
    Returns:
        FPS value
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0.0


def draw_detection_box(frame: np.ndarray, bbox: List[float], 
                      label: str, color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """
    Draw detection box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text
        color: Box color (BGR)
        thickness: Box thickness
        
    Returns:
        Frame with drawn box
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0], y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def resize_frame(frame: np.ndarray, max_width: int = 1280, 
                max_height: int = 720) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if width <= max_width and height <= max_height:
        return frame
    
    # Calculate scale factor
    scale = min(max_width / width, max_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    return resized_frame


def create_video_writer(output_path: str, fps: int, width: int, height: int,
                       codec: str = "mp4v") -> cv2.VideoWriter:
    """
    Create video writer for saving output.
    
    Args:
        output_path: Output video path
        fps: Video FPS
        width: Video width
        height: Video height
        codec: Video codec
        
    Returns:
        Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_results(results: List[Dict], output_path: str):
    """
    Save detection/tracking results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(input_path: str) -> List[Dict]:
    """
    Load detection/tracking results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        List of result dictionaries
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_performance_plot(metrics: Dict, output_path: str):
    """
    Create performance visualization plot.
    
    Args:
        metrics: Performance metrics dictionary
        output_path: Output plot path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Performance Metrics', fontsize=16)
    
    # FPS over time
    if 'fps_history' in metrics:
        axes[0, 0].plot(metrics['fps_history'])
        axes[0, 0].set_title('FPS Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True)
    
    # Detection count
    if 'detection_counts' in metrics:
        axes[0, 1].bar(range(len(metrics['detection_counts'])), metrics['detection_counts'])
        axes[0, 1].set_title('Detection Count per Frame')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Count')
    
    # Class distribution
    if 'class_distribution' in metrics:
        classes = list(metrics['class_distribution'].keys())
        counts = list(metrics['class_distribution'].values())
        axes[1, 0].pie(counts, labels=classes, autopct='%1.1f%%')
        axes[1, 0].set_title('Class Distribution')
    
    # Performance summary
    if 'summary' in metrics:
        summary = metrics['summary']
        summary_text = f"""
        Average FPS: {summary.get('avg_fps', 0):.2f}
        Total Frames: {summary.get('total_frames', 0)}
        Total Detections: {summary.get('total_detections', 0)}
        Average Detections/Frame: {summary.get('avg_detections_per_frame', 0):.2f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


class PerformanceMonitor:
    """
    Monitor and track performance metrics.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Size of history buffer
        """
        self.history_size = history_size
        self.fps_history = deque(maxlen=history_size)
        self.detection_counts = deque(maxlen=history_size)
        self.inference_times = deque(maxlen=history_size)
        self.class_distribution = {}
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
    
    def update(self, fps: float, detection_count: int, 
               inference_time: float, detections: List[Dict]):
        """
        Update performance metrics.
        
        Args:
            fps: Current FPS
            detection_count: Number of detections
            inference_time: Inference time in ms
            detections: List of detections
        """
        self.fps_history.append(fps)
        self.detection_counts.append(detection_count)
        self.inference_times.append(inference_time)
        
        self.total_frames += 1
        self.total_detections += detection_count
        
        # Update class distribution
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            self.class_distribution[class_name] = self.class_distribution.get(class_name, 0) + 1
    
    def get_metrics(self) -> Dict:
        """
        Get current performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'fps_history': list(self.fps_history),
            'detection_counts': list(self.detection_counts),
            'inference_times': list(self.inference_times),
            'class_distribution': self.class_distribution.copy(),
            'summary': {
                'avg_fps': avg_fps,
                'avg_detections_per_frame': avg_detections,
                'avg_inference_time_ms': avg_inference_time,
                'total_frames': self.total_frames,
                'total_detections': self.total_detections,
                'elapsed_time': elapsed_time
            }
        }
    
    def reset(self):
        """Reset all metrics."""
        self.fps_history.clear()
        self.detection_counts.clear()
        self.inference_times.clear()
        self.class_distribution.clear()
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()


def validate_bbox(bbox: List[float], frame_shape: Tuple[int, int]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        frame_shape: Frame shape (height, width)
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    
    # Check size
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check minimum size
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return False
    
    return True


def non_max_suppression(boxes: List[List[float]], scores: List[float], 
                       iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: List of bounding boxes
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if not boxes or not scores:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = []
        for box in remaining_boxes:
            iou = calculate_iou(current_box.tolist(), box.tolist())
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][ious <= iou_threshold]
    
    return [int(idx) for idx in keep]
