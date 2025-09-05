"""
Real-Time Object Detection using YOLOv8
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

from .config import YOLO_CONFIG, COCO_CLASSES, COLORS
from .utils import setup_logger, calculate_fps, draw_detection_box


class YOLODetector:
    """
    YOLOv8-based object detector with optimization features.
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model file
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.logger = setup_logger(__name__)
        
        # Set device
        self.device = device or YOLO_CONFIG["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Load model
        model_path = model_path or YOLO_CONFIG["model_size"]
        self.logger.info(f"Loading YOLO model: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection history for smoothing
        self.detection_history = []
        self.history_size = 5
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize frame if needed for better performance
        height, width = frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280/width, 720/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference
            results = self.model(
                processed_frame,
                conf=YOLO_CONFIG["confidence_threshold"],
                iou=YOLO_CONFIG["iou_threshold"],
                max_det=YOLO_CONFIG["max_detections"],
                device=self.device,
                half=YOLO_CONFIG["half_precision"],
                verbose=False
            )
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        detection = {
                            'bbox': box,
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'class_{class_id}',
                            'track_id': None,  # Will be set by tracker
                            'timestamp': time.time()
                        }
                        detections.append(detection)
            
            # Update FPS
            self.update_fps()
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return float(self.current_fps)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_confidence: bool = True, show_class: bool = True) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            track_id = detection.get('track_id')
            
            # Get color for this class
            color = COLORS[class_id % len(COLORS)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if track_id is not None:
                label_parts.append(f"ID:{track_id}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'device': self.device,
            'num_classes': len(COCO_CLASSES),
            'input_size': getattr(self.model.model, 'imgsz', 640),
            'half_precision': YOLO_CONFIG["half_precision"]
        }
    
    def benchmark(self, test_frames: int = 100) -> Dict:
        """
        Benchmark the detector performance.
        
        Args:
            test_frames: Number of test frames to process
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Starting benchmark with {test_frames} frames...")
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(10):
            self.detect(dummy_frame)
        
        # Benchmark
        start_time = time.time()
        for _ in range(test_frames):
            self.detect(dummy_frame)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_fps = test_frames / total_time
        avg_inference_time = total_time / test_frames * 1000  # ms
        
        metrics = {
            'total_frames': test_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'avg_inference_time_ms': avg_inference_time,
            'device': self.device
        }
        
        self.logger.info(f"Benchmark completed: {avg_fps:.2f} FPS, {avg_inference_time:.2f}ms per frame")
        return metrics
