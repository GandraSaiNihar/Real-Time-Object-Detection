"""
Configuration settings for the Real-Time Object Detection & Tracking system.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [MODELS_DIR, DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# YOLOv8 Configuration
YOLO_CONFIG = {
    "model_size": "yolov8n.pt",  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 1000,
    "device": "cpu",  # Options: "cpu", "cuda", "0", "1", etc.
    "half_precision": False,  # Use FP16 for faster inference
}

# Tracking Configuration
TRACKING_CONFIG = {
    "tracker_type": "bytetrack",  # Options: "bytetrack", "botsort", "strongsort"
    "track_buffer": 30,
    "match_thresh": 0.8,
    "frame_rate": 30,
    "min_box_area": 10,
    "aspect_ratio_thresh": 3.0,
}

# Video/Stream Configuration
VIDEO_CONFIG = {
    "input_source": 0,  # 0 for webcam, or path to video file
    "output_fps": 30,
    "output_codec": "mp4v",
    "display_width": 1280,
    "display_height": 720,
    "save_output": True,
    "show_fps": True,
    "show_confidence": True,
    "show_tracking_id": True,
}

# Performance Optimization
PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "batch_size": 1,
    "num_workers": 4,
    "pin_memory": True,
    "optimize_for_mobile": False,
    "quantization": False,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "detection.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# COCO Class Names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colors for visualization (BGR format)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0),
    (255, 20, 147), (0, 191, 255), (50, 205, 50), (255, 69, 0), (255, 215, 0)
]
