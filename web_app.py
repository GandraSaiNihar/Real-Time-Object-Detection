#!/usr/bin/env python3
"""
Web Application for Real-Time Object Detection & Tracking
Flask-based web interface for accessing the detection system via localhost.
"""

import os
import sys
import cv2
import base64
import threading
import time
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.utils import setup_logger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'real-time-detection-2024'

# Global variables for the detection system
detector = None
tracker = None
camera = None
is_running = False
current_frame = None
detection_stats = {
    'fps': 0,
    'objects_detected': 0,
    'total_frames': 0,
    'start_time': None
}

logger = setup_logger("WebApp")

def initialize_detection_system():
    """Initialize the detection and tracking system."""
    global detector, tracker
    try:
        logger.info("Initializing detection system...")
        detector = YOLODetector(device="cpu")
        tracker = MultiObjectTracker(detector, tracker_type="bytetrack")
        logger.info("Detection system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize detection system: {e}")
        return False

def start_camera():
    """Start the camera capture."""
    global camera, is_running, detection_stats
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return False
        
        is_running = True
        detection_stats['start_time'] = time.time()
        logger.info("Camera started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start camera: {e}")
        return False

def stop_camera():
    """Stop the camera capture."""
    global camera, is_running
    is_running = False
    if camera:
        camera.release()
    logger.info("Camera stopped")

def generate_frames():
    """Generate video frames with detection and tracking."""
    global current_frame, detection_stats
    
    while is_running:
        if camera is None:
            break
            
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue
        
        try:
            # Perform detection and tracking
            if detector and tracker:
                annotated_frame, tracked_objects = tracker.track(frame)
                
                # Update statistics
                detection_stats['objects_detected'] = len(tracked_objects)
                detection_stats['fps'] = detector.get_fps()
                detection_stats['total_frames'] += 1
                
                # Add info overlay
                cv2.putText(annotated_frame, f"FPS: {detection_stats['fps']:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {detection_stats['objects_detected']}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Frames: {detection_stats['total_frames']}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                current_frame = annotated_frame
            else:
                current_frame = frame
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            current_frame = frame
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get detection statistics."""
    global detection_stats
    return jsonify(detection_stats)

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start detection system."""
    global is_running
    
    if not is_running:
        if initialize_detection_system() and start_camera():
            # Start frame generation in a separate thread
            threading.Thread(target=generate_frames, daemon=True).start()
            return jsonify({'status': 'success', 'message': 'Detection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start detection'})
    else:
        return jsonify({'status': 'info', 'message': 'Detection already running'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop detection system."""
    global is_running
    stop_camera()
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/api/status')
def get_status():
    """Get system status."""
    return jsonify({
        'is_running': is_running,
        'camera_available': camera is not None and camera.isOpened() if camera else False,
        'detector_loaded': detector is not None,
        'tracker_loaded': tracker is not None
    })

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run performance benchmark."""
    if not detector:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'})
    
    try:
        frames = request.json.get('frames', 10)
        metrics = detector.benchmark(frames)
        return jsonify({'status': 'success', 'metrics': metrics})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Real-Time Object Detection Web Application...")
    logger.info("Access the application at: http://localhost:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down web application...")
        stop_camera()
    except Exception as e:
        logger.error(f"Error running web application: {e}")
