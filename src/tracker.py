"""
Object Tracking using ByteTrack and other tracking algorithms
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import logging
from collections import defaultdict, deque

from .config import TRACKING_CONFIG
from .utils import setup_logger, calculate_iou, calculate_center_distance


class ByteTracker:
    """
    ByteTrack-based object tracker with optimizations.
    """
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, 
                 match_thresh: float = 0.8, frame_rate: int = 30):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for track matching
            frame_rate: Video frame rate
        """
        self.logger = setup_logger(__name__)
        
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        # Track management
        self.track_id_count = 0
        self.tracks = {}  # Active tracks
        self.lost_tracks = {}  # Lost tracks
        self.removed_tracks = set()  # Removed track IDs
        
        # Track states
        self.track_states = defaultdict(lambda: 'New')  # New, Tracked, Lost, Removed
        
        # Performance tracking
        self.frame_count = 0
        self.tracking_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of tracked objects
        """
        self.frame_count += 1
        
        # Filter detections by confidence
        high_conf_detections = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_conf_detections = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # Update existing tracks
        self._update_tracks()
        
        # Match high confidence detections with existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._match_detections(
            high_conf_detections, list(self.tracks.keys())
        )
        
        # Update matched tracks
        for track_id, detection in matched_tracks.items():
            self._update_track(track_id, detection)
        
        # Create new tracks for unmatched high confidence detections
        for detection in unmatched_detections:
            self._create_track(detection)
        
        # Try to match low confidence detections with lost tracks
        if low_conf_detections and self.lost_tracks:
            lost_track_ids = list(self.lost_tracks.keys())
            matched_lost, _, _ = self._match_detections(low_conf_detections, lost_track_ids)
            
            for track_id, detection in matched_lost.items():
                self._recover_track(track_id, detection)
        
        # Remove old lost tracks
        self._remove_old_tracks()
        
        # Update FPS
        self._update_fps()
        
        # Return tracked objects
        return self._get_tracked_objects()
    
    def _update_tracks(self):
        """Update track states and positions."""
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track['age'] += 1
            track['time_since_update'] += 1
            
            # Predict next position using simple motion model
            if len(track['history']) >= 2:
                last_pos = track['history'][-1]['bbox']
                prev_pos = track['history'][-2]['bbox']
                
                # Simple velocity estimation
                velocity = np.array(last_pos) - np.array(prev_pos)
                predicted_bbox = np.array(last_pos) + velocity
                
                track['predicted_bbox'] = predicted_bbox.tolist()
    
    def _match_detections(self, detections: List[Dict], track_ids: List[int]) -> Tuple[Dict, List, List]:
        """
        Match detections with tracks using IoU.
        
        Args:
            detections: List of detections
            track_ids: List of track IDs
            
        Returns:
            Tuple of (matched_tracks, unmatched_detections, unmatched_tracks)
        """
        if not detections or not track_ids:
            return {}, detections, track_ids
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks.get(track_id) or self.lost_tracks.get(track_id)
                if track and 'bbox' in track:
                    iou = calculate_iou(detection['bbox'], track['bbox'])
                    iou_matrix[i, j] = iou
        
        # Match using Hungarian algorithm (simplified greedy approach)
        matched_tracks = {}
        matched_det_indices = set()
        matched_track_indices = set()
        
        # Find matches above threshold
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if iou_matrix[i, j] >= self.match_thresh:
                    if i not in matched_det_indices and j not in matched_track_indices:
                        matched_tracks[track_ids[j]] = detections[i]
                        matched_det_indices.add(i)
                        matched_track_indices.add(j)
        
        # Get unmatched items
        unmatched_detections = [detections[i] for i in range(len(detections)) 
                              if i not in matched_det_indices]
        unmatched_tracks = [track_ids[j] for j in range(len(track_ids)) 
                          if j not in matched_track_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _create_track(self, detection: Dict):
        """Create a new track from detection."""
        track_id = self.track_id_count
        self.track_id_count += 1
        
        track = {
            'track_id': track_id,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'class_id': detection['class_id'],
            'class_name': detection['class_name'],
            'age': 1,
            'time_since_update': 0,
            'history': [{
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp']
            }],
            'predicted_bbox': detection['bbox']
        }
        
        self.tracks[track_id] = track
        self.track_states[track_id] = 'Tracked'
        
        self.logger.debug(f"Created new track {track_id} for {detection['class_name']}")
    
    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track['bbox'] = detection['bbox']
            track['confidence'] = detection['confidence']
            track['time_since_update'] = 0
            
            # Add to history
            track['history'].append({
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp']
            })
            
            # Keep history size manageable
            if len(track['history']) > 30:
                track['history'] = track['history'][-30:]
            
            self.track_states[track_id] = 'Tracked'
    
    def _recover_track(self, track_id: int, detection: Dict):
        """Recover a lost track."""
        if track_id in self.lost_tracks:
            track = self.lost_tracks.pop(track_id)
            track['bbox'] = detection['bbox']
            track['confidence'] = detection['confidence']
            track['time_since_update'] = 0
            
            # Add to history
            track['history'].append({
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp']
            })
            
            self.tracks[track_id] = track
            self.track_states[track_id] = 'Tracked'
            
            self.logger.debug(f"Recovered track {track_id}")
    
    def _remove_old_tracks(self):
        """Remove tracks that have been lost for too long."""
        tracks_to_remove = []
        
        # Move tracks to lost if not updated recently
        for track_id, track in list(self.tracks.items()):
            if track['time_since_update'] > 1:
                self.lost_tracks[track_id] = track
                del self.tracks[track_id]
                self.track_states[track_id] = 'Lost'
        
        # Remove tracks that have been lost for too long
        for track_id, track in list(self.lost_tracks.items()):
            if track['time_since_update'] > self.track_buffer:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
            self.track_states[track_id] = 'Removed'
            self.removed_tracks.add(track_id)
    
    def _get_tracked_objects(self) -> List[Dict]:
        """Get list of currently tracked objects."""
        tracked_objects = []
        
        for track_id, track in self.tracks.items():
            if self.track_states[track_id] == 'Tracked':
                obj = {
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'class_id': track['class_id'],
                    'class_name': track['class_name'],
                    'age': track['age'],
                    'timestamp': time.time()
                }
                tracked_objects.append(obj)
        
        return tracked_objects
    
    def _update_fps(self):
        """Update tracking FPS."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.tracking_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            'active_tracks': len(self.tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks_created': self.track_id_count,
            'tracking_fps': self.tracking_fps,
            'frame_count': self.frame_count
        }
    
    def reset(self):
        """Reset the tracker."""
        self.track_id_count = 0
        self.tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.track_states.clear()
        self.frame_count = 0
        self.tracking_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        self.logger.info("Tracker reset")


class MultiObjectTracker:
    """
    Multi-object tracker that combines detection and tracking.
    """
    
    def __init__(self, detector, tracker_type: str = "bytetrack"):
        """
        Initialize multi-object tracker.
        
        Args:
            detector: YOLO detector instance
            tracker_type: Type of tracker to use
        """
        self.detector = detector
        self.tracker_type = tracker_type
        
        if tracker_type == "bytetrack":
            self.tracker = ByteTracker(
                track_thresh=TRACKING_CONFIG["match_thresh"],
                track_buffer=TRACKING_CONFIG["track_buffer"],
                match_thresh=TRACKING_CONFIG["match_thresh"],
                frame_rate=TRACKING_CONFIG["frame_rate"]
            )
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        
        self.logger = setup_logger(__name__)
    
    def track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Perform detection and tracking on a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, tracked_objects)
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Draw results
        annotated_frame = self.detector.draw_detections(frame, tracked_objects)
        
        return annotated_frame, tracked_objects
    
    def get_performance_stats(self) -> Dict:
        """Get combined performance statistics."""
        detector_stats = {
            'detection_fps': self.detector.get_fps(),
            'model_info': self.detector.get_model_info()
        }
        
        tracker_stats = self.tracker.get_tracking_stats()
        
        return {**detector_stats, **tracker_stats}
