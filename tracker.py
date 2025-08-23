#!/usr/bin/env python3
"""
Object Tracker for AbnoGuard
Handles object tracking across video frames
"""

import cv2
import numpy as np
import time
from collections import defaultdict

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3):
        """
        Initialize object tracker
        
        Args:
            max_age: Maximum frames to keep track of lost objects
            n_init: Number of frames to confirm a track
        """
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.frame_count = 0
        
        print(f"🎯 Object Tracker initialized (max_age={max_age}, n_init={n_init})")
    
    def _iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _distance(self, bbox1, bbox2):
        """Calculate center distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf, class_id, class_name]
            frame: Current frame for visualization
            
        Returns:
            List of tracked objects [track_id, x1, y1, x2, y2, class_name, confidence]
        """
        self.frame_count += 1
        
        # Convert detections to bbox format
        detection_bboxes = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id, class_name = det
            detection_bboxes.append((x1, y1, x2, y2, conf, class_id, class_name))
        
        # Update existing tracks
        updated_tracks = {}
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track['lost'] > self.max_age:
                continue  # Remove old tracks
            
            # Try to match with detections
            best_match = None
            best_score = 0
            
            for i, (x1, y1, x2, y2, conf, class_id, class_name) in enumerate(detection_bboxes):
                if i in matched_detections:
                    continue
                
                # Check class compatibility
                if track['class_name'] == class_name:
                    # Calculate matching score (combination of IOU and distance)
                    iou = self._iou(track['bbox'], (x1, y1, x2, y2))
                    distance = self._distance(track['bbox'], (x1, y1, x2, y2))
                    
                    # Normalize distance (assuming frame size ~640x480)
                    normalized_distance = distance / 800.0
                    
                    # Combined score
                    score = iou * 0.7 + (1 - normalized_distance) * 0.3
                    
                    if score > best_score and score > 0.3:  # Minimum threshold
                        best_score = score
                        best_match = i
            
            if best_match is not None:
                # Update track
                x1, y1, x2, y2, conf, class_id, class_name = detection_bboxes[best_match]
                track['bbox'] = (x1, y1, x2, y2)
                track['confidence'] = conf
                track['lost'] = 0
                track['hits'] += 1
                matched_detections.add(best_match)
                
                updated_tracks[track_id] = track
            else:
                # Track not matched, increment lost counter
                track['lost'] += 1
                updated_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for i, (x1, y1, x2, y2, conf, class_id, class_name) in enumerate(detection_bboxes):
            if i not in matched_detections:
                new_track = {
                    'bbox': (x1, y1, x2, y2),
                    'class_name': class_name,
                    'confidence': conf,
                    'hits': 1,
                    'lost': 0,
                    'start_time': time.time()
                }
                
                updated_tracks[self.next_id] = new_track
                self.next_id += 1
        
        self.tracks = updated_tracks
        
        # Return tracked objects in expected format
        tracked_objects = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.n_init:  # Only return confirmed tracks
                x1, y1, x2, y2 = track['bbox']
                tracked_objects.append([
                    track_id, x1, y1, x2, y2, track['class_name'], track['confidence']
                ])
        
        return tracked_objects
    
    def get_track_info(self, track_id):
        """Get information about a specific track"""
        if track_id in self.tracks:
            return self.tracks[track_id]
        return None
    
    def get_all_tracks(self):
        """Get all active tracks"""
        return self.tracks
    
    def reset(self):
        """Reset all tracks"""
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0

