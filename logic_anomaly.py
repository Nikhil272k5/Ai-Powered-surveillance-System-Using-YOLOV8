"""
Abnormal Movement Detection Logic
Detects suspicious behaviors: speed spikes, counterflow, loitering
"""

import time
import numpy as np
from collections import defaultdict, deque

class AnomalyDetector:
    def __init__(self, 
                 speed_threshold=30.0, 
                 loitering_threshold=15.0,
                 counterflow_threshold=0.7):
        """
        Initialize anomaly detector
        
        Args:
            speed_threshold: Speed threshold for speed spike detection (pixels/frame)
            loitering_threshold: Time threshold for loitering detection (seconds)
            counterflow_threshold: Threshold for counterflow detection (cosine similarity)
        """
        self.speed_threshold = speed_threshold
        self.loitering_threshold = loitering_threshold
        self.counterflow_threshold = counterflow_threshold
        
        # Track movement patterns
        self.movement_history = defaultdict(lambda: deque(maxlen=30))
        self.alerted_tracks = set()
        
        # Define expected flow direction (can be adjusted based on scene)
        self.expected_flow = np.array([1.0, 0.0])  # Rightward movement
        
        print(f"🚨 Anomaly Detector initialized")
        print(f"   Speed threshold: {speed_threshold} px/frame")
        print(f"   Loitering threshold: {loitering_threshold}s")
        print(f"   Counterflow threshold: {counterflow_threshold}")
    
    def update(self, tracked_objects, track_history):
        """
        Update anomaly detection with new tracking data
        
        Args:
            tracked_objects: Current tracked objects
            track_history: Complete tracking history from tracker
            
        Returns:
            List of anomaly alerts
        """
        current_time = time.time()
        alerts = []
        
        # Process each tracked person
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            # Only analyze people
            if class_name != 'person':
                continue
            
            # Skip if already alerted
            if track_id in self.alerted_tracks:
                continue
            
            # Get current position
            current_pos = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            # Update movement history
            self.movement_history[track_id].append({
                'timestamp': current_time,
                'position': current_pos,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Check for anomalies
            anomaly_alerts = self._check_anomalies(track_id, current_time)
            alerts.extend(anomaly_alerts)
            
            # Mark as alerted if any anomaly detected
            if anomaly_alerts:
                self.alerted_tracks.add(track_id)
        
        return alerts
    
    def _check_anomalies(self, track_id, current_time):
        """Check for various types of anomalies"""
        alerts = []
        history = self.movement_history[track_id]
        
        if len(history) < 5:  # Need minimum history for analysis
            return alerts
        
        # Check speed spike
        speed_alert = self._check_speed_spike(track_id, history)
        if speed_alert:
            alerts.append(speed_alert)
        
        # Check loitering
        loitering_alert = self._check_loitering(track_id, history, current_time)
        if loitering_alert:
            alerts.append(loitering_alert)
        
        # Check counterflow
        counterflow_alert = self._check_counterflow(track_id, history)
        if counterflow_alert:
            alerts.append(counterflow_alert)
        
        return alerts
    
    def _check_speed_spike(self, track_id, history):
        """Check for sudden speed increase"""
        if len(history) < 3:
            return None
        
        # Calculate recent speeds
        speeds = []
        for i in range(1, len(history)):
            prev_pos = history[i-1]['position']
            curr_pos = history[i]['position']
            
            # Calculate distance moved
            distance = np.linalg.norm(curr_pos - prev_pos)
            speeds.append(distance)
        
        if len(speeds) < 2:
            return None
        
        # Check if recent speed is significantly higher than average
        recent_speed = speeds[-1]
        avg_speed = np.mean(speeds[:-1])
        
        if recent_speed > self.speed_threshold and recent_speed > avg_speed * 1.5:
            return {
                'timestamp': history[-1]['timestamp'],
                'type': 'speed_spike',
                'track_id': track_id,
                'description': f'Speed spike detected: {recent_speed:.1f} px/frame (avg: {avg_speed:.1f})',
                'severity': 'high',
                'speed': recent_speed,
                'avg_speed': avg_speed
            }
        
        return None
    
    def _check_loitering(self, track_id, history, current_time):
        """Check for loitering behavior"""
        if len(history) < 10:
            return None
        
        # Ensure history is a list and convert to list if needed
        if not isinstance(history, list):
            history = list(history)
        
        # Calculate area covered by movement
        positions = [h['position'] for h in history]
        positions = np.array(positions)
        
        # Calculate bounding box of movement
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        # Calculate area covered
        area_covered = (max_x - min_x) * (max_y - min_y)
        
        # Calculate time span
        time_span = current_time - history[0]['timestamp']
        
        # If person has been in small area for long time, consider loitering
        if area_covered < 2000 and time_span > self.loitering_threshold:
            return {
                'timestamp': current_time,
                'type': 'loitering',
                'track_id': track_id,
                'description': f'Loitering detected: {time_span:.1f}s in small area ({area_covered:.1f} px²)',
                'severity': 'medium',
                'time_span': time_span,
                'area_covered': area_covered
            }
        
        return None
    
    def _check_counterflow(self, track_id, history):
        """Check for movement against expected flow direction"""
        if len(history) < 5:
            return None
        
        # Ensure history is a list and convert to list if needed
        if not isinstance(history, list):
            history = list(history)
        
        # Calculate movement direction over last few frames
        recent_positions = [h['position'] for h in history[-5:]]
        if len(recent_positions) < 2:
            return None
        
        # Calculate average movement vector
        movement_vectors = []
        for i in range(1, len(recent_positions)):
            vector = recent_positions[i] - recent_positions[i-1]
            if np.linalg.norm(vector) > 0:  # Only consider non-zero movements
                movement_vectors.append(vector)
        
        if not movement_vectors:
            return None
        
        # Calculate average movement direction
        avg_movement = np.mean(movement_vectors, axis=0)
        avg_movement = avg_movement / np.linalg.norm(avg_movement)
        
        # Calculate cosine similarity with expected flow
        cosine_sim = np.dot(avg_movement, self.expected_flow)
        
        # If movement is opposite to expected flow (negative cosine similarity)
        if cosine_sim < -self.counterflow_threshold:
            return {
                'timestamp': history[-1]['timestamp'],
                'type': 'counterflow',
                'track_id': track_id,
                'description': f'Counterflow movement detected (cosine similarity: {cosine_sim:.2f})',
                'severity': 'medium',
                'cosine_similarity': cosine_sim,
                'movement_direction': avg_movement.tolist()
            }
        
        return None
    
    def get_movement_history(self, track_id):
        """Get movement history for a specific track"""
        return list(self.movement_history.get(track_id, []))
    
    def get_active_tracks(self):
        """Get all currently tracked IDs"""
        return list(self.movement_history.keys())
    
    def reset_alerts(self):
        """Reset alerted tracks (useful for new video)"""
        self.alerted_tracks.clear()

