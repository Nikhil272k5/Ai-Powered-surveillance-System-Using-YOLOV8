"""
Abandoned Object Detection Logic
Detects objects left behind by people using tracking history
"""

import time
import numpy as np
from collections import defaultdict

class AbandonmentDetector:
    def __init__(self, abandonment_threshold=5.0, proximity_threshold=100.0):
        """
        Initialize abandonment detector
        
        Args:
            abandonment_threshold: Time in seconds to consider object abandoned
            proximity_threshold: Distance in pixels to consider person near object
        """
        self.abandonment_threshold = abandonment_threshold
        self.proximity_threshold = proximity_threshold
        
        # Track object-person associations
        self.object_person_associations = defaultdict(list)
        self.abandoned_objects = set()
        self.alerted_objects = set()
        
        print(f"🚨 Abandonment Detector initialized (threshold: {abandonment_threshold}s)")
    
    def update(self, tracked_objects, track_history):
        """
        Update abandonment detection with new tracking data
        
        Args:
            tracked_objects: Current tracked objects
            track_history: Complete tracking history from tracker
            
        Returns:
            List of abandonment alerts
        """
        current_time = time.time()
        alerts = []
        
        # Process each tracked object
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            # Skip if not an object we care about
            if class_name not in ['backpack', 'handbag', 'suitcase', 'bag']:
                continue
            
            # Skip if already alerted
            if track_id in self.alerted_objects:
                continue
            
            # Get object center
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            # Check if any person is near this object
            person_nearby = self._check_person_proximity(
                obj_center_x, obj_center_y, tracked_objects, track_history
            )
            
            if person_nearby:
                # Person is nearby, update association
                if track_id not in self.object_person_associations:
                    self.object_person_associations[track_id] = []
                person_pos = self._get_person_position(person_nearby, tracked_objects)
                if person_pos[0] is not None:
                    distance = self._calculate_distance(
                        obj_center_x, obj_center_y, 
                        person_pos[0], person_pos[1]
                    )
                else:
                    distance = float('inf')
                
                self.object_person_associations[track_id].append({
                    'timestamp': current_time,
                    'person_id': person_nearby,
                    'distance': distance
                })
                
                # Remove from abandoned set if it was there
                if track_id in self.abandoned_objects:
                    self.abandoned_objects.remove(track_id)
            else:
                # No person nearby, check if object should be considered abandoned
                if self._is_object_abandoned(track_id, current_time):
                    if track_id not in self.abandoned_objects:
                        self.abandoned_objects.add(track_id)
                        
                        # Generate alert
                        alert = self._generate_abandonment_alert(
                            track_id, class_name, obj_center_x, obj_center_y, current_time
                        )
                        alerts.append(alert)
                        self.alerted_objects.add(track_id)
                else:
                    # Object just appeared without person nearby, start tracking abandonment time
                    if track_id not in self.object_person_associations:
                        self.object_person_associations[track_id] = []
                    # Add a dummy entry to start the abandonment timer
                    self.object_person_associations[track_id].append({
                        'timestamp': current_time,
                        'person_id': None,
                        'distance': float('inf')
                    })
        
        return alerts
    
    def _check_person_proximity(self, obj_x, obj_y, tracked_objects, track_history):
        """
        Check if any person is near the given object
        
        Returns:
            Person track ID if nearby, None otherwise
        """
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            if class_name != 'person':
                continue
            
            person_center_x = (x1 + x2) / 2
            person_center_y = (y1 + y2) / 2
            
            distance = self._calculate_distance(obj_x, obj_y, person_center_x, person_center_y)
            
            if distance < self.proximity_threshold:
                return track_id
        
        return None
    
    def _calculate_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _get_person_position(self, person_id, tracked_objects):
        """Get current position of a person"""
        for obj in tracked_objects:
            if obj[0] == person_id:
                x1, y1, x2, y2 = obj[1:5]
                return (x1 + x2) / 2, (y1 + y2) / 2
        return None, None
    
    def _is_object_abandoned(self, track_id, current_time):
        """
        Check if object has been abandoned based on association history
        """
        if track_id not in self.object_person_associations:
            return False
        
        associations = self.object_person_associations[track_id]
        if not associations:
            return False
        
        # Check if object has been without a person nearby for threshold time
        last_association = associations[-1]
        
        # If the last association was with a person, check time since then
        if last_association['person_id'] is not None:
            time_since_person = current_time - last_association['timestamp']
            return time_since_person > self.abandonment_threshold
        else:
            # Object appeared without person nearby, check if it's been there long enough
            time_since_appearance = current_time - last_association['timestamp']
            return time_since_appearance > self.abandonment_threshold
    
    def _generate_abandonment_alert(self, track_id, class_name, x, y, timestamp):
        """Generate abandonment alert"""
        return {
            'timestamp': timestamp,
            'type': 'abandoned_object',
            'track_id': track_id,
            'description': f'Abandoned {class_name} detected at position ({x:.1f}, {y:.1f})',
            'position': (x, y),
            'object_type': class_name,
            'severity': 'medium'
        }
    
    def get_abandoned_objects(self):
        """Get currently abandoned objects"""
        return list(self.abandoned_objects)
    
    def get_object_associations(self, track_id):
        """Get person-object association history for a track"""
        return self.object_person_associations.get(track_id, [])

