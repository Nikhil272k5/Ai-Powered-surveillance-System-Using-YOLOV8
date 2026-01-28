"""
AbnoGuard Perception - Multi-Model Fusion
Fuses outputs from YOLO, optical flow, pose analysis, and scene zones
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FusedPerception:
    """Fused perception output for a single tracked entity"""
    track_id: int
    timestamp: float
    
    # Detection
    bbox: Tuple[int, int, int, int]
    class_name: str
    detection_confidence: float
    
    # Motion (from optical flow)
    speed: float
    motion_pattern: str
    motion_anomaly: bool
    
    # Pose (body language)
    posture: str
    stress_level: str
    stress_score: float
    
    # Zone context
    current_zone: Optional[str]
    zone_type: Optional[str]
    zone_dwell_time: float
    zone_violation: bool
    
    # Fused analysis
    threat_level: ThreatLevel
    threat_score: float
    anomaly_reasons: List[str]
    context_factors: Dict[str, Any]
    
    # Behavior trajectory
    behavior_trend: str  # stable, escalating, de-escalating
    historical_anomalies: int


@dataclass
class FramePerception:
    """Complete perception output for a frame"""
    frame_id: int
    timestamp: float
    
    # All entities
    entities: List[FusedPerception]
    
    # Scene-level analysis
    crowd_density: float
    global_motion_pattern: str
    scene_anomaly_score: float
    
    # Zone summary
    zone_occupancy: Dict[str, int]
    zone_violations: List[Dict]
    
    # Alerts to generate
    perception_alerts: List[Dict]


class PerceptionFusion:
    """
    Fuses multiple perception models into unified understanding.
    
    Inputs:
    - YOLO detections + tracking
    - Optical flow motion analysis
    - Pose estimation
    - Scene zone analysis
    
    Output:
    - Fused threat assessment
    - Context-aware anomaly detection
    - Behavioral trajectory
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Import perception modules
        from perception.optical_flow import OpticalFlowAnalyzer
        from perception.pose_analyzer import PoseAnalyzer
        from perception.scene_zones import SceneZoneAnalyzer
        
        # Initialize analyzers
        self.flow_analyzer = OpticalFlowAnalyzer(config.get('optical_flow', {}))
        self.pose_analyzer = PoseAnalyzer(config.get('pose', {}))
        self.zone_analyzer = SceneZoneAnalyzer(config.get('zones', {}))
        
        # History for behavior trajectory
        self.entity_history: Dict[int, List[Dict]] = {}
        self.history_length = self.config.get('history_length', 60)
        
        # Fusion weights
        self.weights = {
            'detection': self.config.get('weight_detection', 0.3),
            'motion': self.config.get('weight_motion', 0.25),
            'pose': self.config.get('weight_pose', 0.25),
            'zone': self.config.get('weight_zone', 0.2)
        }
        
        # Frame counter
        self.frame_count = 0
        self.zones_scaled = False
        
        print("🔗 Perception Fusion Engine initialized")
    
    def process_frame(self, frame: np.ndarray, 
                     detections: List[Dict],
                     tracked_objects: List[Tuple]) -> FramePerception:
        """
        Process a frame through all perception models and fuse results
        
        Args:
            frame: BGR image
            detections: Raw YOLO detections
            tracked_objects: List of (track_id, x1, y1, x2, y2, class, conf)
        
        Returns:
            FramePerception with fused analysis
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Scale zones on first frame
        if not self.zones_scaled and frame is not None:
            h, w = frame.shape[:2]
            self.zone_analyzer.scale_zones_to_frame(w, h)
            self.zones_scaled = True
        
        # Run all perception models
        flow_result = self.flow_analyzer.analyze(frame)
        pose_results = self.pose_analyzer.analyze_frame(frame, tracked_objects)
        zone_result = self.zone_analyzer.analyze(tracked_objects, current_time)
        
        # Create lookup for pose results
        pose_lookup = {p.track_id: p for p in pose_results}
        
        # Fuse for each tracked entity
        entities = []
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, conf = obj
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Get motion for this entity
            speed = 0.0
            motion_anomaly = False
            motion_pattern = "unknown"
            
            if flow_result['flow'] is not None:
                speed = self.flow_analyzer.get_speed_for_bbox(flow_result['flow'], bbox)
                motion_pattern = flow_result['global_motion'].get('pattern', 'unknown')
                if hasattr(motion_pattern, 'value'):
                    motion_pattern = motion_pattern.value
            
            # Get pose for this entity
            pose = pose_lookup.get(track_id)
            posture = pose.posture.value if pose else "unknown"
            stress_level = pose.stress_level.value if pose else "low"
            stress_score = pose.stress_score if pose else 0.0
            
            # Get zone context
            current_zone = None
            zone_type = None
            zone_dwell = 0.0
            zone_violation = False
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            zone = self.zone_analyzer.get_zone_for_point(int(center_x), int(center_y))
            
            if zone:
                current_zone = zone.zone_id
                zone_type = zone.zone_type.value
                if track_id in self.zone_analyzer.track_zones:
                    enter_time = self.zone_analyzer.track_zones[track_id].get(zone.zone_id)
                    if enter_time:
                        zone_dwell = current_time - enter_time
            
            # Check for zone violations
            for v in zone_result['violations']:
                if v['track_id'] == track_id:
                    zone_violation = True
                    break
            
            # Fuse into threat assessment
            threat_score, threat_level, anomaly_reasons = self._compute_threat(
                detection_conf=conf,
                speed=speed,
                stress_score=stress_score,
                zone_violation=zone_violation,
                zone_type=zone_type,
                class_name=class_name
            )
            
            # Update history and compute trajectory
            behavior_trend, historical_anomalies = self._update_history(
                track_id, threat_score, anomaly_reasons, current_time
            )
            
            # Create fused perception
            entity = FusedPerception(
                track_id=track_id,
                timestamp=current_time,
                bbox=bbox,
                class_name=class_name,
                detection_confidence=conf,
                speed=speed,
                motion_pattern=motion_pattern,
                motion_anomaly=motion_anomaly,
                posture=posture,
                stress_level=stress_level,
                stress_score=stress_score,
                current_zone=current_zone,
                zone_type=zone_type,
                zone_dwell_time=zone_dwell,
                zone_violation=zone_violation,
                threat_level=threat_level,
                threat_score=threat_score,
                anomaly_reasons=anomaly_reasons,
                context_factors=zone_result.get('context_adjustments', {}).get(track_id, {}),
                behavior_trend=behavior_trend,
                historical_anomalies=historical_anomalies
            )
            
            entities.append(entity)
        
        # Compute scene-level metrics
        crowd_density = len(entities) / 100  # Normalized
        scene_anomaly = self._compute_scene_anomaly(entities, flow_result)
        
        # Generate perception-level alerts
        alerts = self._generate_alerts(entities, zone_result)
        
        return FramePerception(
            frame_id=self.frame_count,
            timestamp=current_time,
            entities=entities,
            crowd_density=crowd_density,
            global_motion_pattern=flow_result['global_motion'].get('pattern', 'unknown') if flow_result['global_motion'] else 'unknown',
            scene_anomaly_score=scene_anomaly,
            zone_occupancy=zone_result['zone_occupancy'],
            zone_violations=zone_result['violations'],
            perception_alerts=alerts
        )
    
    def _compute_threat(self, detection_conf: float, speed: float,
                       stress_score: float, zone_violation: bool,
                       zone_type: Optional[str], class_name: str) -> Tuple[float, ThreatLevel, List[str]]:
        """Compute fused threat score and level"""
        anomaly_reasons = []
        
        # Base scores from each modality
        motion_score = min(1.0, speed / 20)  # Normalize speed
        pose_score = stress_score
        zone_score = 1.0 if zone_violation else 0.0
        
        # Context adjustments
        if zone_type == "restricted":
            zone_score *= 1.5
        elif zone_type == "waiting":
            motion_score *= 0.5  # Less sensitive in waiting areas
        
        # Weighted fusion
        threat_score = (
            self.weights['detection'] * (1 - detection_conf) +  # Low conf = higher threat
            self.weights['motion'] * motion_score +
            self.weights['pose'] * pose_score +
            self.weights['zone'] * zone_score
        )
        
        threat_score = min(1.0, threat_score)
        
        # Determine reasons
        if motion_score > 0.5:
            anomaly_reasons.append(f"high_speed_{speed:.1f}")
        if pose_score > 0.5:
            anomaly_reasons.append("stress_detected")
        if zone_violation:
            anomaly_reasons.append("zone_violation")
        
        # Determine level
        if threat_score >= 0.8:
            level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            level = ThreatLevel.HIGH
        elif threat_score >= 0.4:
            level = ThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            level = ThreatLevel.LOW
        else:
            level = ThreatLevel.NONE
        
        return threat_score, level, anomaly_reasons
    
    def _update_history(self, track_id: int, threat_score: float,
                       anomaly_reasons: List[str], timestamp: float) -> Tuple[str, int]:
        """Update entity history and compute behavior trend"""
        if track_id not in self.entity_history:
            self.entity_history[track_id] = []
        
        self.entity_history[track_id].append({
            'threat_score': threat_score,
            'anomalies': len(anomaly_reasons),
            'timestamp': timestamp
        })
        
        # Trim history
        if len(self.entity_history[track_id]) > self.history_length:
            self.entity_history[track_id].pop(0)
        
        history = self.entity_history[track_id]
        
        # Count historical anomalies
        historical_anomalies = sum(h['anomalies'] for h in history)
        
        # Compute trend
        if len(history) < 5:
            return "stable", historical_anomalies
        
        recent = [h['threat_score'] for h in history[-5:]]
        older = [h['threat_score'] for h in history[-10:-5]] if len(history) >= 10 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.1:
            trend = "escalating"
        elif recent_avg < older_avg - 0.1:
            trend = "de-escalating"
        else:
            trend = "stable"
        
        return trend, historical_anomalies
    
    def _compute_scene_anomaly(self, entities: List[FusedPerception], 
                              flow_result: Dict) -> float:
        """Compute scene-level anomaly score"""
        if not entities:
            return 0.0
        
        # Average entity threat
        avg_threat = np.mean([e.threat_score for e in entities])
        
        # Motion anomalies
        motion_anomalies = len(flow_result.get('anomalies', []))
        motion_score = min(1.0, motion_anomalies / 3)
        
        # Crowd factor
        crowd_factor = min(1.0, len(entities) / 20)
        
        # Zone violations
        violations = sum(1 for e in entities if e.zone_violation)
        violation_score = min(1.0, violations / 3)
        
        scene_score = (avg_threat * 0.4 + motion_score * 0.2 + 
                      crowd_factor * 0.2 + violation_score * 0.2)
        
        return min(1.0, scene_score)
    
    def _generate_alerts(self, entities: List[FusedPerception],
                        zone_result: Dict) -> List[Dict]:
        """Generate perception-level alerts"""
        alerts = []
        
        for entity in entities:
            if entity.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                alerts.append({
                    'type': 'perception_threat',
                    'track_id': entity.track_id,
                    'threat_level': entity.threat_level.value,
                    'threat_score': entity.threat_score,
                    'reasons': entity.anomaly_reasons,
                    'zone': entity.current_zone,
                    'trend': entity.behavior_trend
                })
        
        # Add zone violation alerts
        for violation in zone_result['violations']:
            alerts.append({
                'type': 'zone_violation',
                'track_id': violation['track_id'],
                'zone_id': violation['zone_id'],
                'violation_type': violation['violation_type'],
                'severity': violation['severity'],
                'description': violation['description']
            })
        
        return alerts
    
    def visualize(self, frame: np.ndarray, perception: FramePerception) -> np.ndarray:
        """Visualize perception results on frame"""
        import cv2
        
        vis = frame.copy()
        
        # Draw zones
        vis = self.zone_analyzer.draw_zones(vis)
        
        # Draw entities with threat colors
        threat_colors = {
            ThreatLevel.NONE: (0, 255, 0),      # Green
            ThreatLevel.LOW: (0, 255, 255),     # Yellow
            ThreatLevel.MEDIUM: (0, 165, 255),  # Orange
            ThreatLevel.HIGH: (0, 0, 255),      # Red
            ThreatLevel.CRITICAL: (255, 0, 255) # Magenta
        }
        
        for entity in perception.entities:
            x1, y1, x2, y2 = entity.bbox
            color = threat_colors[entity.threat_level]
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"T{entity.track_id} {entity.threat_level.value} ({entity.threat_score:.2f})"
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Stress indicator
            if entity.stress_score > 0.5:
                cv2.putText(vis, "STRESS", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Scene info
        cv2.putText(vis, f"Scene Anomaly: {perception.scene_anomaly_score:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def cleanup(self):
        """Release resources"""
        self.pose_analyzer.cleanup()
