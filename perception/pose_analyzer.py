"""
AbnoGuard Perception - Pose Analyzer
Analyzes body posture for stress, intent, and behavioral cues using MediaPipe
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import mediapipe, provide fallback if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available. Install with: pip install mediapipe")


class PostureState(Enum):
    NORMAL = "normal"
    TENSE = "tense"
    CROUCHING = "crouching"
    RUNNING = "running"
    FALLEN = "fallen"
    HANDS_UP = "hands_up"
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"


class StressLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PoseAnalysis:
    """Analysis results for a single person"""
    track_id: int
    posture: PostureState
    stress_level: StressLevel
    stress_score: float  # 0-1
    body_angle: float  # Lean angle
    arm_position: str  # raised, sides, forward
    movement_intensity: float
    anomaly_indicators: List[str]
    confidence: float


class PoseAnalyzer:
    """
    Analyzes human poses for behavioral cues, stress indicators, and intent.
    Uses MediaPipe for pose estimation when available.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Thresholds
        self.stress_threshold_medium = self.config.get('stress_threshold_medium', 0.4)
        self.stress_threshold_high = self.config.get('stress_threshold_high', 0.7)
        self.stress_threshold_critical = self.config.get('stress_threshold_critical', 0.9)
        
        # State tracking
        self.pose_history: Dict[int, List[Dict]] = {}
        self.history_length = self.config.get('history_length', 30)
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("🏃 Pose Analyzer initialized with MediaPipe")
        else:
            self.pose_detector = None
            print("🏃 Pose Analyzer initialized (fallback mode)")
    
    def analyze_frame(self, frame: np.ndarray, tracked_objects: List[Tuple]) -> List[PoseAnalysis]:
        """
        Analyze poses for all tracked persons in frame
        
        Args:
            frame: BGR image
            tracked_objects: List of (track_id, x1, y1, x2, y2, class_name, conf)
        
        Returns:
            List of PoseAnalysis for each person
        """
        results = []
        
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            if class_name != 'person':
                continue
            
            # Extract person region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            person_roi = frame[y1:y2, x1:x2]
            
            # Analyze pose
            analysis = self._analyze_person(track_id, person_roi, (x1, y1, x2, y2))
            results.append(analysis)
        
        return results
    
    def _analyze_person(self, track_id: int, roi: np.ndarray, bbox: Tuple) -> PoseAnalysis:
        """Analyze a single person's pose"""
        
        # Get pose landmarks if MediaPipe available
        landmarks = None
        if self.pose_detector is not None:
            try:
                rgb_roi = roi[:, :, ::-1]  # BGR to RGB
                pose_results = self.pose_detector.process(rgb_roi)
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
            except Exception as e:
                pass
        
        # Analyze based on landmarks or fallback to heuristics
        if landmarks:
            analysis = self._analyze_landmarks(track_id, landmarks, roi.shape)
        else:
            analysis = self._analyze_heuristic(track_id, roi, bbox)
        
        # Update history
        self._update_history(track_id, analysis)
        
        # Apply temporal analysis
        analysis = self._apply_temporal_analysis(track_id, analysis)
        
        return analysis
    
    def _analyze_landmarks(self, track_id: int, landmarks, roi_shape: Tuple) -> PoseAnalysis:
        """Analyze pose using MediaPipe landmarks"""
        h, w = roi_shape[:2]
        
        # Extract key points
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Calculate body measurements
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        # Body angle (lean)
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_x = (left_hip.x + right_hip.x) / 2
        body_angle = np.degrees(np.arctan2(hip_y - shoulder_y, hip_x - shoulder_x)) - 90
        
        # Arm position
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
        if avg_wrist_y < shoulder_y - 0.1:
            arm_position = "raised"
        elif avg_wrist_y < shoulder_y + 0.1:
            arm_position = "forward"
        else:
            arm_position = "sides"
        
        # Posture detection
        posture = PostureState.NORMAL
        anomaly_indicators = []
        
        # Check for crouching
        torso_ratio = (hip_y - shoulder_y) / max(0.01, ankle_y - shoulder_y)
        if torso_ratio < 0.3:
            posture = PostureState.CROUCHING
            anomaly_indicators.append("crouching_detected")
        
        # Check for hands up
        if arm_position == "raised":
            posture = PostureState.HANDS_UP
            anomaly_indicators.append("hands_raised")
        
        # Check for fallen
        if abs(body_angle) > 60:
            posture = PostureState.FALLEN
            anomaly_indicators.append("person_fallen")
        
        # Calculate stress score based on posture indicators
        stress_score = 0.0
        
        if posture == PostureState.FALLEN:
            stress_score = 0.9
        elif posture == PostureState.HANDS_UP:
            stress_score = 0.7
        elif posture == PostureState.CROUCHING:
            stress_score = 0.5
        elif abs(body_angle) > 20:
            stress_score = 0.3
        
        # Visibility affects confidence
        avg_visibility = np.mean([l.visibility for l in landmarks[:17]])
        
        # Determine stress level
        stress_level = self._score_to_level(stress_score)
        
        return PoseAnalysis(
            track_id=track_id,
            posture=posture,
            stress_level=stress_level,
            stress_score=stress_score,
            body_angle=body_angle,
            arm_position=arm_position,
            movement_intensity=0.0,  # Updated in temporal analysis
            anomaly_indicators=anomaly_indicators,
            confidence=float(avg_visibility)
        )
    
    def _analyze_heuristic(self, track_id: int, roi: np.ndarray, bbox: Tuple) -> PoseAnalysis:
        """Fallback analysis using simple heuristics when MediaPipe unavailable"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / max(1, width)
        
        # Simple posture estimation from aspect ratio
        posture = PostureState.NORMAL
        anomaly_indicators = []
        stress_score = 0.0
        
        if aspect_ratio < 1.0:  # Wide = possibly fallen
            posture = PostureState.FALLEN
            anomaly_indicators.append("unusual_aspect_ratio")
            stress_score = 0.7
        elif aspect_ratio < 1.5:  # Short = crouching
            posture = PostureState.CROUCHING
            anomaly_indicators.append("crouching_detected")
            stress_score = 0.4
        elif aspect_ratio > 3.0:  # Very tall = standing straight
            posture = PostureState.NORMAL
            stress_score = 0.1
        
        return PoseAnalysis(
            track_id=track_id,
            posture=posture,
            stress_level=self._score_to_level(stress_score),
            stress_score=stress_score,
            body_angle=0.0,
            arm_position="unknown",
            movement_intensity=0.0,
            anomaly_indicators=anomaly_indicators,
            confidence=0.5  # Lower confidence for heuristic
        )
    
    def _score_to_level(self, score: float) -> StressLevel:
        """Convert stress score to level"""
        if score >= self.stress_threshold_critical:
            return StressLevel.CRITICAL
        elif score >= self.stress_threshold_high:
            return StressLevel.HIGH
        elif score >= self.stress_threshold_medium:
            return StressLevel.MEDIUM
        else:
            return StressLevel.LOW
    
    def _update_history(self, track_id: int, analysis: PoseAnalysis):
        """Update pose history for a track"""
        if track_id not in self.pose_history:
            self.pose_history[track_id] = []
        
        self.pose_history[track_id].append({
            'posture': analysis.posture,
            'stress_score': analysis.stress_score,
            'body_angle': analysis.body_angle
        })
        
        if len(self.pose_history[track_id]) > self.history_length:
            self.pose_history[track_id].pop(0)
    
    def _apply_temporal_analysis(self, track_id: int, analysis: PoseAnalysis) -> PoseAnalysis:
        """Apply temporal analysis using history"""
        history = self.pose_history.get(track_id, [])
        
        if len(history) < 2:
            return analysis
        
        # Calculate movement intensity from posture changes
        posture_changes = 0
        for i in range(1, len(history)):
            if history[i]['posture'] != history[i-1]['posture']:
                posture_changes += 1
        
        analysis.movement_intensity = posture_changes / len(history)
        
        # Check for stress escalation
        recent_stress = [h['stress_score'] for h in history[-5:]]
        if len(recent_stress) >= 3:
            if all(recent_stress[i] < recent_stress[i+1] for i in range(len(recent_stress)-1)):
                analysis.anomaly_indicators.append("stress_escalating")
                analysis.stress_score = min(1.0, analysis.stress_score + 0.2)
        
        return analysis
    
    def get_frame_summary(self, analyses: List[PoseAnalysis]) -> Dict:
        """Get summary of all pose analyses in frame"""
        if not analyses:
            return {
                'total_persons': 0,
                'stress_distribution': {},
                'anomalies': [],
                'highest_stress': None
            }
        
        stress_dist = {level.value: 0 for level in StressLevel}
        all_anomalies = []
        highest_stress = None
        
        for a in analyses:
            stress_dist[a.stress_level.value] += 1
            all_anomalies.extend(a.anomaly_indicators)
            
            if highest_stress is None or a.stress_score > highest_stress.stress_score:
                highest_stress = a
        
        return {
            'total_persons': len(analyses),
            'stress_distribution': stress_dist,
            'anomalies': list(set(all_anomalies)),
            'highest_stress': highest_stress
        }
    
    def cleanup(self):
        """Release resources"""
        if self.pose_detector is not None:
            self.pose_detector.close()
