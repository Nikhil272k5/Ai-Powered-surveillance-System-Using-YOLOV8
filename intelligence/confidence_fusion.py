"""
Multi-Signal Confidence Fusion Engine
Validates every alert before emission using multiple signals.

Fuses:
- Vision model confidence (YOLO)
- Temporal persistence (how long detected)
- Motion stability (consistent vs erratic)
- Crowd context (density appropriateness)
- Zone rules (restricted areas, entry/exit points)
- Historical accuracy (past false-alarm rate)
- Audio confirmation (optional)

Outputs:
- Trust Score (0-100)
- Human-readable explanation
- Suppression decision
"""

import time
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class SignalType(Enum):
    """Types of signals that can be fused"""
    VISION_CONFIDENCE = "vision_confidence"
    TEMPORAL_PERSISTENCE = "temporal_persistence"
    MOTION_STABILITY = "motion_stability"
    CROWD_CONTEXT = "crowd_context"
    ZONE_RULES = "zone_rules"
    HISTORICAL_ACCURACY = "historical_accuracy"
    AUDIO_CONFIRMATION = "audio_confirmation"
    NORMALITY_DEVIATION = "normality_deviation"


@dataclass
class Signal:
    """Individual signal for fusion"""
    signal_type: SignalType
    value: float  # 0.0 to 1.0
    weight: float
    explanation: str
    raw_data: Optional[Dict] = None


@dataclass
class FusionResult:
    """Result of confidence fusion"""
    trust_score: float  # 0-100
    is_suppressed: bool
    signals: List[Signal]
    explanation: str
    detailed_explanation: str
    alert_id: str
    timestamp: float


@dataclass
class AlertHistory:
    """Historical record for an alert type"""
    total_alerts: int = 0
    acknowledged: int = 0
    dismissed: int = 0
    timed_out: int = 0
    false_positive_rate: float = 0.0


class ConfidenceFusionEngine:
    """
    Multi-Signal Confidence Fusion Engine
    
    Aggregates multiple signals to compute a Trust Score for each alert.
    Provides human-readable explanations for why alerts are trusted or suppressed.
    """
    
    def __init__(self,
                 trust_threshold: int = 60,
                 weights: Optional[Dict[str, float]] = None,
                 log_suppressed: bool = True,
                 explanation_verbosity: str = "standard"):
        """
        Initialize the Confidence Fusion Engine.
        
        Args:
            trust_threshold: Minimum score (0-100) to emit alert
            weights: Signal weights (must sum to ~1.0)
            log_suppressed: Whether to log suppressed alerts
            explanation_verbosity: "minimal", "standard", or "detailed"
        """
        self.trust_threshold = trust_threshold
        self.log_suppressed = log_suppressed
        self.explanation_verbosity = explanation_verbosity
        
        # Default weights if not provided
        self.weights = weights or {
            SignalType.VISION_CONFIDENCE.value: 0.25,
            SignalType.TEMPORAL_PERSISTENCE.value: 0.20,
            SignalType.MOTION_STABILITY.value: 0.15,
            SignalType.CROWD_CONTEXT.value: 0.15,
            SignalType.ZONE_RULES.value: 0.10,
            SignalType.HISTORICAL_ACCURACY.value: 0.15,
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Alert history for learning
        self.alert_history: Dict[str, AlertHistory] = defaultdict(AlertHistory)
        
        # Recent alerts for temporal analysis
        self.recent_alerts: deque = deque(maxlen=1000)
        
        # Track history for motion analysis
        self.track_motion_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        
        # Statistics
        self.total_processed = 0
        self.total_suppressed = 0
        self.total_emitted = 0
        
        print(f"🔗 Confidence Fusion Engine initialized")
        print(f"   Trust threshold: {trust_threshold}%")
        print(f"   Active signals: {len(self.weights)}")
    
    def evaluate_alert(self,
                       alert: Dict[str, Any],
                       vision_confidence: float = 0.5,
                       tracked_objects: Optional[List] = None,
                       crowd_density: int = 0,
                       zone_info: Optional[Dict] = None,
                       audio_signal: Optional[Dict] = None,
                       normality_result: Optional[Dict] = None) -> FusionResult:
        """
        Evaluate an alert and compute its trust score.
        
        Args:
            alert: The alert dictionary from detection logic
            vision_confidence: YOLO detection confidence (0-1)
            tracked_objects: List of currently tracked objects
            crowd_density: Number of people in scene
            zone_info: Information about zones/regions
            audio_signal: Audio analysis results (optional)
            normality_result: Results from normality engine (optional)
        
        Returns:
            FusionResult with trust score and explanation
        """
        self.total_processed += 1
        alert_id = f"{alert.get('type', 'unknown')}_{alert.get('track_id', 0)}_{int(time.time())}"
        timestamp = time.time()
        
        signals = []
        
        # 1. Vision Confidence Signal
        vision_signal = self._compute_vision_signal(vision_confidence)
        signals.append(vision_signal)
        
        # 2. Temporal Persistence Signal
        temporal_signal = self._compute_temporal_signal(alert)
        signals.append(temporal_signal)
        
        # 3. Motion Stability Signal
        motion_signal = self._compute_motion_signal(alert, tracked_objects)
        signals.append(motion_signal)
        
        # 4. Crowd Context Signal
        crowd_signal = self._compute_crowd_signal(alert, crowd_density)
        signals.append(crowd_signal)
        
        # 5. Zone Rules Signal
        zone_signal = self._compute_zone_signal(alert, zone_info)
        signals.append(zone_signal)
        
        # 6. Historical Accuracy Signal
        history_signal = self._compute_history_signal(alert)
        signals.append(history_signal)
        
        # 7. Audio Confirmation Signal (optional)
        if audio_signal:
            audio_sig = self._compute_audio_signal(alert, audio_signal)
            signals.append(audio_sig)
        
        # 8. Normality Deviation Signal (optional)
        if normality_result:
            normality_sig = self._compute_normality_signal(normality_result)
            signals.append(normality_sig)
        
        # Compute weighted trust score
        trust_score = self._compute_trust_score(signals)
        
        # Determine if suppressed
        is_suppressed = trust_score < self.trust_threshold
        
        # Generate explanations
        explanation = self._generate_explanation(signals, trust_score, is_suppressed)
        detailed_explanation = self._generate_detailed_explanation(signals, trust_score)
        
        # Update statistics
        if is_suppressed:
            self.total_suppressed += 1
        else:
            self.total_emitted += 1
        
        # Store for history
        self.recent_alerts.append({
            'alert_id': alert_id,
            'type': alert.get('type'),
            'trust_score': trust_score,
            'is_suppressed': is_suppressed,
            'timestamp': timestamp
        })
        
        result = FusionResult(
            trust_score=trust_score,
            is_suppressed=is_suppressed,
            signals=signals,
            explanation=explanation,
            detailed_explanation=detailed_explanation,
            alert_id=alert_id,
            timestamp=timestamp
        )
        
        return result
    
    def _compute_vision_signal(self, confidence: float) -> Signal:
        """Compute signal from YOLO vision confidence"""
        # Scale confidence to 0-1 (it's usually already in this range)
        value = min(max(confidence, 0.0), 1.0)
        
        if value >= 0.8:
            explanation = "High detection confidence"
        elif value >= 0.5:
            explanation = "Moderate detection confidence"
        else:
            explanation = "Low detection confidence"
        
        return Signal(
            signal_type=SignalType.VISION_CONFIDENCE,
            value=value,
            weight=self.weights.get(SignalType.VISION_CONFIDENCE.value, 0.25),
            explanation=explanation,
            raw_data={'confidence': confidence}
        )
    
    def _compute_temporal_signal(self, alert: Dict) -> Signal:
        """Compute signal from how long the event has persisted"""
        alert_type = alert.get('type', 'unknown')
        track_id = alert.get('track_id', 0)
        
        # Count recent similar alerts
        recent_count = sum(
            1 for a in self.recent_alerts
            if a['type'] == alert_type and time.time() - a['timestamp'] < 30
        )
        
        # Also consider dwell time if available
        time_span = alert.get('time_span', 0)
        
        # Longer persistence = higher confidence
        if time_span > 10 or recent_count >= 3:
            value = 0.9
            explanation = "Event persisted over multiple observations"
        elif time_span > 5 or recent_count >= 2:
            value = 0.7
            explanation = "Event confirmed by repeated detection"
        elif recent_count >= 1:
            value = 0.5
            explanation = "Event detected recently"
        else:
            value = 0.3
            explanation = "New event, awaiting confirmation"
        
        return Signal(
            signal_type=SignalType.TEMPORAL_PERSISTENCE,
            value=value,
            weight=self.weights.get(SignalType.TEMPORAL_PERSISTENCE.value, 0.20),
            explanation=explanation,
            raw_data={'recent_count': recent_count, 'time_span': time_span}
        )
    
    def _compute_motion_signal(self, alert: Dict, tracked_objects: Optional[List]) -> Signal:
        """Compute signal from motion stability"""
        track_id = alert.get('track_id', 0)
        alert_type = alert.get('type', 'unknown')
        
        # Get motion history for this track
        history = self.track_motion_history.get(track_id, deque())
        
        if not tracked_objects:
            return Signal(
                signal_type=SignalType.MOTION_STABILITY,
                value=0.5,
                weight=self.weights.get(SignalType.MOTION_STABILITY.value, 0.15),
                explanation="No motion data available",
                raw_data={}
            )
        
        # Find the object in tracked_objects
        obj_data = None
        for obj in tracked_objects:
            if obj[0] == track_id:
                obj_data = obj
                break
        
        if obj_data:
            x1, y1, x2, y2 = obj_data[1:5]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.track_motion_history[track_id].append({
                'position': center,
                'timestamp': time.time()
            })
        
        history = self.track_motion_history.get(track_id, deque())
        
        if len(history) < 3:
            value = 0.5
            explanation = "Insufficient motion history"
        else:
            # Calculate motion consistency
            positions = [h['position'] for h in history]
            velocities = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocities.append((dx, dy))
            
            if velocities:
                # Calculate velocity variance
                vx_var = np.var([v[0] for v in velocities])
                vy_var = np.var([v[1] for v in velocities])
                total_var = vx_var + vy_var
                
                # Low variance = stable motion = higher confidence
                if total_var < 10:
                    value = 0.9
                    explanation = "Very stable motion pattern"
                elif total_var < 50:
                    value = 0.7
                    explanation = "Fairly consistent motion"
                elif total_var < 200:
                    value = 0.5
                    explanation = "Moderate motion variability"
                else:
                    value = 0.3
                    explanation = "Erratic motion pattern"
                    
                # For speed spike alerts, high variance might be expected
                if alert_type == 'speed_spike' and total_var > 100:
                    value = 0.8
                    explanation = "Motion variability consistent with speed spike"
            else:
                value = 0.5
                explanation = "No velocity data"
        
        return Signal(
            signal_type=SignalType.MOTION_STABILITY,
            value=value,
            weight=self.weights.get(SignalType.MOTION_STABILITY.value, 0.15),
            explanation=explanation,
            raw_data={'history_length': len(history)}
        )
    
    def _compute_crowd_signal(self, alert: Dict, crowd_density: int) -> Signal:
        """Compute signal based on crowd context"""
        alert_type = alert.get('type', 'unknown')
        
        # Different alert types have different crowd expectations
        if alert_type in ['speed_spike', 'counterflow']:
            # Speed spikes in crowded areas are more concerning
            if crowd_density > 5:
                value = 0.9
                explanation = "High crowd density increases concern"
            elif crowd_density > 2:
                value = 0.7
                explanation = "Moderate crowd presence"
            else:
                value = 0.5
                explanation = "Low crowd density"
        
        elif alert_type == 'loitering':
            # Loitering is less concerning in busy areas
            if crowd_density > 5:
                value = 0.4
                explanation = "Busy area, loitering less suspicious"
            elif crowd_density > 2:
                value = 0.6
                explanation = "Moderate crowd, context neutral"
            else:
                value = 0.8
                explanation = "Empty area, loitering more suspicious"
        
        elif alert_type == 'abandoned_object':
            # Abandoned objects are concerning regardless of crowd
            if crowd_density > 3:
                value = 0.8
                explanation = "Object left in crowded area"
            else:
                value = 0.6
                explanation = "Object left in less crowded area"
        
        else:
            value = 0.5
            explanation = f"Crowd density: {crowd_density}"
        
        return Signal(
            signal_type=SignalType.CROWD_CONTEXT,
            value=value,
            weight=self.weights.get(SignalType.CROWD_CONTEXT.value, 0.15),
            explanation=explanation,
            raw_data={'crowd_density': crowd_density}
        )
    
    def _compute_zone_signal(self, alert: Dict, zone_info: Optional[Dict]) -> Signal:
        """Compute signal based on zone rules"""
        if not zone_info:
            return Signal(
                signal_type=SignalType.ZONE_RULES,
                value=0.5,
                weight=self.weights.get(SignalType.ZONE_RULES.value, 0.10),
                explanation="No zone rules configured",
                raw_data={}
            )
        
        position = alert.get('position', (0, 0))
        restricted_areas = zone_info.get('restricted_areas', [])
        entry_points = zone_info.get('entry_points', [])
        exit_points = zone_info.get('exit_points', [])
        
        # Check if in restricted area
        in_restricted = self._point_in_zones(position, restricted_areas)
        near_entry = self._point_near_zones(position, entry_points, threshold=100)
        near_exit = self._point_near_zones(position, exit_points, threshold=100)
        
        if in_restricted:
            value = 0.95
            explanation = "Event in restricted area"
        elif near_exit and alert.get('type') == 'counterflow':
            value = 0.85
            explanation = "Counterflow near exit point"
        elif near_entry or near_exit:
            value = 0.6
            explanation = "Event near entry/exit point"
        else:
            value = 0.5
            explanation = "Standard zone"
        
        return Signal(
            signal_type=SignalType.ZONE_RULES,
            value=value,
            weight=self.weights.get(SignalType.ZONE_RULES.value, 0.10),
            explanation=explanation,
            raw_data={'in_restricted': in_restricted, 'near_entry': near_entry, 'near_exit': near_exit}
        )
    
    def _compute_history_signal(self, alert: Dict) -> Signal:
        """Compute signal based on historical accuracy for this alert type"""
        alert_type = alert.get('type', 'unknown')
        history = self.alert_history.get(alert_type, AlertHistory())
        
        if history.total_alerts < 10:
            value = 0.5
            explanation = "Insufficient history for this alert type"
        else:
            # Higher acknowledged rate = higher confidence
            acknowledged_rate = history.acknowledged / history.total_alerts
            false_positive_rate = history.dismissed / history.total_alerts
            
            if false_positive_rate < 0.1:
                value = 0.9
                explanation = "Alert type has excellent historical accuracy"
            elif false_positive_rate < 0.3:
                value = 0.7
                explanation = "Alert type has good historical accuracy"
            elif false_positive_rate < 0.5:
                value = 0.5
                explanation = "Alert type has moderate historical accuracy"
            else:
                value = 0.3
                explanation = "Alert type has poor historical accuracy"
        
        return Signal(
            signal_type=SignalType.HISTORICAL_ACCURACY,
            value=value,
            weight=self.weights.get(SignalType.HISTORICAL_ACCURACY.value, 0.15),
            explanation=explanation,
            raw_data={
                'total': history.total_alerts,
                'acknowledged': history.acknowledged,
                'false_positive_rate': history.false_positive_rate
            }
        )
    
    def _compute_audio_signal(self, alert: Dict, audio_signal: Dict) -> Signal:
        """Compute signal from audio analysis"""
        audio_detected = audio_signal.get('detected', False)
        audio_type = audio_signal.get('type', 'unknown')
        audio_confidence = audio_signal.get('confidence', 0.0)
        
        alert_type = alert.get('type', 'unknown')
        
        # Audio confirmation is valuable for certain alert types
        if audio_detected:
            if audio_type == 'scream' and alert_type in ['speed_spike', 'panic']:
                value = 0.95
                explanation = f"Audio confirms panic ({audio_type})"
            elif audio_type == 'glass_break' and alert_type == 'abandoned_object':
                value = 0.85
                explanation = "Audio detected glass breaking"
            elif audio_type == 'loud_noise':
                value = 0.7
                explanation = f"Unusual audio detected ({audio_confidence:.0%} confidence)"
            else:
                value = 0.6
                explanation = "Audio signal detected"
        else:
            value = 0.5
            explanation = "No significant audio detected"
        
        return Signal(
            signal_type=SignalType.AUDIO_CONFIRMATION,
            value=value,
            weight=0.10,  # Extra weight for audio
            explanation=explanation,
            raw_data=audio_signal
        )
    
    def _compute_normality_signal(self, normality_result: Dict) -> Signal:
        """Compute signal from normality engine deviation"""
        anomaly_score = normality_result.get('anomaly_score', 0.0)
        anomaly_types = normality_result.get('anomaly_types', [])
        
        # Higher anomaly score = more confidence in alert
        value = min(0.5 + anomaly_score * 0.5, 1.0)
        
        if anomaly_score > 0.8:
            explanation = f"Strong deviation from normal ({', '.join(anomaly_types)})"
        elif anomaly_score > 0.5:
            explanation = f"Moderate deviation from normal ({', '.join(anomaly_types)})"
        elif anomaly_score > 0.2:
            explanation = "Slight deviation from normal"
        else:
            explanation = "Within normal parameters"
        
        return Signal(
            signal_type=SignalType.NORMALITY_DEVIATION,
            value=value,
            weight=0.15,  # Extra weight for normality
            explanation=explanation,
            raw_data=normality_result
        )
    
    def _compute_trust_score(self, signals: List[Signal]) -> float:
        """Compute weighted trust score from all signals"""
        total_weight = sum(s.weight for s in signals)
        
        if total_weight == 0:
            return 50.0
        
        weighted_sum = sum(s.value * s.weight for s in signals)
        normalized_score = weighted_sum / total_weight
        
        # Scale to 0-100
        return round(normalized_score * 100, 1)
    
    def _generate_explanation(self, signals: List[Signal], trust_score: float, 
                             is_suppressed: bool) -> str:
        """Generate human-readable explanation"""
        if is_suppressed:
            # Find the weakest signals
            weak_signals = sorted(signals, key=lambda s: s.value)[:2]
            reasons = [s.explanation for s in weak_signals if s.value < 0.5]
            
            if reasons:
                return f"Suppressed (score: {trust_score:.0f}%) - {'; '.join(reasons)}"
            else:
                return f"Suppressed (score: {trust_score:.0f}%) - Below trust threshold"
        else:
            # Find the strongest signals
            strong_signals = sorted(signals, key=lambda s: s.value, reverse=True)[:2]
            reasons = [s.explanation for s in strong_signals if s.value >= 0.7]
            
            if reasons:
                return f"Trusted (score: {trust_score:.0f}%) - {'; '.join(reasons)}"
            else:
                return f"Trusted (score: {trust_score:.0f}%)"
    
    def _generate_detailed_explanation(self, signals: List[Signal], 
                                       trust_score: float) -> str:
        """Generate detailed breakdown explanation"""
        lines = [f"Trust Score: {trust_score:.0f}%", ""]
        lines.append("Signal Breakdown:")
        
        for signal in sorted(signals, key=lambda s: s.value * s.weight, reverse=True):
            contribution = signal.value * signal.weight * 100
            lines.append(
                f"  • {signal.signal_type.value}: {signal.value:.0%} "
                f"(weight: {signal.weight:.0%}, contributes: {contribution:.1f}%)"
            )
            lines.append(f"    → {signal.explanation}")
        
        return "\n".join(lines)
    
    def _point_in_zones(self, point: Tuple[float, float], 
                        zones: List[List[Tuple[float, float]]]) -> bool:
        """Check if point is inside any zone polygon"""
        # Simple bounding box check for now
        # TODO: Implement proper polygon containment
        for zone in zones:
            if len(zone) >= 4:
                min_x = min(p[0] for p in zone)
                max_x = max(p[0] for p in zone)
                min_y = min(p[1] for p in zone)
                max_y = max(p[1] for p in zone)
                
                if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                    return True
        return False
    
    def _point_near_zones(self, point: Tuple[float, float],
                          zones: List[Tuple[float, float]], 
                          threshold: float) -> bool:
        """Check if point is near any zone point"""
        for zone_point in zones:
            distance = np.sqrt(
                (point[0] - zone_point[0]) ** 2 + 
                (point[1] - zone_point[1]) ** 2
            )
            if distance < threshold:
                return True
        return False
    
    def record_feedback(self, alert_id: str, alert_type: str, 
                       outcome: str) -> None:
        """
        Record feedback for an alert.
        
        Args:
            alert_id: The alert identifier
            alert_type: Type of alert (e.g., 'speed_spike', 'loitering')
            outcome: 'acknowledged', 'dismissed', or 'timed_out'
        """
        history = self.alert_history[alert_type]
        history.total_alerts += 1
        
        if outcome == 'acknowledged':
            history.acknowledged += 1
        elif outcome == 'dismissed':
            history.dismissed += 1
        elif outcome == 'timed_out':
            history.timed_out += 1
        
        # Update false positive rate
        if history.total_alerts > 0:
            history.false_positive_rate = (
                history.dismissed + history.timed_out * 0.5
            ) / history.total_alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        suppression_rate = (
            self.total_suppressed / self.total_processed 
            if self.total_processed > 0 else 0
        )
        
        return {
            'total_processed': self.total_processed,
            'total_emitted': self.total_emitted,
            'total_suppressed': self.total_suppressed,
            'suppression_rate': suppression_rate,
            'trust_threshold': self.trust_threshold,
            'alert_type_history': {
                k: {
                    'total': v.total_alerts,
                    'acknowledged': v.acknowledged,
                    'dismissed': v.dismissed,
                    'false_positive_rate': v.false_positive_rate
                }
                for k, v in self.alert_history.items()
            }
        }
    
    def update_threshold(self, new_threshold: int) -> None:
        """Update the trust threshold"""
        old_threshold = self.trust_threshold
        self.trust_threshold = max(0, min(100, new_threshold))
        print(f"🔧 Trust threshold updated: {old_threshold}% → {self.trust_threshold}%")
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update signal weights"""
        self.weights.update(new_weights)
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        print(f"🔧 Signal weights updated")
