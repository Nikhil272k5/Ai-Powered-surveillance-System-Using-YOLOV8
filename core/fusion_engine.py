"""
AbnoGuard Core - Dual-Brain Fusion Engine
Fuses Perception (Brain-1) with Cognition (Brain-2)
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import perception modules
from perception.fusion import PerceptionFusion, FramePerception, FusedPerception, ThreatLevel
from perception.scene_zones import SceneZoneAnalyzer

# Import cognition modules
from cognition.narrative_engine import NarrativeEngine, NarrativeStyle
from cognition.temporal_memory import TemporalMemory, MemoryEntry
from cognition.risk_matrix import RiskMatrix, RiskAssessment, RiskCategory
from cognition.ethics_layer import EthicsLayer, AlertModification
from cognition.explainability import ExplainabilityEngine, ExplainableDecision
from cognition.brief_generator import BriefGenerator, IncidentBrief


@dataclass
class IntelligenceOutput:
    """Complete output from the dual-brain system"""
    frame_id: int
    timestamp: float
    
    # From Perception
    perception: FramePerception
    
    # From Cognition
    narratives: List[str]
    risk_assessments: List[RiskAssessment]
    ethical_assessments: List[Dict]
    explanations: List[ExplainableDecision]
    
    # Fused decisions
    final_alerts: List[Dict]
    incident_briefs: List[IncidentBrief]
    
    # Memory updates
    new_memories: List[str]  # memory IDs
    triggered_patterns: List[str]
    
    # System state
    system_confidence: float
    processing_time_ms: float


class DualBrainFusion:
    """
    The core fusion engine that connects Perception and Cognition.
    
    Flow:
    1. Perception processes frame → detections, motion, pose, zones
    2. Cognition analyzes → risk, ethics, history
    3. Fusion combines → final decision
    4. Memory stores → for future learning
    5. Output generates → alerts, briefs, narratives
    
    Feedback loops:
    - Cognition adjusts perception thresholds
    - Memory affects alert sensitivity
    - Ethics modifies severity
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Initialize Brain-1: Perception
        print("🧠 Initializing Brain-1: Perception...")
        self.perception = PerceptionFusion(config.get('perception', {}))
        
        # Initialize Brain-2: Cognition
        print("🧠 Initializing Brain-2: Cognition...")
        self.narrative = NarrativeEngine(config.get('narrative', {}))
        self.memory = TemporalMemory(config.get('memory', {}))
        self.risk = RiskMatrix(config.get('risk', {}))
        self.ethics = EthicsLayer(config.get('ethics', {}))
        self.explain = ExplainabilityEngine(config.get('explain', {}))
        self.brief_gen = BriefGenerator(config.get('brief', {}))
        
        # State tracking
        self.frame_count = 0
        self.active_incidents: Dict[str, Dict] = {}
        self.alert_cooldowns: Dict[int, float] = {}  # track_id -> last_alert_time
        self.cooldown_seconds = self.config.get('alert_cooldown', 30)
        
        # Adaptive thresholds (adjusted by cognition)
        self.adaptive_thresholds = {
            'threat_sensitivity': 1.0,  # Multiplier for threat scores
            'alert_threshold': 0.5,     # Minimum score to alert
        }
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'alerts_generated': 0,
            'alerts_suppressed': 0,
            'false_positives_marked': 0,
            'avg_processing_time': 0
        }
        
        print("🔗 Dual-Brain Fusion Engine ready!")
    
    def process(self, frame: np.ndarray, detections: List[Dict],
               tracked_objects: List[Tuple]) -> IntelligenceOutput:
        """
        Process a frame through both brains and fuse results
        
        Args:
            frame: BGR image
            detections: Raw YOLO detections
            tracked_objects: Tracked objects list
        
        Returns:
            IntelligenceOutput with complete analysis
        """
        start_time = time.time()
        self.frame_count += 1
        current_time = time.time()
        
        # =====================================================
        # BRAIN-1: Perception
        # =====================================================
        perception_result = self.perception.process_frame(
            frame, detections, tracked_objects
        )
        
        # =====================================================
        # BRAIN-2: Cognition
        # =====================================================
        narratives = []
        risk_assessments = []
        ethical_assessments = []
        explanations = []
        final_alerts = []
        briefs = []
        new_memories = []
        patterns = []
        
        # Process each entity through cognition
        for entity in perception_result.entities:
            # Skip low-threat entities for detailed analysis
            if entity.threat_level == ThreatLevel.NONE:
                continue
            
            # Prepare entity data for cognition
            entity_data = self._prepare_entity_data(entity)
            
            # Risk assessment
            risk_assessment = self.risk.assess_risk(entity_data)
            risk_assessments.append(risk_assessment)
            
            # Historical comparison
            history_comparison = self.memory.compare_to_history({
                'type': self._get_incident_type(entity),
                'zone_id': entity.current_zone,
                'threat_score': entity.threat_score
            })
            
            if history_comparison.get('patterns_triggered'):
                patterns.extend(history_comparison['patterns_triggered'])
            
            # Determine if we should alert
            should_alert = self._should_generate_alert(
                entity, risk_assessment, history_comparison, current_time
            )
            
            if should_alert:
                # Create alert
                alert = self._create_alert(entity, risk_assessment)
                
                # Ethical evaluation
                ethical_eval = self.ethics.evaluate(alert, {
                    'zone_type': entity.zone_type,
                    'duration': entity.zone_dwell_time,
                    'hour': time.localtime().tm_hour
                })
                ethical_assessments.append({
                    'track_id': entity.track_id,
                    'modification': ethical_eval.modification.value,
                    'reasoning': ethical_eval.reasoning
                })
                
                if ethical_eval.should_alert:
                    # Apply ethical modifications
                    alert['severity'] = ethical_eval.modified_severity
                    
                    # Generate explanation
                    explanation = self.explain.explain_alert(
                        alert,
                        perception_data={
                            'detection_confidence': entity.detection_confidence,
                            'speed': entity.speed,
                            'stress_score': entity.stress_score,
                            'zone_violation': entity.zone_violation,
                            'current_zone': entity.current_zone
                        },
                        cognition_data={
                            'risk_score': risk_assessment.risk_score,
                            'is_unusual': history_comparison.get('is_unusual'),
                            'historical_context': history_comparison.get('historical_context'),
                            'patterns_triggered': history_comparison.get('patterns_triggered', [])
                        }
                    )
                    explanations.append(explanation)
                    
                    # Generate narrative
                    narrative = self.narrative.narrate_from_alert(alert)
                    narratives.append(narrative)
                    
                    # Add to final alerts
                    alert['explanation'] = self.explain.get_brief_explanation(explanation)
                    alert['narrative'] = narrative
                    final_alerts.append(alert)
                    
                    # Store in memory
                    memory_entry = MemoryEntry(
                        memory_id=f"MEM-{int(current_time)}-{entity.track_id}",
                        timestamp=current_time,
                        incident_type=alert['type'],
                        threat_level=alert['severity'],
                        zone_id=entity.current_zone,
                        track_ids=[entity.track_id],
                        duration=entity.zone_dwell_time,
                        description=alert['description'],
                        narrative=narrative,
                        outcome=None,  # To be updated later
                        metadata={'risk_score': risk_assessment.risk_score}
                    )
                    mem_id = self.memory.remember(memory_entry)
                    new_memories.append(mem_id)
                    
                    # Update cooldown
                    self.alert_cooldowns[entity.track_id] = current_time
                    
                    # Generate brief for high severity
                    if alert['severity'] in ['high', 'critical']:
                        brief = self.brief_gen.generate_brief(
                            incident_data={
                                'type': alert['type'],
                                'severity': alert['severity'],
                                'start_time': current_time,
                                'zone': entity.current_zone or 'Unknown',
                                'track_ids': [entity.track_id],
                                'confidence': explanation.overall_confidence
                            },
                            narrative=narrative,
                            risk_assessment={
                                'risk_score': risk_assessment.risk_score,
                                'risk_category': risk_assessment.risk_category
                            }
                        )
                        briefs.append(brief)
                    
                    self.stats['alerts_generated'] += 1
                else:
                    self.stats['alerts_suppressed'] += 1
        
        # Update adaptive thresholds based on patterns
        self._update_adaptive_thresholds(patterns, final_alerts)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['frames_processed'] = self.frame_count
        self.stats['avg_processing_time'] = (
            self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
        )
        
        # Compute system confidence
        system_confidence = self._compute_system_confidence(
            perception_result, risk_assessments
        )
        
        return IntelligenceOutput(
            frame_id=self.frame_count,
            timestamp=current_time,
            perception=perception_result,
            narratives=narratives,
            risk_assessments=risk_assessments,
            ethical_assessments=ethical_assessments,
            explanations=explanations,
            final_alerts=final_alerts,
            incident_briefs=briefs,
            new_memories=new_memories,
            triggered_patterns=list(set(patterns)),
            system_confidence=system_confidence,
            processing_time_ms=processing_time
        )
    
    def _prepare_entity_data(self, entity: FusedPerception) -> Dict:
        """Prepare entity data for cognition modules"""
        behaviors = entity.anomaly_reasons.copy()
        
        if entity.speed > 15:
            behaviors.append('running')
        if entity.stress_score > 0.5:
            behaviors.append('stress_detected')
        if entity.zone_violation:
            behaviors.append('restricted_zone_entry')
        if entity.posture in ['crouching', 'fallen']:
            behaviors.append(entity.posture)
        
        context = []
        if entity.zone_type == 'restricted':
            context.append('restricted_zone')
        if entity.zone_dwell_time > 120:
            context.append('prolonged_duration')
        
        return {
            'track_id': entity.track_id,
            'behaviors': behaviors,
            'context': context,
            'threat_score': entity.threat_score,
            'zone_type': entity.zone_type,
            'duration': entity.zone_dwell_time
        }
    
    def _get_incident_type(self, entity: FusedPerception) -> str:
        """Determine incident type from entity"""
        if entity.zone_violation:
            return 'restricted_zone_entry'
        if entity.speed > 15:
            return 'speed_spike'
        if entity.zone_dwell_time > 120 and entity.zone_type == 'transit':
            return 'loitering'
        if entity.stress_score > 0.7:
            return 'distress'
        if 'counterflow' in entity.motion_pattern:
            return 'counterflow'
        return 'anomaly'
    
    def _should_generate_alert(self, entity: FusedPerception,
                               risk: RiskAssessment,
                               history: Dict,
                               current_time: float) -> bool:
        """Determine if we should generate an alert"""
        # Check cooldown
        last_alert = self.alert_cooldowns.get(entity.track_id, 0)
        if current_time - last_alert < self.cooldown_seconds:
            return False
        
        # Apply adaptive threshold
        adjusted_score = entity.threat_score * self.adaptive_thresholds['threat_sensitivity']
        threshold = self.adaptive_thresholds['alert_threshold']
        
        # Lower threshold for unusual events
        if history.get('is_unusual'):
            threshold *= 0.8
        
        # Higher threshold for repeated patterns (might be false positive)
        if history.get('similar_incidents_24h', 0) > 5:
            threshold *= 1.2
        
        return adjusted_score >= threshold
    
    def _create_alert(self, entity: FusedPerception, risk: RiskAssessment) -> Dict:
        """Create alert dictionary"""
        return {
            'type': self._get_incident_type(entity),
            'track_id': entity.track_id,
            'timestamp': time.time(),
            'severity': risk.risk_category.value if hasattr(risk.risk_category, 'value') else str(risk.risk_category),
            'threat_score': entity.threat_score,
            'risk_score': risk.risk_score,
            'description': f"Track {entity.track_id}: {', '.join(entity.anomaly_reasons)}",
            'zone': entity.current_zone,
            'position': entity.bbox
        }
    
    def _update_adaptive_thresholds(self, patterns: List[str], alerts: List[Dict]):
        """Update thresholds based on feedback loops"""
        # If many patterns are triggering, increase sensitivity
        if len(patterns) > 3:
            self.adaptive_thresholds['threat_sensitivity'] = min(1.5, 
                self.adaptive_thresholds['threat_sensitivity'] + 0.05)
        else:
            # Slowly return to baseline
            self.adaptive_thresholds['threat_sensitivity'] = max(0.8,
                self.adaptive_thresholds['threat_sensitivity'] * 0.99)
    
    def _compute_system_confidence(self, perception: FramePerception,
                                   risks: List[RiskAssessment]) -> float:
        """Compute overall system confidence"""
        if not perception.entities:
            return 1.0
        
        # Average detection confidence
        det_conf = np.mean([e.detection_confidence for e in perception.entities])
        
        # Average risk assessment uncertainty
        if risks:
            risk_conf = np.mean([r.intent_confidence for r in risks])
        else:
            risk_conf = 1.0
        
        return (det_conf + risk_conf) / 2
    
    def mark_false_positive(self, memory_id: str):
        """Mark an alert as false positive for learning"""
        memory = self.memory.recall(memory_id)
        if memory:
            memory.outcome = 'false_positive'
            self.memory.remember(memory)  # Update
            self.stats['false_positives_marked'] += 1
            
            # Adjust thresholds
            self.adaptive_thresholds['alert_threshold'] *= 1.05
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'frames_processed': self.frame_count,
            'alerts_generated': self.stats['alerts_generated'],
            'alerts_suppressed': self.stats['alerts_suppressed'],
            'false_positives': self.stats['false_positives_marked'],
            'avg_processing_time_ms': self.stats['avg_processing_time'],
            'adaptive_thresholds': self.adaptive_thresholds,
            'active_patterns': len(self.memory.get_active_patterns()),
            'memory_stats': self.memory.get_statistics(7)
        }
    
    def cleanup(self):
        """Release resources"""
        self.perception.cleanup()


# Convenience function for quick setup
def create_intelligence_system(config_path: Optional[str] = None) -> DualBrainFusion:
    """Create a configured dual-brain intelligence system"""
    config = {}
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    return DualBrainFusion(config)
