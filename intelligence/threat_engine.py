"""
Threat Intelligence Engine - Central Brain of the Security Platform

Correlates multiple weak signals into meaningful threats.
Produces semantic threat conclusions, not raw object detections.
"""
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import threading
import json

from .event_bus import (
    EventBus, SecurityEvent, EventType, 
    event_bus, create_threat_event
)


class ThreatSeverity(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of correlated threats"""
    HOSTILE_RECONNAISSANCE = "hostile_reconnaissance"
    SUSPICIOUS_LOITERING = "suspicious_loitering"
    PERIMETER_BREACH = "perimeter_breach"
    ABANDONED_OBJECT = "abandoned_object"
    COORDINATED_MOVEMENT = "coordinated_movement"
    EVASIVE_BEHAVIOR = "evasive_behavior"
    SECURITY_PROBING = "security_probing"
    CROWD_ANOMALY = "crowd_anomaly"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    TAILGATING = "tailgating"
    DISTRACTION_PATTERN = "distraction_pattern"
    PRE_ATTACK_INDICATOR = "pre_attack_indicator"
    UNKNOWN = "unknown"


@dataclass
class WeakSignal:
    """A single weak signal from Layer-1 or Layer-2"""
    signal_id: str
    signal_type: str
    entity_id: str
    zone_id: str
    timestamp: datetime
    confidence: float
    weight: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatConclusion:
    """Semantic threat conclusion from correlated signals"""
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    title: str
    description: str
    confidence: float
    uncertainty: float
    entity_ids: List[str]
    zone_ids: List[str]
    evidence: List[WeakSignal]
    timestamp: datetime
    is_active: bool = True
    escalation_count: int = 0
    resolution_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "entity_ids": self.entity_ids,
            "zone_ids": self.zone_ids,
            "evidence_count": len(self.evidence),
            "timestamp": self.timestamp.isoformat(),
            "is_active": self.is_active,
            "escalation_count": self.escalation_count
        }
    
    def get_explanation(self) -> str:
        """Generate human-readable explanation of the threat"""
        evidence_summary = []
        for sig in self.evidence[:5]:  # Top 5 signals
            evidence_summary.append(f"- {sig.signal_type} at {sig.timestamp.strftime('%H:%M:%S')} (conf: {sig.confidence:.0%})")
        
        return f"""
THREAT ANALYSIS: {self.title}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {self.threat_type.value.replace('_', ' ').title()}
Severity: {self.severity.value.upper()}
Confidence: {self.confidence:.0%} (Uncertainty: {self.uncertainty:.0%})

DESCRIPTION:
{self.description}

EVIDENCE ({len(self.evidence)} signals):
{chr(10).join(evidence_summary)}

ENTITIES INVOLVED: {', '.join(self.entity_ids[:5])}
ZONES AFFECTED: {', '.join(self.zone_ids)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class CorrelationRule:
    """Rule for correlating signals into threats"""
    
    def __init__(
        self,
        rule_id: str,
        threat_type: ThreatType,
        required_signals: List[str],
        optional_signals: List[str] = None,
        min_confidence: float = 0.5,
        time_window_seconds: int = 300,
        zone_required: bool = True,
        description_template: str = ""
    ):
        self.rule_id = rule_id
        self.threat_type = threat_type
        self.required_signals = required_signals
        self.optional_signals = optional_signals or []
        self.min_confidence = min_confidence
        self.time_window_seconds = time_window_seconds
        self.zone_required = zone_required
        self.description_template = description_template
    
    def evaluate(self, signals: List[WeakSignal]) -> Optional[Tuple[float, str]]:
        """
        Evaluate if signals match this rule.
        Returns (confidence, description) or None.
        """
        if not signals:
            return None
        
        # Check time window
        now = datetime.now()
        window_start = now - timedelta(seconds=self.time_window_seconds)
        recent_signals = [s for s in signals if s.timestamp >= window_start]
        
        if not recent_signals:
            return None
        
        # Check required signals
        signal_types = {s.signal_type for s in recent_signals}
        
        required_met = all(req in signal_types for req in self.required_signals)
        if not required_met:
            return None
        
        # Calculate confidence
        base_confidence = sum(s.confidence * s.weight for s in recent_signals) / len(recent_signals)
        
        # Boost for optional signals
        optional_boost = sum(0.1 for opt in self.optional_signals if opt in signal_types)
        confidence = min(1.0, base_confidence + optional_boost)
        
        if confidence < self.min_confidence:
            return None
        
        # Generate description
        description = self.description_template.format(
            entity_count=len({s.entity_id for s in recent_signals}),
            zone_count=len({s.zone_id for s in recent_signals}),
            signal_count=len(recent_signals)
        )
        
        return (confidence, description)


class ThreatIntelligenceEngine:
    """
    Central Threat Intelligence Engine
    
    Correlates multiple weak signals into meaningful threats.
    Produces semantic conclusions, not raw detections.
    """
    
    def __init__(self):
        self.signals: Dict[str, List[WeakSignal]] = defaultdict(list)  # by entity_id
        self.zone_signals: Dict[str, List[WeakSignal]] = defaultdict(list)  # by zone_id
        self.active_threats: Dict[str, ThreatConclusion] = {}
        self.resolved_threats: List[ThreatConclusion] = []
        self.correlation_rules: List[CorrelationRule] = []
        self._lock = threading.Lock()
        self._enabled = True
        
        # Initialize default correlation rules
        self._init_default_rules()
        
        # Subscribe to events
        event_bus.subscribe(EventType.DETECTION, self._on_detection)
        event_bus.subscribe(EventType.BEHAVIOR_DETECTED, self._on_behavior)
        event_bus.subscribe(EventType.ANOMALY_SCORED, self._on_anomaly)
    
    def _init_default_rules(self):
        """Initialize default correlation rules"""
        
        # Rule 1: Hostile Reconnaissance
        self.add_rule(CorrelationRule(
            rule_id="hostile_recon_01",
            threat_type=ThreatType.HOSTILE_RECONNAISSANCE,
            required_signals=["loitering", "repeated_presence"],
            optional_signals=["camera_avoidance", "perimeter_approach"],
            min_confidence=0.6,
            time_window_seconds=600,
            description_template="Potential hostile reconnaissance detected. {entity_count} subject(s) showing repeated presence with loitering behavior across {zone_count} zone(s)."
        ))
        
        # Rule 2: Suspicious Loitering
        self.add_rule(CorrelationRule(
            rule_id="suspicious_loiter_01",
            threat_type=ThreatType.SUSPICIOUS_LOITERING,
            required_signals=["loitering"],
            optional_signals=["unusual_time", "restricted_zone"],
            min_confidence=0.5,
            time_window_seconds=300,
            description_template="Suspicious loitering detected. Subject lingering in monitored area beyond normal duration."
        ))
        
        # Rule 3: Abandoned Object
        self.add_rule(CorrelationRule(
            rule_id="abandoned_obj_01",
            threat_type=ThreatType.ABANDONED_OBJECT,
            required_signals=["bag_detected", "person_departed"],
            optional_signals=["unattended_duration"],
            min_confidence=0.7,
            time_window_seconds=180,
            description_template="Possible abandoned object. Bag/package left unattended after owner departure."
        ))
        
        # Rule 4: Evasive Behavior
        self.add_rule(CorrelationRule(
            rule_id="evasive_01",
            threat_type=ThreatType.EVASIVE_BEHAVIOR,
            required_signals=["camera_avoidance", "erratic_movement"],
            optional_signals=["direction_change", "blind_spot_seeking"],
            min_confidence=0.65,
            time_window_seconds=120,
            description_template="Evasive behavior pattern detected. Subject actively avoiding camera coverage with erratic movements."
        ))
        
        # Rule 5: Coordinated Movement
        self.add_rule(CorrelationRule(
            rule_id="coordinated_01",
            threat_type=ThreatType.COORDINATED_MOVEMENT,
            required_signals=["group_formation", "synchronized_movement"],
            optional_signals=["hand_signals", "sequential_entry"],
            min_confidence=0.7,
            time_window_seconds=300,
            description_template="Coordinated group movement detected. {entity_count} individuals showing synchronized behavior patterns."
        ))
        
        # Rule 6: Security Probing
        self.add_rule(CorrelationRule(
            rule_id="security_probe_01",
            threat_type=ThreatType.SECURITY_PROBING,
            required_signals=["perimeter_test", "access_attempt"],
            optional_signals=["guard_observation", "timing_test"],
            min_confidence=0.6,
            time_window_seconds=600,
            description_template="Security probing behavior detected. Subject testing security measures and response times."
        ))
        
        # Rule 7: Pre-Attack Indicator
        self.add_rule(CorrelationRule(
            rule_id="pre_attack_01",
            threat_type=ThreatType.PRE_ATTACK_INDICATOR,
            required_signals=["dry_run", "reconnaissance", "timing_observation"],
            optional_signals=["equipment_check", "exit_mapping"],
            min_confidence=0.8,
            time_window_seconds=1800,
            description_template="CRITICAL: Pre-attack indicators detected. Multiple reconnaissance behaviors suggesting planning activity."
        ))
    
    def add_rule(self, rule: CorrelationRule):
        """Add a correlation rule"""
        self.correlation_rules.append(rule)
    
    def add_signal(self, signal: WeakSignal):
        """Add a weak signal for correlation"""
        with self._lock:
            self.signals[signal.entity_id].append(signal)
            self.zone_signals[signal.zone_id].append(signal)
            
            # Clean old signals
            cutoff = datetime.now() - timedelta(hours=1)
            for entity_id in list(self.signals.keys()):
                self.signals[entity_id] = [
                    s for s in self.signals[entity_id] if s.timestamp > cutoff
                ]
        
        # Trigger correlation
        self._correlate_signals(signal.entity_id, signal.zone_id)
    
    def _correlate_signals(self, entity_id: str, zone_id: str):
        """Correlate signals and generate threats"""
        if not self._enabled:
            return
        
        with self._lock:
            entity_signals = self.signals.get(entity_id, [])
            zone_signals = self.zone_signals.get(zone_id, [])
            
            # Deduplicate by signal_id (can't use set() because WeakSignal is unhashable)
            seen_ids = set()
            all_signals = []
            for sig in entity_signals + zone_signals:
                if sig.signal_id not in seen_ids:
                    seen_ids.add(sig.signal_id)
                    all_signals.append(sig)
        
        for rule in self.correlation_rules:
            result = rule.evaluate(all_signals)
            if result:
                confidence, description = result
                self._create_or_update_threat(rule, all_signals, confidence, description)
    
    def _create_or_update_threat(
        self, 
        rule: CorrelationRule, 
        signals: List[WeakSignal],
        confidence: float,
        description: str
    ):
        """Create new threat or update existing one"""
        
        # Check for existing active threat of same type
        existing = None
        for threat in self.active_threats.values():
            if threat.threat_type == rule.threat_type and threat.is_active:
                # Check if entities overlap
                threat_entities = set(threat.entity_ids)
                signal_entities = {s.entity_id for s in signals}
                if threat_entities & signal_entities:
                    existing = threat
                    break
        
        if existing:
            # Update existing threat
            existing.evidence.extend(signals)
            existing.confidence = max(existing.confidence, confidence)
            if confidence > existing.confidence:
                existing.description = description
        else:
            # Create new threat
            threat_id = f"THR-{uuid.uuid4().hex[:8].upper()}"
            
            threat = ThreatConclusion(
                threat_id=threat_id,
                threat_type=rule.threat_type,
                severity=self._calculate_severity(confidence, rule.threat_type),
                title=f"{rule.threat_type.value.replace('_', ' ').title()} Detected",
                description=description,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                entity_ids=list({s.entity_id for s in signals}),
                zone_ids=list({s.zone_id for s in signals}),
                evidence=signals.copy(),
                timestamp=datetime.now()
            )
            
            self.active_threats[threat_id] = threat
            
            # Publish threat event
            event = create_threat_event(
                threat_id=threat_id,
                threat_type=rule.threat_type.value,
                severity=threat.severity.value,
                confidence=confidence,
                entity_ids=threat.entity_ids,
                description=description,
                evidence=[s.payload for s in signals[:5]]
            )
            event_bus.publish(event)
    
    def _calculate_severity(self, confidence: float, threat_type: ThreatType) -> ThreatSeverity:
        """Calculate threat severity based on confidence and type"""
        
        # Critical threat types
        critical_types = {ThreatType.PRE_ATTACK_INDICATOR, ThreatType.COORDINATED_MOVEMENT}
        high_types = {ThreatType.HOSTILE_RECONNAISSANCE, ThreatType.EVASIVE_BEHAVIOR, ThreatType.SECURITY_PROBING}
        
        if threat_type in critical_types and confidence > 0.7:
            return ThreatSeverity.CRITICAL
        elif threat_type in critical_types:
            return ThreatSeverity.HIGH
        elif threat_type in high_types and confidence > 0.7:
            return ThreatSeverity.HIGH
        elif confidence > 0.7:
            return ThreatSeverity.MEDIUM
        elif confidence > 0.5:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.INFO
    
    def escalate_threat(self, threat_id: str, reason: str = "") -> bool:
        """Escalate threat severity"""
        with self._lock:
            if threat_id not in self.active_threats:
                return False
            
            threat = self.active_threats[threat_id]
            severity_order = [ThreatSeverity.INFO, ThreatSeverity.LOW, ThreatSeverity.MEDIUM, ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]
            current_idx = severity_order.index(threat.severity)
            
            if current_idx < len(severity_order) - 1:
                threat.severity = severity_order[current_idx + 1]
                threat.escalation_count += 1
                
                event = SecurityEvent(
                    event_type=EventType.THREAT_ESCALATED,
                    timestamp=datetime.now(),
                    source_layer=4,
                    source_module="threat_engine",
                    threat_id=threat_id,
                    payload={"new_severity": threat.severity.value, "reason": reason}
                )
                event_bus.publish(event)
                return True
        return False
    
    def deescalate_threat(self, threat_id: str, reason: str = "") -> bool:
        """De-escalate threat severity"""
        with self._lock:
            if threat_id not in self.active_threats:
                return False
            
            threat = self.active_threats[threat_id]
            severity_order = [ThreatSeverity.INFO, ThreatSeverity.LOW, ThreatSeverity.MEDIUM, ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]
            current_idx = severity_order.index(threat.severity)
            
            if current_idx > 0:
                threat.severity = severity_order[current_idx - 1]
                
                event = SecurityEvent(
                    event_type=EventType.THREAT_DEESCALATED,
                    timestamp=datetime.now(),
                    source_layer=4,
                    source_module="threat_engine",
                    threat_id=threat_id,
                    payload={"new_severity": threat.severity.value, "reason": reason}
                )
                event_bus.publish(event)
                return True
        return False
    
    def resolve_threat(self, threat_id: str, notes: str = "") -> bool:
        """Resolve/close a threat"""
        with self._lock:
            if threat_id not in self.active_threats:
                return False
            
            threat = self.active_threats[threat_id]
            threat.is_active = False
            threat.resolution_notes = notes
            
            self.resolved_threats.append(threat)
            del self.active_threats[threat_id]
            
            event = SecurityEvent(
                event_type=EventType.INCIDENT_RESOLVED,
                timestamp=datetime.now(),
                source_layer=4,
                source_module="threat_engine",
                threat_id=threat_id,
                payload={"notes": notes}
            )
            event_bus.publish(event)
            return True
    
    def get_active_threats(self) -> List[ThreatConclusion]:
        """Get all active threats"""
        with self._lock:
            return list(self.active_threats.values())
    
    def get_threat(self, threat_id: str) -> Optional[ThreatConclusion]:
        """Get specific threat by ID"""
        with self._lock:
            return self.active_threats.get(threat_id)
    
    def get_threats_by_severity(self, severity: ThreatSeverity) -> List[ThreatConclusion]:
        """Get threats filtered by severity"""
        with self._lock:
            return [t for t in self.active_threats.values() if t.severity == severity]
    
    def _on_detection(self, event: SecurityEvent):
        """Handle detection events from Layer-1"""
        signal = WeakSignal(
            signal_id=f"SIG-{uuid.uuid4().hex[:8]}",
            signal_type=event.payload.get("detection_type", "object"),
            entity_id=event.entity_id or "unknown",
            zone_id=event.zone_id or "default",
            timestamp=event.timestamp,
            confidence=event.confidence,
            weight=1.0,
            payload=event.payload
        )
        self.add_signal(signal)
    
    def _on_behavior(self, event: SecurityEvent):
        """Handle behavior events from Layer-2"""
        signal = WeakSignal(
            signal_id=f"SIG-{uuid.uuid4().hex[:8]}",
            signal_type=event.payload.get("behavior_type", "unknown"),
            entity_id=event.entity_id or "unknown",
            zone_id=event.zone_id or "default",
            timestamp=event.timestamp,
            confidence=event.confidence,
            weight=1.5,  # Higher weight for behavior signals
            payload=event.payload
        )
        self.add_signal(signal)
    
    def _on_anomaly(self, event: SecurityEvent):
        """Handle anomaly events"""
        anomaly_score = event.payload.get("anomaly_score", 0)
        if anomaly_score > 0.3:
            signal = WeakSignal(
                signal_id=f"SIG-{uuid.uuid4().hex[:8]}",
                signal_type="anomaly",
                entity_id=event.entity_id or "unknown",
                zone_id=event.zone_id or "default",
                timestamp=event.timestamp,
                confidence=anomaly_score,
                weight=2.0,  # Higher weight for anomalies
                payload=event.payload
            )
            self.add_signal(signal)
    
    def enable(self):
        """Enable the engine"""
        self._enabled = True
    
    def disable(self):
        """Disable the engine (graceful degradation)"""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global instance
threat_engine = ThreatIntelligenceEngine()
