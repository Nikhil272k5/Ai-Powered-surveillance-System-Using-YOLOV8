"""
Event Bus - Central Communication System for 5-Layer Security Architecture

All modules communicate via events, never modifying Layer-1 directly.
This enables loose coupling and optional feature toggling.
"""
import asyncio
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import json
import threading


class EventType(Enum):
    """Event types for inter-layer communication"""
    # Layer 1: Sense (from existing detection)
    DETECTION = "detection"
    TRACK_UPDATE = "track_update"
    ALERT_BASIC = "alert_basic"
    
    # Layer 2: Understand
    BEHAVIOR_DETECTED = "behavior_detected"
    PATTERN_RECOGNIZED = "pattern_recognized"
    ANOMALY_SCORED = "anomaly_scored"
    PROFILE_UPDATED = "profile_updated"
    
    # Layer 3: Predict
    THREAT_PREDICTED = "threat_predicted"
    RISK_FORECAST = "risk_forecast"
    PRE_ATTACK_INDICATOR = "pre_attack_indicator"
    
    # Layer 4: Respond
    THREAT_CORRELATED = "threat_correlated"
    THREAT_ESCALATED = "threat_escalated"
    THREAT_DEESCALATED = "threat_deescalated"
    PLAYBOOK_TRIGGERED = "playbook_triggered"
    RESPONSE_EXECUTED = "response_executed"
    
    # Layer 5: Audit
    INCIDENT_CREATED = "incident_created"
    INCIDENT_RESOLVED = "incident_resolved"
    AUDIT_LOG = "audit_log"
    EVIDENCE_CAPTURED = "evidence_captured"


@dataclass
class SecurityEvent:
    """Base event structure for all security events"""
    event_type: EventType
    timestamp: datetime
    source_layer: int  # 1-5
    source_module: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    threat_id: Optional[str] = None
    entity_id: Optional[str] = None
    zone_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source_layer": self.source_layer,
            "source_module": self.source_module,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "threat_id": self.threat_id,
            "entity_id": self.entity_id,
            "zone_id": self.zone_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class EventBus:
    """
    Central event bus for security platform communication.
    Enables loose coupling between layers.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[SecurityEvent] = []
        self._history_limit = 10000
        self._enabled = True
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: EventType, handler: Callable[[SecurityEvent], None]):
        """Subscribe to events of a specific type"""
        with self._lock:
            self._subscribers[event_type].append(handler)
    
    def subscribe_async(self, event_type: EventType, handler: Callable[[SecurityEvent], Any]):
        """Subscribe async handler to events"""
        with self._lock:
            self._async_subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from events"""
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
            if handler in self._async_subscribers[event_type]:
                self._async_subscribers[event_type].remove(handler)
    
    def publish(self, event: SecurityEvent):
        """Publish event to all subscribers"""
        if not self._enabled:
            return
        
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._history_limit:
                self._event_history = self._event_history[-self._history_limit:]
        
        # Notify sync subscribers
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] Handler error: {e}")
        
        # Notify wildcard subscribers (subscribe to all)
        for handler in self._subscribers.get(None, []):
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] Wildcard handler error: {e}")
    
    async def publish_async(self, event: SecurityEvent):
        """Async publish for async handlers"""
        if not self._enabled:
            return
        
        self.publish(event)  # Also trigger sync handlers
        
        # Notify async subscribers
        for handler in self._async_subscribers.get(event.event_type, []):
            try:
                await handler(event)
            except Exception as e:
                print(f"[EventBus] Async handler error: {e}")
    
    def get_recent_events(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[SecurityEvent]:
        """Get recent events, optionally filtered by type"""
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e.event_type == event_type]
            else:
                events = self._event_history.copy()
            return events[-limit:]
    
    def get_events_by_threat(self, threat_id: str) -> List[SecurityEvent]:
        """Get all events related to a specific threat"""
        with self._lock:
            return [e for e in self._event_history if e.threat_id == threat_id]
    
    def get_events_by_entity(self, entity_id: str) -> List[SecurityEvent]:
        """Get all events related to a specific entity"""
        with self._lock:
            return [e for e in self._event_history if e.entity_id == entity_id]
    
    def enable(self):
        """Enable event bus"""
        self._enabled = True
    
    def disable(self):
        """Disable event bus (graceful degradation)"""
        self._enabled = False
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global event bus instance
event_bus = EventBus()


def create_detection_event(
    entity_id: str,
    detection_type: str,
    confidence: float,
    zone_id: str,
    bbox: tuple,
    metadata: Dict[str, Any] = None
) -> SecurityEvent:
    """Helper to create Layer-1 detection events"""
    return SecurityEvent(
        event_type=EventType.DETECTION,
        timestamp=datetime.now(),
        source_layer=1,
        source_module="yolo_detector",
        entity_id=entity_id,
        zone_id=zone_id,
        confidence=confidence,
        payload={
            "detection_type": detection_type,
            "bbox": bbox,
            "confidence": confidence
        },
        metadata=metadata or {}
    )


def create_threat_event(
    threat_id: str,
    threat_type: str,
    severity: str,
    confidence: float,
    entity_ids: List[str],
    description: str,
    evidence: List[Dict]
) -> SecurityEvent:
    """Helper to create Layer-4 threat events"""
    return SecurityEvent(
        event_type=EventType.THREAT_CORRELATED,
        timestamp=datetime.now(),
        source_layer=4,
        source_module="threat_engine",
        threat_id=threat_id,
        confidence=confidence,
        payload={
            "threat_type": threat_type,
            "severity": severity,
            "entity_ids": entity_ids,
            "description": description,
            "evidence": evidence
        }
    )
