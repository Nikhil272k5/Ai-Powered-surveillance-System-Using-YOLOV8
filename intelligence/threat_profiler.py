"""
Threat Profiler - Persistent Entity Threat Profiles

Maintains behavioral history and risk evolution for tracked entities.
"""
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from collections import defaultdict
import threading
import json

from .event_bus import EventBus, SecurityEvent, EventType, event_bus


class RiskLevel(Enum):
    """Entity risk levels"""
    UNKNOWN = "unknown"
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehaviorRecord:
    """Single behavior observation"""
    behavior_type: str
    timestamp: datetime
    zone_id: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ThreatProfile:
    """Persistent threat profile for an entity"""
    profile_id: str
    entity_id: str
    first_seen: datetime
    last_seen: datetime
    total_sightings: int
    risk_level: RiskLevel
    risk_score: float
    behavior_history: List[BehaviorRecord]
    zones_visited: Dict[str, int]  # zone_id -> visit count
    associated_threats: List[str]  # threat_ids
    tags: List[str]
    notes: str
    is_watchlist: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "entity_id": self.entity_id,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "total_sightings": self.total_sightings,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "behavior_count": len(self.behavior_history),
            "zones_visited": list(self.zones_visited.keys()),
            "threat_count": len(self.associated_threats),
            "tags": self.tags,
            "is_watchlist": self.is_watchlist
        }
    
    def get_risk_trajectory(self) -> str:
        """Determine if risk is increasing, stable, or decreasing"""
        if len(self.behavior_history) < 3:
            return "insufficient_data"
        
        recent = self.behavior_history[-5:]
        suspicious_count = sum(1 for b in recent if b.confidence > 0.5)
        
        if suspicious_count >= 4:
            return "increasing"
        elif suspicious_count <= 1:
            return "decreasing"
        else:
            return "stable"


class ThreatProfiler:
    """
    Maintains persistent threat profiles for entities
    """
    
    def __init__(self):
        self.profiles: Dict[str, ThreatProfile] = {}
        self.entity_to_profile: Dict[str, str] = {}  # entity_id -> profile_id
        self._lock = threading.Lock()
        self._enabled = True
        
        # Subscribe to events
        event_bus.subscribe(EventType.DETECTION, self._on_detection)
        event_bus.subscribe(EventType.BEHAVIOR_DETECTED, self._on_behavior)
        event_bus.subscribe(EventType.THREAT_CORRELATED, self._on_threat)
    
    def get_or_create_profile(self, entity_id: str) -> ThreatProfile:
        """Get existing profile or create new one"""
        with self._lock:
            if entity_id in self.entity_to_profile:
                return self.profiles[self.entity_to_profile[entity_id]]
            
            profile_id = f"PROF-{uuid.uuid4().hex[:8].upper()}"
            profile = ThreatProfile(
                profile_id=profile_id,
                entity_id=entity_id,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                total_sightings=1,
                risk_level=RiskLevel.UNKNOWN,
                risk_score=0.0,
                behavior_history=[],
                zones_visited={},
                associated_threats=[],
                tags=[],
                notes="",
                is_watchlist=False
            )
            
            self.profiles[profile_id] = profile
            self.entity_to_profile[entity_id] = profile_id
            
            return profile
    
    def update_sighting(self, entity_id: str, zone_id: str):
        """Update profile with new sighting"""
        profile = self.get_or_create_profile(entity_id)
        
        with self._lock:
            profile.last_seen = datetime.now()
            profile.total_sightings += 1
            profile.zones_visited[zone_id] = profile.zones_visited.get(zone_id, 0) + 1
    
    def add_behavior(self, entity_id: str, behavior: BehaviorRecord):
        """Add behavior observation to profile"""
        profile = self.get_or_create_profile(entity_id)
        
        with self._lock:
            profile.behavior_history.append(behavior)
            
            # Keep last 100 behaviors
            if len(profile.behavior_history) > 100:
                profile.behavior_history = profile.behavior_history[-100:]
            
            # Recalculate risk
            self._recalculate_risk(profile)
        
        # Publish event
        event = SecurityEvent(
            event_type=EventType.PROFILE_UPDATED,
            timestamp=datetime.now(),
            source_layer=2,
            source_module="threat_profiler",
            entity_id=entity_id,
            payload={
                "profile_id": profile.profile_id,
                "behavior_type": behavior.behavior_type,
                "new_risk_score": profile.risk_score
            }
        )
        event_bus.publish(event)
    
    def _recalculate_risk(self, profile: ThreatProfile):
        """Recalculate risk score and level"""
        if not profile.behavior_history:
            profile.risk_score = 0.0
            profile.risk_level = RiskLevel.UNKNOWN
            return
        
        # Recent behavior weighting
        now = datetime.now()
        weights = []
        
        for behavior in profile.behavior_history:
            age_hours = (now - behavior.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            weights.append(behavior.confidence * recency_weight)
        
        base_score = sum(weights) / len(weights) if weights else 0
        
        # Boost for repeated presence
        if profile.total_sightings > 10:
            base_score *= 1.2
        
        # Boost for multiple zones
        if len(profile.zones_visited) > 3:
            base_score *= 1.1
        
        # Boost for associated threats
        if profile.associated_threats:
            base_score *= (1 + 0.1 * len(profile.associated_threats))
        
        # Watchlist boost
        if profile.is_watchlist:
            base_score *= 1.5
        
        profile.risk_score = min(1.0, base_score)
        
        # Determine level
        if profile.risk_score > 0.8:
            profile.risk_level = RiskLevel.CRITICAL
        elif profile.risk_score > 0.6:
            profile.risk_level = RiskLevel.HIGH
        elif profile.risk_score > 0.4:
            profile.risk_level = RiskLevel.ELEVATED
        elif profile.risk_score > 0.2:
            profile.risk_level = RiskLevel.LOW
        else:
            profile.risk_level = RiskLevel.UNKNOWN
    
    def add_to_watchlist(self, entity_id: str, tags: List[str] = None) -> bool:
        """Add entity to watchlist"""
        profile = self.get_or_create_profile(entity_id)
        
        with self._lock:
            profile.is_watchlist = True
            if tags:
                profile.tags.extend(tags)
            self._recalculate_risk(profile)
        
        return True
    
    def remove_from_watchlist(self, entity_id: str) -> bool:
        """Remove entity from watchlist"""
        if entity_id not in self.entity_to_profile:
            return False
        
        profile = self.profiles[self.entity_to_profile[entity_id]]
        
        with self._lock:
            profile.is_watchlist = False
            self._recalculate_risk(profile)
        
        return True
    
    def get_profile(self, entity_id: str) -> Optional[ThreatProfile]:
        """Get profile by entity ID"""
        with self._lock:
            if entity_id in self.entity_to_profile:
                return self.profiles[self.entity_to_profile[entity_id]]
        return None
    
    def get_high_risk_profiles(self) -> List[ThreatProfile]:
        """Get all elevated+ risk profiles"""
        with self._lock:
            return [
                p for p in self.profiles.values()
                if p.risk_level in [RiskLevel.ELEVATED, RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
    
    def get_watchlist(self) -> List[ThreatProfile]:
        """Get all watchlist profiles"""
        with self._lock:
            return [p for p in self.profiles.values() if p.is_watchlist]
    
    def _on_detection(self, event: SecurityEvent):
        """Handle detection events"""
        if not self._enabled:
            return
        
        entity_id = event.entity_id
        zone_id = event.zone_id or "default"
        
        if entity_id:
            self.update_sighting(entity_id, zone_id)
    
    def _on_behavior(self, event: SecurityEvent):
        """Handle behavior events"""
        if not self._enabled:
            return
        
        entity_id = event.entity_id
        if not entity_id:
            return
        
        behavior = BehaviorRecord(
            behavior_type=event.payload.get("behavior_type", "unknown"),
            timestamp=event.timestamp,
            zone_id=event.zone_id or "default",
            confidence=event.confidence,
            details=event.payload
        )
        
        self.add_behavior(entity_id, behavior)
    
    def _on_threat(self, event: SecurityEvent):
        """Handle threat events"""
        if not self._enabled:
            return
        
        threat_id = event.threat_id
        entity_ids = event.payload.get("entity_ids", [])
        
        for entity_id in entity_ids:
            profile = self.get_or_create_profile(entity_id)
            with self._lock:
                if threat_id not in profile.associated_threats:
                    profile.associated_threats.append(threat_id)
                self._recalculate_risk(profile)
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False


# Global instance
threat_profiler = ThreatProfiler()
