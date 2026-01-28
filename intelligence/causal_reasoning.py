"""
Causal Reasoning & Event Chain Engine
Builds event graphs and explains why incidents occurred.

Features:
- Event graph with temporal ordering
- Cause-effect relationship inference
- Natural language explanation generation
- Rule-based causal analysis

Example chains:
- crowd_surge → exit_blockage → panic_movement
- object_obstruction → crowd_diversion → counterflow
"""

import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class EventType(Enum):
    """Types of events that can be causally linked"""
    # Crowd events
    CROWD_SURGE = "crowd_surge"
    CROWD_DISPERSION = "crowd_dispersion"
    CROWD_GATHERING = "crowd_gathering"
    CROWD_DIVERSION = "crowd_diversion"
    
    # Movement events
    PANIC_MOVEMENT = "panic_movement"
    COUNTERFLOW = "counterflow"
    MASS_DIRECTION_CHANGE = "mass_direction_change"
    SPEED_SPIKE = "speed_spike"
    
    # Object events
    OBJECT_OBSTRUCTION = "object_obstruction"
    ABANDONED_OBJECT = "abandoned_object"
    OBJECT_REMOVED = "object_removed"
    
    # Location events
    EXIT_BLOCKAGE = "exit_blockage"
    ENTRY_CONGESTION = "entry_congestion"
    ZONE_INTRUSION = "zone_intrusion"
    
    # Behavior events
    LOITERING_DETECTED = "loitering_detected"
    EVASIVE_BEHAVIOR = "evasive_behavior"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Generic
    ANOMALY_DETECTED = "anomaly_detected"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class Event:
    """Single event in the causal graph"""
    event_id: str
    event_type: EventType
    timestamp: float
    position: Optional[Tuple[float, float]] = None
    track_ids: List[int] = field(default_factory=list)
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.event_id)
    
    def __eq__(self, other):
        if isinstance(other, Event):
            return self.event_id == other.event_id
        return False


@dataclass
class CausalLink:
    """Link between cause and effect events"""
    cause_id: str
    effect_id: str
    confidence: float  # 0-1
    rule_name: str
    explanation: str


@dataclass
class CausalChain:
    """Complete causal chain explaining an incident"""
    chain_id: str
    events: List[Event]
    links: List[CausalLink]
    root_cause: Event
    final_effect: Event
    explanation: str
    confidence: float
    timestamp: float


# Causal rules: (cause_type, effect_type, time_window, spatial_proximity, confidence)
CAUSAL_RULES = [
    # Crowd dynamics
    (EventType.CROWD_SURGE, EventType.EXIT_BLOCKAGE, 10, 200, 0.8),
    (EventType.CROWD_SURGE, EventType.PANIC_MOVEMENT, 15, 300, 0.7),
    (EventType.EXIT_BLOCKAGE, EventType.PANIC_MOVEMENT, 20, 200, 0.85),
    (EventType.EXIT_BLOCKAGE, EventType.COUNTERFLOW, 15, 200, 0.75),
    (EventType.CROWD_GATHERING, EventType.CROWD_SURGE, 30, 150, 0.6),
    
    # Object-related
    (EventType.OBJECT_OBSTRUCTION, EventType.COUNTERFLOW, 10, 100, 0.7),
    (EventType.OBJECT_OBSTRUCTION, EventType.CROWD_DIVERSION, 10, 150, 0.75),
    (EventType.ABANDONED_OBJECT, EventType.CROWD_DISPERSION, 20, 200, 0.5),
    
    # Movement chains
    (EventType.SPEED_SPIKE, EventType.PANIC_MOVEMENT, 5, 100, 0.8),
    (EventType.MASS_DIRECTION_CHANGE, EventType.COUNTERFLOW, 10, 200, 0.7),
    (EventType.EVASIVE_BEHAVIOR, EventType.SUSPICIOUS_ACTIVITY, 15, 100, 0.6),
    
    # Alert chains
    (EventType.LOITERING_DETECTED, EventType.SUSPICIOUS_ACTIVITY, 30, 50, 0.5),
    (EventType.ZONE_INTRUSION, EventType.ALERT_TRIGGERED, 5, 300, 0.9),
]

# Explanation templates
EXPLANATION_TEMPLATES = {
    # Cause → Effect explanations
    (EventType.CROWD_SURGE, EventType.EXIT_BLOCKAGE):
        "Crowd surge at {cause_loc} blocked exit at {effect_loc}",
    (EventType.CROWD_SURGE, EventType.PANIC_MOVEMENT):
        "Crowd surge caused panic movement among nearby individuals",
    (EventType.EXIT_BLOCKAGE, EventType.PANIC_MOVEMENT):
        "Exit blockage at {cause_loc} triggered panic movement",
    (EventType.EXIT_BLOCKAGE, EventType.COUNTERFLOW):
        "Exit blockage caused people to move against normal flow direction",
    (EventType.OBJECT_OBSTRUCTION, EventType.COUNTERFLOW):
        "Object obstruction at {cause_loc} diverted pedestrian flow",
    (EventType.SPEED_SPIKE, EventType.PANIC_MOVEMENT):
        "Sudden speed increase indicates possible panic or emergency response",
    (EventType.EVASIVE_BEHAVIOR, EventType.SUSPICIOUS_ACTIVITY):
        "Evasive movement pattern indicates potential suspicious activity",
}


class CausalReasoningEngine:
    """
    Causal Reasoning & Event Chain Engine
    
    Builds event graphs, infers cause-effect relationships,
    and generates natural language explanations for incidents.
    """
    
    def __init__(self,
                 lookback_seconds: int = 30,
                 spatial_proximity: int = 200,
                 min_chain_length: int = 2,
                 max_events: int = 5000):
        """
        Initialize the Causal Reasoning Engine.
        
        Args:
            lookback_seconds: Time window for causal analysis
            spatial_proximity: Maximum distance for spatial proximity
            min_chain_length: Minimum events to form a chain
            max_events: Maximum events to keep in memory
        """
        self.lookback_seconds = lookback_seconds
        self.spatial_proximity = spatial_proximity
        self.min_chain_length = min_chain_length
        
        # Event storage
        self.events: deque = deque(maxlen=max_events)
        self.event_index: Dict[str, Event] = {}
        
        # Graph structure
        self.causal_links: List[CausalLink] = []
        self.effect_to_causes: Dict[str, List[str]] = defaultdict(list)
        self.cause_to_effects: Dict[str, List[str]] = defaultdict(list)
        
        # Causal chains
        self.chains: List[CausalChain] = []
        
        # Statistics
        self.total_events = 0
        self.total_chains = 0
        
        print(f"🔗 Causal Reasoning Engine initialized")
        print(f"   Lookback window: {lookback_seconds}s")
        print(f"   Spatial proximity: {spatial_proximity}px")
        print(f"   Loaded {len(CAUSAL_RULES)} causal rules")
    
    def add_event(self, 
                  event_type: EventType,
                  position: Optional[Tuple[float, float]] = None,
                  track_ids: Optional[List[int]] = None,
                  severity: str = "medium",
                  metadata: Optional[Dict] = None) -> Event:
        """
        Add a new event to the causal graph.
        
        Args:
            event_type: Type of event
            position: (x, y) position in frame
            track_ids: Related track IDs
            severity: Event severity level
            metadata: Additional event data
        
        Returns:
            The created Event object
        """
        event = Event(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            timestamp=time.time(),
            position=position,
            track_ids=track_ids or [],
            severity=severity,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.event_index[event.event_id] = event
        self.total_events += 1
        
        # Check for causal links to this new event
        self._find_causes_for_event(event)
        
        return event
    
    def add_event_from_alert(self, alert: Dict) -> Event:
        """
        Add an event based on an alert dictionary.
        
        Args:
            alert: Alert dictionary from detection logic
        
        Returns:
            The created Event object
        """
        # Map alert types to event types
        alert_type = alert.get('type', 'unknown')
        
        type_mapping = {
            'speed_spike': EventType.SPEED_SPIKE,
            'loitering': EventType.LOITERING_DETECTED,
            'counterflow': EventType.COUNTERFLOW,
            'abandoned_object': EventType.ABANDONED_OBJECT,
            'panic': EventType.PANIC_MOVEMENT,
            'evasive': EventType.EVASIVE_BEHAVIOR,
        }
        
        event_type = type_mapping.get(alert_type, EventType.ANOMALY_DETECTED)
        
        return self.add_event(
            event_type=event_type,
            position=alert.get('position'),
            track_ids=[alert.get('track_id')] if alert.get('track_id') else [],
            severity=alert.get('severity', 'medium'),
            metadata={'original_alert': alert}
        )
    
    def _find_causes_for_event(self, effect: Event) -> None:
        """Find potential causes for a new event"""
        current_time = effect.timestamp
        
        for event in self.events:
            if event.event_id == effect.event_id:
                continue
            
            # Check time window
            time_diff = current_time - event.timestamp
            if time_diff < 0 or time_diff > self.lookback_seconds:
                continue
            
            # Check each causal rule
            for rule in CAUSAL_RULES:
                cause_type, effect_type, time_window, spatial_prox, confidence = rule
                
                if event.event_type != cause_type:
                    continue
                if effect.event_type != effect_type:
                    continue
                
                # Check time window
                if time_diff > time_window:
                    continue
                
                # Check spatial proximity
                if event.position and effect.position:
                    distance = self._calculate_distance(event.position, effect.position)
                    if distance > spatial_prox:
                        continue
                    
                    # Adjust confidence based on distance
                    distance_factor = 1.0 - (distance / spatial_prox) * 0.3
                    confidence = confidence * distance_factor
                
                # Create causal link
                explanation = self._generate_link_explanation(event, effect)
                
                link = CausalLink(
                    cause_id=event.event_id,
                    effect_id=effect.event_id,
                    confidence=confidence,
                    rule_name=f"{cause_type.value}_causes_{effect_type.value}",
                    explanation=explanation
                )
                
                self.causal_links.append(link)
                self.effect_to_causes[effect.event_id].append(event.event_id)
                self.cause_to_effects[event.event_id].append(effect.event_id)
    
    def _generate_link_explanation(self, cause: Event, effect: Event) -> str:
        """Generate explanation for a causal link"""
        template_key = (cause.event_type, effect.event_type)
        
        if template_key in EXPLANATION_TEMPLATES:
            template = EXPLANATION_TEMPLATES[template_key]
            
            # Fill in location information
            cause_loc = self._format_location(cause.position)
            effect_loc = self._format_location(effect.position)
            
            return template.format(
                cause_loc=cause_loc,
                effect_loc=effect_loc
            )
        
        # Default explanation
        return (
            f"{cause.event_type.value.replace('_', ' ').title()} "
            f"may have caused {effect.event_type.value.replace('_', ' ')}"
        )
    
    def _format_location(self, position: Optional[Tuple[float, float]]) -> str:
        """Format position as human-readable location"""
        if position is None:
            return "unknown location"
        
        x, y = position
        
        # Convert to general location description
        if x < 200:
            h_loc = "left side"
        elif x > 800:
            h_loc = "right side"
        else:
            h_loc = "center"
        
        if y < 200:
            v_loc = "top"
        elif y > 500:
            v_loc = "bottom"
        else:
            v_loc = "middle"
        
        return f"{v_loc}-{h_loc} ({x:.0f}, {y:.0f})"
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def analyze_event(self, event: Event) -> Optional[CausalChain]:
        """
        Analyze an event and build its causal chain.
        
        Args:
            event: The event to analyze
        
        Returns:
            CausalChain if causes found, None otherwise
        """
        # Find all causes (backward traversal)
        cause_chain = self._trace_causes(event.event_id, set())
        
        if len(cause_chain) < self.min_chain_length:
            return None
        
        # Build causal chain
        chain_events = [self.event_index[eid] for eid in cause_chain if eid in self.event_index]
        chain_events.sort(key=lambda e: e.timestamp)
        
        # Get links in the chain
        chain_links = []
        for i in range(len(cause_chain) - 1):
            for link in self.causal_links:
                if link.cause_id in cause_chain and link.effect_id in cause_chain:
                    chain_links.append(link)
        
        # Remove duplicates
        seen = set()
        unique_links = []
        for link in chain_links:
            key = (link.cause_id, link.effect_id)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        
        # Calculate chain confidence
        if unique_links:
            chain_confidence = np.mean([link.confidence for link in unique_links])
        else:
            chain_confidence = 0.5
        
        # Generate full explanation
        explanation = self._generate_chain_explanation(chain_events, unique_links)
        
        chain = CausalChain(
            chain_id=str(uuid.uuid4())[:8],
            events=chain_events,
            links=unique_links,
            root_cause=chain_events[0] if chain_events else event,
            final_effect=event,
            explanation=explanation,
            confidence=chain_confidence,
            timestamp=time.time()
        )
        
        self.chains.append(chain)
        self.total_chains += 1
        
        return chain
    
    def _trace_causes(self, event_id: str, visited: Set[str]) -> List[str]:
        """Recursively trace causes of an event"""
        if event_id in visited:
            return []
        
        visited.add(event_id)
        chain = [event_id]
        
        # Get causes
        causes = self.effect_to_causes.get(event_id, [])
        
        for cause_id in causes:
            if cause_id not in visited:
                cause_chain = self._trace_causes(cause_id, visited)
                chain = cause_chain + chain
        
        return chain
    
    def _generate_chain_explanation(self, events: List[Event], 
                                    links: List[CausalLink]) -> str:
        """Generate comprehensive explanation for a causal chain"""
        if not events:
            return "Unable to determine causal chain"
        
        if len(events) == 1:
            return f"Single event: {events[0].event_type.value.replace('_', ' ')}"
        
        # Build narrative explanation
        parts = []
        
        # Root cause
        root = events[0]
        root_time = time.strftime('%H:%M:%S', time.localtime(root.timestamp))
        parts.append(
            f"Chain started with {root.event_type.value.replace('_', ' ')} "
            f"at {root_time}"
        )
        
        # Intermediate events
        for link in links:
            cause = self.event_index.get(link.cause_id)
            effect = self.event_index.get(link.effect_id)
            
            if cause and effect:
                time_diff = effect.timestamp - cause.timestamp
                parts.append(
                    f"After {time_diff:.1f}s, this led to "
                    f"{effect.event_type.value.replace('_', ' ')}"
                )
        
        # Final effect
        final = events[-1]
        final_time = time.strftime('%H:%M:%S', time.localtime(final.timestamp))
        parts.append(
            f"Resulting in {final.event_type.value.replace('_', ' ')} "
            f"at {final_time}"
        )
        
        return ". ".join(parts) + "."
    
    def get_explanation_for_alert(self, alert: Dict) -> str:
        """
        Get causal explanation for an alert.
        
        Args:
            alert: Alert dictionary
        
        Returns:
            Human-readable causal explanation
        """
        # Add event and analyze
        event = self.add_event_from_alert(alert)
        chain = self.analyze_event(event)
        
        if chain:
            return chain.explanation
        else:
            return f"Direct detection of {alert.get('type', 'anomaly').replace('_', ' ')}"
    
    def get_recent_chains(self, count: int = 10) -> List[CausalChain]:
        """Get the most recent causal chains"""
        return sorted(self.chains, key=lambda c: c.timestamp, reverse=True)[:count]
    
    def get_events_in_window(self, seconds: int = 60) -> List[Event]:
        """Get events within a time window"""
        current_time = time.time()
        return [
            event for event in self.events
            if current_time - event.timestamp <= seconds
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        event_type_counts = defaultdict(int)
        for event in self.events:
            event_type_counts[event.event_type.value] += 1
        
        return {
            'total_events': self.total_events,
            'events_in_memory': len(self.events),
            'total_links': len(self.causal_links),
            'total_chains': self.total_chains,
            'event_type_counts': dict(event_type_counts),
            'lookback_seconds': self.lookback_seconds,
            'rules_loaded': len(CAUSAL_RULES)
        }
    
    def clear_old_events(self, max_age_seconds: int = 300) -> int:
        """Clear events older than max_age"""
        current_time = time.time()
        old_events = [
            e for e in self.events
            if current_time - e.timestamp > max_age_seconds
        ]
        
        for event in old_events:
            self.events.remove(event)
            self.event_index.pop(event.event_id, None)
        
        # Clean up related links
        valid_ids = {e.event_id for e in self.events}
        self.causal_links = [
            link for link in self.causal_links
            if link.cause_id in valid_ids and link.effect_id in valid_ids
        ]
        
        return len(old_events)
