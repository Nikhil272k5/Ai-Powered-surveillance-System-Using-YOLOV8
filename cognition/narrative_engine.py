"""
AbnoGuard Cognition - Narrative Engine
Generates human-readable stories about incidents, not just alerts
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class NarrativeStyle(Enum):
    TECHNICAL = "technical"      # For logs and systems
    SECURITY = "security"        # For security personnel
    EXECUTIVE = "executive"      # Brief summary for management
    DETAILED = "detailed"        # Full investigative report


@dataclass
class NarrativeEvent:
    """A single event in a narrative"""
    timestamp: float
    action: str           # "entered", "placed", "departed", etc.
    subject: str          # "A person", "An individual", etc.
    object_desc: str      # "a backpack", "near the exit", etc.
    location: str         # Zone or area description
    significance: str     # Why this matters
    

@dataclass
class Incident:
    """Represents a complete incident for narration"""
    incident_id: str
    start_time: float
    end_time: Optional[float]
    track_ids: List[int]
    events: List[NarrativeEvent]
    threat_level: str
    primary_type: str     # "abandoned_object", "loitering", etc.
    resolved: bool
    resolution: Optional[str]
    

class NarrativeEngine:
    """
    Generates human-readable narratives describing incidents.
    Transforms detections and events into comprehensible stories.
    
    Example output:
    "At 14:32, an individual entered the waiting area carrying a backpack.
    After 45 seconds, they placed the bag near Exit B and departed the vicinity.
    The object remained unattended for 74 seconds during a period of low traffic,
    representing a significant deviation from normal behavior patterns."
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Active incidents being tracked
        self.active_incidents: Dict[str, Incident] = {}
        
        # Completed incidents for history
        self.completed_incidents: List[Incident] = []
        self.max_history = self.config.get('max_history', 100)
        
        # Language templates
        self.action_verbs = {
            'enter': ['entered', 'arrived at', 'approached'],
            'exit': ['exited', 'departed from', 'left'],
            'stop': ['stopped', 'paused', 'halted'],
            'place': ['placed', 'set down', 'deposited'],
            'pickup': ['retrieved', 'picked up', 'collected'],
            'loiter': ['remained stationary', 'lingered', 'stayed'],
            'run': ['ran', 'sprinted', 'moved quickly'],
            'approach': ['approached', 'moved toward', 'drew near to'],
            'retreat': ['retreated', 'moved away from', 'withdrew']
        }
        
        self.subject_descriptions = {
            'person': ['An individual', 'A person', 'Someone'],
            'group': ['A group of individuals', 'Several people', 'Multiple persons'],
            'object': ['An object', 'An item', 'A package'],
            'vehicle': ['A vehicle', 'A car', 'An automobile']
        }
        
        self.zone_descriptions = {
            'entry': 'the entry area',
            'exit': 'the exit zone',
            'waiting': 'the waiting area',
            'restricted': 'a restricted zone',
            'transit': 'the transit corridor',
            'emergency': 'the emergency area'
        }
        
        print("📖 Narrative Engine initialized")
    
    def create_incident(self, incident_type: str, track_ids: List[int],
                       initial_event: Dict) -> str:
        """Create a new incident to track"""
        incident_id = f"INC-{int(time.time())}-{len(self.active_incidents)}"
        
        first_event = NarrativeEvent(
            timestamp=time.time(),
            action=initial_event.get('action', 'detected'),
            subject=self._get_subject(initial_event),
            object_desc=initial_event.get('object_desc', ''),
            location=initial_event.get('location', 'the monitored area'),
            significance=initial_event.get('significance', 'Event detected')
        )
        
        incident = Incident(
            incident_id=incident_id,
            start_time=time.time(),
            end_time=None,
            track_ids=track_ids,
            events=[first_event],
            threat_level=initial_event.get('threat_level', 'medium'),
            primary_type=incident_type,
            resolved=False,
            resolution=None
        )
        
        self.active_incidents[incident_id] = incident
        return incident_id
    
    def add_event(self, incident_id: str, event_data: Dict) -> bool:
        """Add an event to an existing incident"""
        if incident_id not in self.active_incidents:
            return False
        
        event = NarrativeEvent(
            timestamp=time.time(),
            action=event_data.get('action', 'continued'),
            subject=event_data.get('subject', 'The subject'),
            object_desc=event_data.get('object_desc', ''),
            location=event_data.get('location', ''),
            significance=event_data.get('significance', '')
        )
        
        self.active_incidents[incident_id].events.append(event)
        return True
    
    def resolve_incident(self, incident_id: str, resolution: str) -> Optional[Incident]:
        """Resolve an incident and move to history"""
        if incident_id not in self.active_incidents:
            return None
        
        incident = self.active_incidents.pop(incident_id)
        incident.end_time = time.time()
        incident.resolved = True
        incident.resolution = resolution
        
        self.completed_incidents.append(incident)
        if len(self.completed_incidents) > self.max_history:
            self.completed_incidents.pop(0)
        
        return incident
    
    def generate_narrative(self, incident: Incident, 
                          style: NarrativeStyle = NarrativeStyle.SECURITY) -> str:
        """Generate a human-readable narrative for an incident"""
        
        if style == NarrativeStyle.EXECUTIVE:
            return self._generate_executive(incident)
        elif style == NarrativeStyle.TECHNICAL:
            return self._generate_technical(incident)
        elif style == NarrativeStyle.DETAILED:
            return self._generate_detailed(incident)
        else:
            return self._generate_security(incident)
    
    def _generate_security(self, incident: Incident) -> str:
        """Generate security-focused narrative"""
        lines = []
        
        # Opening
        start_time_str = datetime.fromtimestamp(incident.start_time).strftime('%H:%M:%S')
        lines.append(f"At {start_time_str}, an incident of type '{incident.primary_type}' was detected.")
        
        # Event sequence
        for i, event in enumerate(incident.events):
            time_str = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
            
            if i == 0:
                lines.append(f"{event.subject} {event.action} {event.location}.")
            else:
                elapsed = event.timestamp - incident.events[0].timestamp
                if elapsed < 60:
                    time_ref = f"After {elapsed:.0f} seconds"
                else:
                    time_ref = f"At {time_str}"
                
                if event.object_desc:
                    lines.append(f"{time_ref}, {event.action} {event.object_desc}.")
                else:
                    lines.append(f"{time_ref}, {event.action}.")
            
            if event.significance:
                lines.append(f"({event.significance})")
        
        # Duration
        if incident.end_time:
            duration = incident.end_time - incident.start_time
            lines.append(f"\nTotal incident duration: {duration:.0f} seconds.")
        else:
            duration = time.time() - incident.start_time
            lines.append(f"\nIncident ongoing ({duration:.0f} seconds so far).")
        
        # Resolution
        if incident.resolved and incident.resolution:
            lines.append(f"Resolution: {incident.resolution}")
        
        return " ".join(lines)
    
    def _generate_executive(self, incident: Incident) -> str:
        """Generate brief executive summary"""
        start_str = datetime.fromtimestamp(incident.start_time).strftime('%H:%M')
        duration = (incident.end_time or time.time()) - incident.start_time
        
        summary = f"[{incident.threat_level.upper()}] {incident.primary_type.replace('_', ' ').title()}"
        summary += f" at {start_str}"
        summary += f" ({duration:.0f}s)"
        
        if incident.resolved:
            summary += f" - {incident.resolution}"
        else:
            summary += " - ACTIVE"
        
        return summary
    
    def _generate_technical(self, incident: Incident) -> str:
        """Generate technical log-style narrative"""
        lines = []
        lines.append(f"INCIDENT_ID: {incident.incident_id}")
        lines.append(f"TYPE: {incident.primary_type}")
        lines.append(f"THREAT_LEVEL: {incident.threat_level}")
        lines.append(f"TRACK_IDS: {incident.track_ids}")
        lines.append(f"START: {incident.start_time}")
        lines.append(f"END: {incident.end_time}")
        lines.append(f"EVENTS:")
        
        for event in incident.events:
            lines.append(f"  [{event.timestamp}] {event.action}: {event.object_desc} @ {event.location}")
        
        if incident.resolution:
            lines.append(f"RESOLUTION: {incident.resolution}")
        
        return "\n".join(lines)
    
    def _generate_detailed(self, incident: Incident) -> str:
        """Generate detailed investigative narrative"""
        lines = []
        
        start_time = datetime.fromtimestamp(incident.start_time)
        lines.append(f"## Incident Report: {incident.incident_id}\n")
        lines.append(f"**Date:** {start_time.strftime('%Y-%m-%d')}")
        lines.append(f"**Time:** {start_time.strftime('%H:%M:%S')}")
        lines.append(f"**Type:** {incident.primary_type.replace('_', ' ').title()}")
        lines.append(f"**Threat Level:** {incident.threat_level.upper()}")
        lines.append(f"**Tracked Entities:** {len(incident.track_ids)} (IDs: {incident.track_ids})\n")
        
        lines.append("### Event Timeline\n")
        
        for i, event in enumerate(incident.events, 1):
            event_time = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
            lines.append(f"{i}. **{event_time}** - {event.subject} {event.action}")
            if event.object_desc:
                lines.append(f"   - Object: {event.object_desc}")
            if event.location:
                lines.append(f"   - Location: {event.location}")
            if event.significance:
                lines.append(f"   - *{event.significance}*")
            lines.append("")
        
        if incident.end_time:
            duration = incident.end_time - incident.start_time
            lines.append(f"### Summary")
            lines.append(f"Total duration: {duration:.0f} seconds")
            if incident.resolution:
                lines.append(f"Resolution: {incident.resolution}")
        else:
            lines.append(f"### Status: ONGOING")
        
        return "\n".join(lines)
    
    def _get_subject(self, event: Dict) -> str:
        """Get appropriate subject description"""
        obj_type = event.get('object_type', 'person')
        descriptions = self.subject_descriptions.get(obj_type, ['An entity'])
        return descriptions[0]
    
    def narrate_from_alert(self, alert: Dict) -> str:
        """Generate narrative directly from an alert dictionary"""
        timestamp = alert.get('timestamp', time.time())
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        alert_type = alert.get('type', 'anomaly').replace('_', ' ')
        description = alert.get('description', 'Anomaly detected')
        track_id = alert.get('track_id', 'unknown')
        
        narrative = f"At {time_str}, the system detected {alert_type}. "
        narrative += f"{description}. "
        narrative += f"This involved tracked entity #{track_id}."
        
        if alert.get('severity') == 'high':
            narrative += " This represents a significant security concern."
        
        return narrative
    
    def get_active_incident_summaries(self) -> List[str]:
        """Get summaries of all active incidents"""
        return [
            self.generate_narrative(inc, NarrativeStyle.EXECUTIVE)
            for inc in self.active_incidents.values()
        ]
    
    def get_recent_narratives(self, count: int = 5) -> List[str]:
        """Get narratives for recent completed incidents"""
        recent = self.completed_incidents[-count:]
        return [
            self.generate_narrative(inc, NarrativeStyle.SECURITY)
            for inc in reversed(recent)
        ]
