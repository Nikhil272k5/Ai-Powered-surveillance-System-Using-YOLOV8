"""
AbnoGuard Cognition - Ethics Layer
Context-aware interpretation to avoid harmful assumptions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class ContextType(Enum):
    """Types of contextual information that affect interpretation"""
    DEMOGRAPHIC = "demographic"      # Age, apparent vulnerability
    TEMPORAL = "temporal"            # Time of day, scheduled events
    ENVIRONMENTAL = "environmental"  # Weather, lighting
    SOCIAL = "social"                # Group dynamics, cultural context
    OPERATIONAL = "operational"      # Scheduled drills, maintenance


class AlertModification(Enum):
    """How ethics layer modifies an alert"""
    SUPPRESS = "suppress"            # Don't raise alert
    REDUCE = "reduce"                # Lower severity
    MAINTAIN = "maintain"            # Keep as-is
    ELEVATE = "elevate"              # Increase severity (protect vulnerable)
    REFRAME = "reframe"              # Change interpretation


@dataclass
class EthicalAssessment:
    """Result of ethical evaluation"""
    original_severity: str
    modified_severity: str
    modification: AlertModification
    context_factors: List[str]
    reasoning: str
    should_alert: bool


class EthicsLayer:
    """
    Ethical and social context awareness layer.
    Prevents harmful assumptions and biased alerts.
    
    Key principles:
    1. Context matters - same behavior means different things in different situations
    2. Protect the vulnerable - children, elderly, people in distress
    3. Avoid harmful stereotyping - no demographic-based suspicion
    4. Consider alternative explanations - not everything unusual is threatening
    5. Proportional response - severity should match actual risk
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Scheduled events that affect interpretation
        self.scheduled_events: List[Dict] = []
        
        # Context rules (condition -> modification)
        self.suppression_rules = [
            # Loitering in waiting areas is expected
            {
                'alert_type': 'loitering',
                'zone_type': 'waiting',
                'action': AlertModification.SUPPRESS,
                'reason': "Loitering is expected behavior in waiting areas"
            },
            # Running during scheduled fire drill
            {
                'alert_type': 'running',
                'scheduled_event': 'fire_drill',
                'action': AlertModification.SUPPRESS,
                'reason': "Running is expected during scheduled drill"
            },
            # Brief stops in transit areas
            {
                'alert_type': 'loitering',
                'zone_type': 'transit',
                'duration_under': 60,
                'action': AlertModification.SUPPRESS,
                'reason': "Brief stops in transit areas are normal"
            }
        ]
        
        self.elevation_rules = [
            # Child alone in any zone
            {
                'entity_type': 'child_alone',
                'action': AlertModification.ELEVATE,
                'new_severity': 'high',
                'reason': "Unaccompanied child requires immediate attention"
            },
            # Person fallen
            {
                'posture': 'fallen',
                'action': AlertModification.ELEVATE,
                'new_severity': 'high',
                'reason': "Person may need medical assistance"
            },
            # Person in distress
            {
                'stress_level': 'critical',
                'action': AlertModification.ELEVATE,
                'new_severity': 'high',
                'reason': "Person showing signs of distress"
            }
        ]
        
        self.reframe_rules = [
            # Running to exit might be normal departure
            {
                'alert_type': 'running',
                'zone_type': 'exit',
                'reframe_as': 'hurried_departure',
                'reduce_severity': True,
                'reason': "Running near exit often indicates hurried departure, not threat"
            }
        ]
        
        # Time-based context
        self.time_contexts = {
            'late_night': {'hour_range': (23, 5), 'heightened_sensitivity': True},
            'business_hours': {'hour_range': (9, 17), 'normal_traffic': True},
            'rush_hour': {'hour_range': (7, 9), 'crowding_expected': True}
        }
        
        print("⚖️ Ethics Layer initialized")
    
    def evaluate(self, alert: Dict, context: Dict) -> EthicalAssessment:
        """
        Evaluate an alert through ethical lens
        
        Args:
            alert: The alert to evaluate
            context: Current context information
        
        Returns:
            EthicalAssessment with modifications if any
        """
        original_severity = alert.get('severity', 'medium')
        alert_type = alert.get('type', 'unknown')
        
        factors = []
        
        # Check suppression rules
        for rule in self.suppression_rules:
            if self._matches_rule(rule, alert, context):
                return EthicalAssessment(
                    original_severity=original_severity,
                    modified_severity='none',
                    modification=AlertModification.SUPPRESS,
                    context_factors=[rule['reason']],
                    reasoning=rule['reason'],
                    should_alert=False
                )
        
        # Check elevation rules
        for rule in self.elevation_rules:
            if self._matches_rule(rule, alert, context):
                factors.append(rule['reason'])
                return EthicalAssessment(
                    original_severity=original_severity,
                    modified_severity=rule.get('new_severity', 'high'),
                    modification=AlertModification.ELEVATE,
                    context_factors=factors,
                    reasoning=rule['reason'],
                    should_alert=True
                )
        
        # Check reframe rules
        for rule in self.reframe_rules:
            if self._matches_rule(rule, alert, context):
                new_severity = 'low' if rule.get('reduce_severity') else original_severity
                factors.append(rule['reason'])
                return EthicalAssessment(
                    original_severity=original_severity,
                    modified_severity=new_severity,
                    modification=AlertModification.REFRAME,
                    context_factors=factors,
                    reasoning=f"Reframed as {rule['reframe_as']}: {rule['reason']}",
                    should_alert=True
                )
        
        # Check time-based context
        time_factor = self._get_time_context(context)
        if time_factor:
            factors.append(time_factor)
        
        # Check for vulnerable individuals
        vulnerability = self._check_vulnerability(context)
        if vulnerability:
            factors.append(vulnerability)
            return EthicalAssessment(
                original_severity=original_severity,
                modified_severity='high',
                modification=AlertModification.ELEVATE,
                context_factors=factors,
                reasoning=vulnerability,
                should_alert=True
            )
        
        # Default: maintain original
        return EthicalAssessment(
            original_severity=original_severity,
            modified_severity=original_severity,
            modification=AlertModification.MAINTAIN,
            context_factors=factors if factors else ["No contextual modifications applied"],
            reasoning="Alert evaluated, no ethical modifications required",
            should_alert=True
        )
    
    def _matches_rule(self, rule: Dict, alert: Dict, context: Dict) -> bool:
        """Check if a rule matches the current situation"""
        for key, value in rule.items():
            if key in ['action', 'reason', 'new_severity', 'reframe_as', 'reduce_severity']:
                continue
            
            # Check in alert
            if key in alert and alert[key] != value:
                return False
            
            # Check in context
            if key in context and context[key] != value:
                return False
            
            # Special checks
            if key == 'duration_under':
                if context.get('duration', 999) >= value:
                    return False
            
            if key == 'scheduled_event':
                if not self._has_scheduled_event(value):
                    return False
        
        return True
    
    def _get_time_context(self, context: Dict) -> Optional[str]:
        """Get time-based context"""
        current_hour = context.get('hour', time.localtime().tm_hour)
        
        for name, tc in self.time_contexts.items():
            start, end = tc['hour_range']
            
            if start <= end:
                in_range = start <= current_hour < end
            else:  # Wraps around midnight
                in_range = current_hour >= start or current_hour < end
            
            if in_range:
                if tc.get('heightened_sensitivity'):
                    return f"Late night hours - heightened monitoring"
                elif tc.get('crowding_expected'):
                    return f"Rush hour - crowding expected"
        
        return None
    
    def _check_vulnerability(self, context: Dict) -> Optional[str]:
        """Check for vulnerable individuals"""
        if context.get('appears_child'):
            if not context.get('has_accompanying_adult'):
                return "Unaccompanied minor detected - priority monitoring"
        
        if context.get('appears_elderly'):
            if context.get('posture') == 'fallen':
                return "Elderly person fallen - immediate assistance needed"
        
        if context.get('stress_level') == 'critical':
            return "Person in apparent distress - welfare check recommended"
        
        return None
    
    def _has_scheduled_event(self, event_type: str) -> bool:
        """Check if a scheduled event is currently active"""
        current_time = time.time()
        
        for event in self.scheduled_events:
            if event['type'] == event_type:
                if event['start'] <= current_time <= event['end']:
                    return True
        
        return False
    
    def add_scheduled_event(self, event_type: str, start_time: float, 
                           end_time: float, description: str = ""):
        """Add a scheduled event that affects interpretation"""
        self.scheduled_events.append({
            'type': event_type,
            'start': start_time,
            'end': end_time,
            'description': description
        })
    
    def clear_expired_events(self):
        """Remove expired scheduled events"""
        current_time = time.time()
        self.scheduled_events = [
            e for e in self.scheduled_events 
            if e['end'] > current_time
        ]
    
    def get_active_contexts(self) -> List[str]:
        """Get list of active contextual factors"""
        contexts = []
        
        # Time context
        hour = time.localtime().tm_hour
        for name, tc in self.time_contexts.items():
            start, end = tc['hour_range']
            if start <= end:
                if start <= hour < end:
                    contexts.append(name)
            else:
                if hour >= start or hour < end:
                    contexts.append(name)
        
        # Scheduled events
        current_time = time.time()
        for event in self.scheduled_events:
            if event['start'] <= current_time <= event['end']:
                contexts.append(f"scheduled:{event['type']}")
        
        return contexts
    
    def explain_decision(self, assessment: EthicalAssessment) -> str:
        """Generate human-readable explanation of ethical decision"""
        lines = []
        
        if assessment.modification == AlertModification.SUPPRESS:
            lines.append(f"Alert SUPPRESSED: {assessment.reasoning}")
        elif assessment.modification == AlertModification.ELEVATE:
            lines.append(f"Alert ELEVATED from {assessment.original_severity} to {assessment.modified_severity}")
            lines.append(f"Reason: {assessment.reasoning}")
        elif assessment.modification == AlertModification.REFRAME:
            lines.append(f"Alert REFRAMED: {assessment.reasoning}")
        elif assessment.modification == AlertModification.REDUCE:
            lines.append(f"Alert REDUCED from {assessment.original_severity} to {assessment.modified_severity}")
        else:
            lines.append(f"Alert MAINTAINED at {assessment.original_severity}")
        
        if assessment.context_factors:
            lines.append(f"Context factors: {', '.join(assessment.context_factors)}")
        
        return "\n".join(lines)
