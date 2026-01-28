"""
AbnoGuard Cognition - Incident Brief Generator
Produces professional security incident briefs automatically
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class IncidentBrief:
    """Professional security incident brief"""
    brief_id: str
    generated_at: float
    
    # Header info
    incident_type: str
    severity_level: str
    status: str  # active, resolved, under_review
    
    # Timeline
    start_time: float
    end_time: Optional[float]
    duration_seconds: float
    
    # Location
    zone_name: str
    camera_id: str
    
    # Subjects
    subject_count: int
    subject_descriptions: List[str]
    
    # Event summary
    executive_summary: str
    detailed_narrative: str
    
    # Risk assessment
    risk_score: float
    risk_category: str
    
    # Actions
    recommended_actions: List[str]
    actions_taken: List[str]
    
    # Evidence
    snapshot_paths: List[str]
    video_clip_path: Optional[str]
    
    # Metadata
    confidence_level: float
    generated_by: str


class BriefGenerator:
    """
    Generates professional security incident briefs automatically.
    Produces 1-minute readable summaries for security personnel.
    
    Brief Format:
    1. Executive Summary (2 sentences)
    2. Timeline (key events)
    3. Subjects involved
    4. Risk assessment
    5. Recommended actions
    6. Evidence references
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Generated briefs
        self.briefs: List[IncidentBrief] = []
        
        # Templates
        self.severity_descriptions = {
            'low': 'minor security observation',
            'medium': 'notable security event',
            'high': 'significant security incident',
            'critical': 'critical security emergency'
        }
        
        self.action_templates = {
            'loitering': [
                "Monitor subject for continued presence",
                "Consider verbal check if behavior persists",
                "Document for pattern analysis"
            ],
            'abandoned_object': [
                "Dispatch security to investigate object",
                "Maintain safe perimeter if suspicious",
                "Attempt to locate owner via PA system"
            ],
            'restricted_zone': [
                "Immediate security response required",
                "Verify authorization status",
                "Escort from restricted area if unauthorized"
            ],
            'speed_spike': [
                "Monitor for signs of pursuit or emergency",
                "Check for related incidents in exit areas",
                "Document for review"
            ],
            'counterflow': [
                "Observe for potential crowd issues",
                "Prepare for flow management if needed",
                "Standard documentation"
            ],
            'default': [
                "Continue monitoring",
                "Document for records",
                "Escalate if behavior continues"
            ]
        }
        
        print("📄 Incident Brief Generator initialized")
    
    def generate_brief(self, incident_data: Dict, 
                      narrative: str,
                      risk_assessment: Dict,
                      snapshots: List[str] = None) -> IncidentBrief:
        """
        Generate a complete incident brief
        
        Args:
            incident_data: Incident information
            narrative: Pre-generated narrative
            risk_assessment: Risk matrix output
            snapshots: Paths to evidence screenshots
        
        Returns:
            IncidentBrief ready for delivery
        """
        brief_id = f"BRIEF-{int(time.time())}-{len(self.briefs):04d}"
        current_time = time.time()
        
        # Extract data
        incident_type = incident_data.get('type', 'unknown')
        severity = incident_data.get('severity', 'medium')
        start_time = incident_data.get('start_time', current_time)
        end_time = incident_data.get('end_time')
        zone = incident_data.get('zone', 'Main Area')
        camera = incident_data.get('camera_id', 'CAM-01')
        track_ids = incident_data.get('track_ids', [])
        status = 'resolved' if end_time else 'active'
        
        # Calculate duration
        duration = (end_time or current_time) - start_time
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            incident_type, severity, zone, duration, status
        )
        
        # Generate subject descriptions
        subjects = self._generate_subject_descriptions(track_ids, incident_data)
        
        # Get recommended actions
        actions = self._get_recommended_actions(incident_type, severity)
        
        # Get risk info
        risk_score = risk_assessment.get('risk_score', 0.5)
        risk_category = risk_assessment.get('risk_category', 'investigate')
        if hasattr(risk_category, 'value'):
            risk_category = risk_category.value
        
        # Build brief
        brief = IncidentBrief(
            brief_id=brief_id,
            generated_at=current_time,
            incident_type=incident_type,
            severity_level=severity,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            zone_name=zone,
            camera_id=camera,
            subject_count=len(track_ids),
            subject_descriptions=subjects,
            executive_summary=exec_summary,
            detailed_narrative=narrative,
            risk_score=risk_score,
            risk_category=risk_category,
            recommended_actions=actions,
            actions_taken=incident_data.get('actions_taken', []),
            snapshot_paths=snapshots or [],
            video_clip_path=incident_data.get('video_clip'),
            confidence_level=incident_data.get('confidence', 0.8),
            generated_by="AbnoGuard Intelligence System"
        )
        
        self.briefs.append(brief)
        return brief
    
    def _generate_executive_summary(self, incident_type: str, severity: str,
                                    zone: str, duration: float, status: str) -> str:
        """Generate 2-sentence executive summary"""
        
        severity_desc = self.severity_descriptions.get(severity, 'security event')
        incident_desc = incident_type.replace('_', ' ').title()
        
        # First sentence: What happened
        sentence1 = f"A {severity_desc} of type '{incident_desc}' was detected in {zone}."
        
        # Second sentence: Duration and status
        if duration < 60:
            duration_str = f"{duration:.0f} seconds"
        else:
            duration_str = f"{duration/60:.1f} minutes"
        
        if status == 'active':
            sentence2 = f"The incident is currently active, ongoing for {duration_str}."
        else:
            sentence2 = f"The incident has been resolved after {duration_str}."
        
        return f"{sentence1} {sentence2}"
    
    def _generate_subject_descriptions(self, track_ids: List[int],
                                       incident_data: Dict) -> List[str]:
        """Generate descriptions of subjects involved"""
        descriptions = []
        
        subject_data = incident_data.get('subjects', {})
        
        for track_id in track_ids:
            subj = subject_data.get(track_id, {})
            
            desc = f"Subject #{track_id}"
            
            if subj.get('class') == 'person':
                desc += " (Person)"
            elif subj.get('class'):
                desc += f" ({subj['class'].title()})"
            
            if subj.get('first_seen'):
                first_time = datetime.fromtimestamp(subj['first_seen']).strftime('%H:%M:%S')
                desc += f" - First seen at {first_time}"
            
            if subj.get('current_zone'):
                desc += f" in {subj['current_zone']}"
            
            descriptions.append(desc)
        
        if not descriptions:
            descriptions.append("Subject details not available")
        
        return descriptions
    
    def _get_recommended_actions(self, incident_type: str, severity: str) -> List[str]:
        """Get recommended actions for incident type"""
        actions = []
        
        # Type-specific actions
        for key in self.action_templates:
            if key in incident_type:
                actions = self.action_templates[key].copy()
                break
        
        if not actions:
            actions = self.action_templates['default'].copy()
        
        # Add severity-specific actions
        if severity == 'high':
            actions.insert(0, "Priority response - assign senior personnel")
        elif severity == 'critical':
            actions.insert(0, "EMERGENCY - Initiate emergency protocols")
            actions.insert(1, "Contact authorities if not already done")
        
        return actions
    
    def format_brief_text(self, brief: IncidentBrief) -> str:
        """Format brief as readable text document"""
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("                    SECURITY INCIDENT BRIEF")
        lines.append("=" * 70)
        lines.append(f"Brief ID: {brief.brief_id}")
        lines.append(f"Generated: {datetime.fromtimestamp(brief.generated_at).strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Status: {brief.status.upper()}")
        lines.append("-" * 70)
        
        # Executive Summary
        lines.append("\n📋 EXECUTIVE SUMMARY")
        lines.append(brief.executive_summary)
        
        # Key Details
        lines.append("\n📊 KEY DETAILS")
        lines.append(f"  Incident Type: {brief.incident_type.replace('_', ' ').title()}")
        lines.append(f"  Severity: {brief.severity_level.upper()}")
        lines.append(f"  Location: {brief.zone_name} ({brief.camera_id})")
        lines.append(f"  Duration: {brief.duration_seconds:.0f} seconds")
        start_str = datetime.fromtimestamp(brief.start_time).strftime('%H:%M:%S')
        lines.append(f"  Start Time: {start_str}")
        if brief.end_time:
            end_str = datetime.fromtimestamp(brief.end_time).strftime('%H:%M:%S')
            lines.append(f"  End Time: {end_str}")
        
        # Subjects
        lines.append("\n👤 SUBJECTS INVOLVED")
        lines.append(f"  Count: {brief.subject_count}")
        for desc in brief.subject_descriptions:
            lines.append(f"  • {desc}")
        
        # Narrative
        lines.append("\n📝 DETAILED NARRATIVE")
        for line in brief.detailed_narrative.split('\n'):
            lines.append(f"  {line}")
        
        # Risk Assessment
        lines.append("\n⚠️ RISK ASSESSMENT")
        lines.append(f"  Risk Score: {brief.risk_score:.2f} / 1.00")
        lines.append(f"  Category: {brief.risk_category.upper()}")
        lines.append(f"  Confidence: {brief.confidence_level:.0%}")
        
        # Recommended Actions
        lines.append("\n✅ RECOMMENDED ACTIONS")
        for i, action in enumerate(brief.recommended_actions, 1):
            lines.append(f"  {i}. {action}")
        
        # Actions Taken
        if brief.actions_taken:
            lines.append("\n📋 ACTIONS TAKEN")
            for action in brief.actions_taken:
                lines.append(f"  ✓ {action}")
        
        # Evidence
        if brief.snapshot_paths or brief.video_clip_path:
            lines.append("\n📷 EVIDENCE")
            for snap in brief.snapshot_paths:
                lines.append(f"  • Snapshot: {snap}")
            if brief.video_clip_path:
                lines.append(f"  • Video: {brief.video_clip_path}")
        
        # Footer
        lines.append("\n" + "-" * 70)
        lines.append(f"Generated by: {brief.generated_by}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def format_brief_html(self, brief: IncidentBrief) -> str:
        """Format brief as HTML for web display"""
        severity_colors = {
            'low': '#4ade80',
            'medium': '#fbbf24',
            'high': '#f97316',
            'critical': '#ef4444'
        }
        
        color = severity_colors.get(brief.severity_level, '#6b7280')
        
        html = f"""
        <div class="brief-container">
            <div class="brief-header" style="border-left: 4px solid {color};">
                <h2>Security Incident Brief</h2>
                <span class="brief-id">{brief.brief_id}</span>
                <span class="status status-{brief.status}">{brief.status.upper()}</span>
            </div>
            
            <div class="brief-summary">
                <h3>Executive Summary</h3>
                <p>{brief.executive_summary}</p>
            </div>
            
            <div class="brief-details">
                <div class="detail-item">
                    <label>Type:</label>
                    <span>{brief.incident_type.replace('_', ' ').title()}</span>
                </div>
                <div class="detail-item">
                    <label>Severity:</label>
                    <span class="severity-{brief.severity_level}">{brief.severity_level.upper()}</span>
                </div>
                <div class="detail-item">
                    <label>Location:</label>
                    <span>{brief.zone_name}</span>
                </div>
                <div class="detail-item">
                    <label>Duration:</label>
                    <span>{brief.duration_seconds:.0f}s</span>
                </div>
                <div class="detail-item">
                    <label>Risk Score:</label>
                    <span>{brief.risk_score:.2f}</span>
                </div>
            </div>
            
            <div class="brief-actions">
                <h3>Recommended Actions</h3>
                <ul>
                    {''.join(f'<li>{a}</li>' for a in brief.recommended_actions)}
                </ul>
            </div>
            
            <div class="brief-footer">
                <small>Generated: {datetime.fromtimestamp(brief.generated_at).strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
        </div>
        """
        
        return html
    
    def get_recent_briefs(self, count: int = 10) -> List[IncidentBrief]:
        """Get most recent briefs"""
        return self.briefs[-count:][::-1]
    
    def export_brief_json(self, brief: IncidentBrief) -> Dict:
        """Export brief as JSON-serializable dictionary"""
        return {
            'brief_id': brief.brief_id,
            'generated_at': brief.generated_at,
            'incident_type': brief.incident_type,
            'severity_level': brief.severity_level,
            'status': brief.status,
            'start_time': brief.start_time,
            'end_time': brief.end_time,
            'duration_seconds': brief.duration_seconds,
            'zone_name': brief.zone_name,
            'camera_id': brief.camera_id,
            'subject_count': brief.subject_count,
            'subject_descriptions': brief.subject_descriptions,
            'executive_summary': brief.executive_summary,
            'detailed_narrative': brief.detailed_narrative,
            'risk_score': brief.risk_score,
            'risk_category': brief.risk_category,
            'recommended_actions': brief.recommended_actions,
            'actions_taken': brief.actions_taken,
            'snapshot_paths': brief.snapshot_paths,
            'confidence_level': brief.confidence_level
        }
