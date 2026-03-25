"""
Initialize default playbooks and zone policies for SOC
"""
import uuid
from datetime import datetime
from .database import threat_db


def init_default_playbooks():
    """Create default response playbooks"""
    
    playbooks = [
        {
            "playbook_id": "PB-RECON-001",
            "name": "Reconnaissance Response",
            "description": "Response to hostile reconnaissance detection",
            "threat_types": ["hostile_reconnaissance", "security_probing"],
            "severity_levels": ["medium", "high"],
            "steps": [
                {"name": "Alert Security Team", "action": "notify", "target": "security_team"},
                {"name": "Increase Camera Focus", "action": "camera_zoom", "target": "threat_zone"},
                {"name": "Deploy Patrol", "action": "dispatch", "target": "security_patrol"},
                {"name": "Document Evidence", "action": "capture_evidence", "target": "threat_entities"}
            ],
            "is_automated": False,
            "is_active": True
        },
        {
            "playbook_id": "PB-EVADE-001",
            "name": "Evasive Behavior Response",
            "description": "Response to camera avoidance and evasive movement",
            "threat_types": ["evasive_behavior", "camera_avoidance"],
            "severity_levels": ["medium", "high"],
            "steps": [
                {"name": "Track Entity", "action": "multi_camera_track", "target": "threat_entity"},
                {"name": "Alert Supervisor", "action": "notify", "target": "supervisor"},
                {"name": "Enable Secondary Cameras", "action": "enable_backup", "target": "zone_cameras"}
            ],
            "is_automated": False,
            "is_active": True
        },
        {
            "playbook_id": "PB-ABANDON-001",
            "name": "Abandoned Object Response",
            "description": "Response to unattended bag/package detection",
            "threat_types": ["abandoned_object"],
            "severity_levels": ["high", "critical"],
            "steps": [
                {"name": "Secure Perimeter", "action": "perimeter_lockdown", "target": "threat_zone"},
                {"name": "Alert Bomb Squad", "action": "notify", "target": "bomb_disposal"},
                {"name": "Evacuate Area", "action": "evacuation", "target": "zone_radius_50m"},
                {"name": "Document Timeline", "action": "capture_timeline", "target": "threat"}
            ],
            "is_automated": False,
            "is_active": True
        },
        {
            "playbook_id": "PB-CRITICAL-001",
            "name": "Critical Threat Response",
            "description": "Emergency response to critical/pre-attack indicators",
            "threat_types": ["pre_attack_indicator", "coordinated_movement"],
            "severity_levels": ["critical"],
            "steps": [
                {"name": "Immediate Alert", "action": "emergency_broadcast", "target": "all_security"},
                {"name": "Lock All Entries", "action": "lockdown", "target": "all_entries"},
                {"name": "Notify Law Enforcement", "action": "notify", "target": "police"},
                {"name": "Enable All Cameras", "action": "full_surveillance", "target": "all_zones"},
                {"name": "Prepare Evacuation", "action": "standby_evacuation", "target": "facility"}
            ],
            "is_automated": False,
            "is_active": True
        },
        {
            "playbook_id": "PB-LOITER-001",
            "name": "Suspicious Loitering Response",
            "description": "Response to prolonged loitering in sensitive areas",
            "threat_types": ["suspicious_loitering"],
            "severity_levels": ["low", "medium"],
            "steps": [
                {"name": "Monitor Subject", "action": "enhanced_monitoring", "target": "threat_entity"},
                {"name": "Dispatch Security", "action": "dispatch", "target": "security_officer"},
                {"name": "Verbal Check", "action": "approach_inquiry", "target": "threat_entity"}
            ],
            "is_automated": False,
            "is_active": True
        }
    ]
    
    for playbook in playbooks:
        threat_db.save_playbook(playbook)
    
    print(f"[SOC] Initialized {len(playbooks)} default playbooks")


def init_default_zones():
    """Create default zone policies"""
    from datetime import datetime
    import json
    
    zones = [
        {
            "policy_id": "ZONE-MAIN-001",
            "zone_id": "zone_main",
            "zone_name": "Main Entrance",
            "policy_rules": json.dumps([
                {"rule": "loitering_threshold", "value": 300, "unit": "seconds"},
                {"rule": "crowd_limit", "value": 50, "unit": "people"},
                {"rule": "operating_hours", "start": "06:00", "end": "22:00"}
            ]),
            "threat_multiplier": 1.2,
            "is_restricted": False,
            "is_active": 1
        },
        {
            "policy_id": "ZONE-SEC-001",
            "zone_id": "zone_secure",
            "zone_name": "Secure Area",
            "policy_rules": json.dumps([
                {"rule": "authorization_required", "value": True},
                {"rule": "loitering_threshold", "value": 60, "unit": "seconds"},
                {"rule": "alert_on_unknown", "value": True}
            ]),
            "threat_multiplier": 2.0,
            "is_restricted": True,
            "is_active": 1
        },
        {
            "policy_id": "ZONE-PERI-001",
            "zone_id": "zone_perimeter",
            "zone_name": "Perimeter",
            "policy_rules": json.dumps([
                {"rule": "intrusion_detection", "value": True},
                {"rule": "night_mode_sensitivity", "value": "high"},
                {"rule": "approach_alert", "value": True}
            ]),
            "threat_multiplier": 1.5,
            "is_restricted": False,
            "is_active": 1
        }
    ]
    
    # Save directly to database
    with threat_db._get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        for zone in zones:
            cursor.execute('''
                INSERT OR REPLACE INTO zone_policies 
                (policy_id, zone_id, zone_name, policy_rules, threat_multiplier, 
                 is_restricted, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                zone["policy_id"], zone["zone_id"], zone["zone_name"],
                zone["policy_rules"], zone["threat_multiplier"],
                zone["is_restricted"], zone["is_active"], now, now
            ))
        
        conn.commit()
    
    print(f"[SOC] Initialized {len(zones)} default zones")


# Initialize on module load
try:
    init_default_playbooks()
    init_default_zones()
except Exception as e:
    print(f"[SOC] Init warning: {e}")
