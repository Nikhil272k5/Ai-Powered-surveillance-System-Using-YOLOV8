"""
Database Models for Threat Intelligence Platform

New tables that extend the existing database without modifying it.
"""
import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading
import os


class ThreatDatabase:
    """
    Database layer for Threat Intelligence Platform.
    Uses separate tables, preserves all existing tables.
    """
    
    def __init__(self, db_path: str = "threat_intelligence.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """Initialize all threat intelligence tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Threats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threats (
                    threat_id TEXT PRIMARY KEY,
                    threat_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    uncertainty REAL,
                    entity_ids TEXT,
                    zone_ids TEXT,
                    evidence_count INTEGER,
                    is_active INTEGER DEFAULT 1,
                    escalation_count INTEGER DEFAULT 0,
                    resolution_notes TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    resolved_at TEXT
                )
            ''')
            
            # Threat signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_signals (
                    signal_id TEXT PRIMARY KEY,
                    threat_id TEXT,
                    signal_type TEXT NOT NULL,
                    entity_id TEXT,
                    zone_id TEXT,
                    confidence REAL,
                    weight REAL,
                    payload TEXT,
                    created_at TEXT,
                    FOREIGN KEY (threat_id) REFERENCES threats(threat_id)
                )
            ''')
            
            # Threat profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_profiles (
                    profile_id TEXT PRIMARY KEY,
                    entity_id TEXT UNIQUE,
                    first_seen TEXT,
                    last_seen TEXT,
                    total_sightings INTEGER,
                    risk_level TEXT,
                    risk_score REAL,
                    zones_visited TEXT,
                    associated_threats TEXT,
                    tags TEXT,
                    notes TEXT,
                    is_watchlist INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Incidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    threat_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT,
                    status TEXT DEFAULT 'active',
                    assigned_to TEXT,
                    response_actions TEXT,
                    timeline TEXT,
                    evidence_ids TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    resolved_at TEXT,
                    FOREIGN KEY (threat_id) REFERENCES threats(threat_id)
                )
            ''')
            
            # Playbooks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS playbooks (
                    playbook_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    threat_types TEXT,
                    severity_levels TEXT,
                    steps TEXT,
                    is_automated INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    execution_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Zone policies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zone_policies (
                    policy_id TEXT PRIMARY KEY,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT,
                    policy_rules TEXT,
                    threat_multiplier REAL DEFAULT 1.0,
                    is_restricted INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Audit logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    action_by TEXT,
                    target_type TEXT,
                    target_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    created_at TEXT
                )
            ''')
            
            # Knowledge graph edges table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_graph (
                    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
    
    # ==================== Threats ====================
    
    def save_threat(self, threat_data: Dict[str, Any]) -> str:
        """Save or update a threat"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                threat_id = threat_data.get('threat_id')
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO threats 
                    (threat_id, threat_type, severity, title, description, 
                     confidence, uncertainty, entity_ids, zone_ids, evidence_count,
                     is_active, escalation_count, resolution_notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM threats WHERE threat_id = ?), ?), ?)
                ''', (
                    threat_id,
                    threat_data.get('threat_type'),
                    threat_data.get('severity'),
                    threat_data.get('title'),
                    threat_data.get('description'),
                    threat_data.get('confidence'),
                    threat_data.get('uncertainty'),
                    json.dumps(threat_data.get('entity_ids', [])),
                    json.dumps(threat_data.get('zone_ids', [])),
                    threat_data.get('evidence_count', 0),
                    1 if threat_data.get('is_active', True) else 0,
                    threat_data.get('escalation_count', 0),
                    threat_data.get('resolution_notes', ''),
                    threat_id, now, now
                ))
                
                conn.commit()
                return threat_id
    
    def get_threat(self, threat_id: str) -> Optional[Dict[str, Any]]:
        """Get threat by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threats WHERE threat_id = ?', (threat_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_threat_dict(row)
            return None
    
    def get_active_threats(self) -> List[Dict[str, Any]]:
        """Get all active threats"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threats WHERE is_active = 1 ORDER BY created_at DESC')
            return [self._row_to_threat_dict(row) for row in cursor.fetchall()]
    
    def get_threats_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get threats by severity"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threats WHERE severity = ? AND is_active = 1', (severity,))
            return [self._row_to_threat_dict(row) for row in cursor.fetchall()]
    
    def _row_to_threat_dict(self, row) -> Dict[str, Any]:
        return {
            'threat_id': row['threat_id'],
            'threat_type': row['threat_type'],
            'severity': row['severity'],
            'title': row['title'],
            'description': row['description'],
            'confidence': row['confidence'],
            'uncertainty': row['uncertainty'],
            'entity_ids': json.loads(row['entity_ids'] or '[]'),
            'zone_ids': json.loads(row['zone_ids'] or '[]'),
            'evidence_count': row['evidence_count'],
            'is_active': bool(row['is_active']),
            'escalation_count': row['escalation_count'],
            'resolution_notes': row['resolution_notes'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    # ==================== Incidents ====================
    
    def create_incident(self, incident_data: Dict[str, Any]) -> str:
        """Create a new incident"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                incident_id = incident_data.get('incident_id')
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT INTO incidents 
                    (incident_id, threat_id, title, description, severity, status,
                     assigned_to, response_actions, timeline, evidence_ids, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    incident_id,
                    incident_data.get('threat_id'),
                    incident_data.get('title'),
                    incident_data.get('description'),
                    incident_data.get('severity'),
                    incident_data.get('status', 'active'),
                    incident_data.get('assigned_to'),
                    json.dumps(incident_data.get('response_actions', [])),
                    json.dumps(incident_data.get('timeline', [])),
                    json.dumps(incident_data.get('evidence_ids', [])),
                    now, now
                ))
                
                conn.commit()
                return incident_id
    
    def update_incident_status(self, incident_id: str, status: str, notes: str = "") -> bool:
        """Update incident status"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                resolved_at = now if status == 'resolved' else None
                
                cursor.execute('''
                    UPDATE incidents 
                    SET status = ?, updated_at = ?, resolved_at = COALESCE(?, resolved_at)
                    WHERE incident_id = ?
                ''', (status, now, resolved_at, incident_id))
                
                conn.commit()
                return cursor.rowcount > 0

    def update_incident_description(self, incident_id: str, description: str) -> bool:
        """Update incident description"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    UPDATE incidents 
                    SET description = ?, updated_at = ?
                    WHERE incident_id = ?
                ''', (description, now, incident_id))
                
                conn.commit()
                return cursor.rowcount > 0
    
    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE status != 'resolved' ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Playbooks ====================
    
    def save_playbook(self, playbook_data: Dict[str, Any]) -> str:
        """Save or update a playbook"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                playbook_id = playbook_data.get('playbook_id')
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO playbooks 
                    (playbook_id, name, description, threat_types, severity_levels,
                     steps, is_automated, is_active, execution_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT execution_count FROM playbooks WHERE playbook_id = ?), 0),
                            COALESCE((SELECT created_at FROM playbooks WHERE playbook_id = ?), ?), ?)
                ''', (
                    playbook_id,
                    playbook_data.get('name'),
                    playbook_data.get('description'),
                    json.dumps(playbook_data.get('threat_types', [])),
                    json.dumps(playbook_data.get('severity_levels', [])),
                    json.dumps(playbook_data.get('steps', [])),
                    1 if playbook_data.get('is_automated', False) else 0,
                    1 if playbook_data.get('is_active', True) else 0,
                    playbook_id, playbook_id, now, now
                ))
                
                conn.commit()
                return playbook_id
    
    def get_playbooks(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all playbooks"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute('SELECT * FROM playbooks WHERE is_active = 1')
            else:
                cursor.execute('SELECT * FROM playbooks')
            
            result = []
            for row in cursor.fetchall():
                item = dict(row)
                item['threat_types'] = json.loads(item['threat_types'] or '[]')
                item['severity_levels'] = json.loads(item['severity_levels'] or '[]')
                item['steps'] = json.loads(item['steps'] or '[]')
                result.append(item)
            return result
    
    # ==================== Audit Logs ====================
    
    def log_action(self, action_type: str, action_by: str, target_type: str, 
                   target_id: str, details: Dict[str, Any] = None):
        """Log an audit action"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_logs 
                    (action_type, action_by, target_type, target_id, details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    action_type,
                    action_by,
                    target_type,
                    target_id,
                    json.dumps(details or {}),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
    
    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit logs"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Zone Policies ====================
    
    def get_zone_policies(self) -> List[Dict[str, Any]]:
        """Get all zone policies - returns default zones if none exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM zone_policies WHERE is_active = 1')
            zones = [dict(row) for row in cursor.fetchall()]
            
            # If no zones, return defaults
            if not zones:
                return [
                    {"zone_id": "entry", "zone_name": "Entry Gate", "is_restricted": False},
                    {"zone_id": "main", "zone_name": "Main Area", "is_restricted": False},
                    {"zone_id": "restricted", "zone_name": "Restricted Zone", "is_restricted": True},
                    {"zone_id": "parking", "zone_name": "Parking", "is_restricted": False},
                    {"zone_id": "lobby", "zone_name": "Lobby", "is_restricted": False},
                    {"zone_id": "server", "zone_name": "Server Room", "is_restricted": True},
                ]
            return zones


# Global database instance
threat_db = ThreatDatabase()
