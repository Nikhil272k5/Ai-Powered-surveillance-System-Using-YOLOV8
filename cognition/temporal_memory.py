"""
AbnoGuard Cognition - Temporal Memory
Remembers past incidents for pattern detection and historical context
"""

import sqlite3
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class PatternType(Enum):
    REPETITION = "repetition"       # Same event happening again
    ESCALATION = "escalation"       # Increasing threat level
    CYCLIC = "cyclic"               # Time-based patterns
    LOCATION_BOUND = "location"     # Patterns tied to locations
    ENTITY_BOUND = "entity"         # Patterns tied to individuals


@dataclass
class MemoryEntry:
    """A single memory entry (incident record)"""
    memory_id: str
    timestamp: float
    incident_type: str
    threat_level: str
    zone_id: Optional[str]
    track_ids: List[int]
    duration: float
    description: str
    narrative: str
    outcome: Optional[str]  # confirmed, false_positive, resolved
    metadata: Dict[str, Any]


@dataclass
class Pattern:
    """Detected pattern from memory analysis"""
    pattern_type: PatternType
    confidence: float
    description: str
    related_memories: List[str]  # memory_ids
    first_occurrence: float
    last_occurrence: float
    frequency: int
    prediction: str  # What might happen next


class TemporalMemory:
    """
    Episodic memory for past incidents and patterns.
    Enables historical reasoning and pattern detection.
    
    Features:
    - SQLite database for persistence
    - Pattern detection across time
    - Entity tracking over days
    - Contextual recall for current situations
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Database setup
        db_path = self.config.get('db_path', 'data/incidents.db')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        # Pattern detection settings
        self.repetition_window = self.config.get('repetition_window', 3600)  # 1 hour
        self.repetition_threshold = self.config.get('repetition_threshold', 3)
        
        # Cache for quick access
        self.recent_cache: List[MemoryEntry] = []
        self.cache_size = self.config.get('cache_size', 100)
        
        # Load recent memories into cache
        self._load_recent_to_cache()
        
        print(f"🧠 Temporal Memory initialized ({self.db_path})")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main incidents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                memory_id TEXT PRIMARY KEY,
                timestamp REAL,
                incident_type TEXT,
                threat_level TEXT,
                zone_id TEXT,
                track_ids TEXT,
                duration REAL,
                description TEXT,
                narrative TEXT,
                outcome TEXT,
                metadata TEXT
            )
        ''')
        
        # Patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                confidence REAL,
                description TEXT,
                related_memories TEXT,
                first_occurrence REAL,
                last_occurrence REAL,
                frequency INTEGER,
                prediction TEXT,
                updated_at REAL
            )
        ''')
        
        # Entity history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_history (
                entity_id TEXT,
                first_seen REAL,
                last_seen REAL,
                incident_count INTEGER,
                zones_visited TEXT,
                threat_scores TEXT,
                PRIMARY KEY (entity_id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON incidents(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON incidents(incident_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone ON incidents(zone_id)')
        
        conn.commit()
        conn.close()
    
    def _load_recent_to_cache(self):
        """Load recent memories into cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM incidents 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (self.cache_size,))
        
        for row in cursor.fetchall():
            entry = self._row_to_entry(row)
            self.recent_cache.append(entry)
        
        conn.close()
    
    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        return MemoryEntry(
            memory_id=row[0],
            timestamp=row[1],
            incident_type=row[2],
            threat_level=row[3],
            zone_id=row[4],
            track_ids=json.loads(row[5]) if row[5] else [],
            duration=row[6],
            description=row[7],
            narrative=row[8],
            outcome=row[9],
            metadata=json.loads(row[10]) if row[10] else {}
        )
    
    def remember(self, entry: MemoryEntry) -> str:
        """Store a new memory (incident)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO incidents 
            (memory_id, timestamp, incident_type, threat_level, zone_id,
             track_ids, duration, description, narrative, outcome, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.memory_id,
            entry.timestamp,
            entry.incident_type,
            entry.threat_level,
            entry.zone_id,
            json.dumps(entry.track_ids),
            entry.duration,
            entry.description,
            entry.narrative,
            entry.outcome,
            json.dumps(entry.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # Update cache
        self.recent_cache.insert(0, entry)
        if len(self.recent_cache) > self.cache_size:
            self.recent_cache.pop()
        
        # Check for patterns
        self._detect_patterns_async(entry)
        
        return entry.memory_id
    
    def recall(self, memory_id: str) -> Optional[MemoryEntry]:
        """Recall a specific memory by ID"""
        # Check cache first
        for entry in self.recent_cache:
            if entry.memory_id == memory_id:
                return entry
        
        # Query database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM incidents WHERE memory_id = ?', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_entry(row)
        return None
    
    def recall_similar(self, incident_type: str, zone_id: Optional[str] = None,
                      time_window: float = 86400) -> List[MemoryEntry]:
        """Recall similar past incidents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        min_time = time.time() - time_window
        
        if zone_id:
            cursor.execute('''
                SELECT * FROM incidents 
                WHERE incident_type = ? AND zone_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (incident_type, zone_id, min_time))
        else:
            cursor.execute('''
                SELECT * FROM incidents 
                WHERE incident_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (incident_type, min_time))
        
        results = [self._row_to_entry(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def recall_recent(self, count: int = 10) -> List[MemoryEntry]:
        """Recall most recent memories"""
        return self.recent_cache[:count]
    
    def recall_by_timeframe(self, start_time: float, end_time: float) -> List[MemoryEntry]:
        """Recall memories within a time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM incidents 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        ''', (start_time, end_time))
        
        results = [self._row_to_entry(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def _detect_patterns_async(self, new_entry: MemoryEntry):
        """Detect patterns based on new memory (async in production)"""
        # Repetition detection
        similar = self.recall_similar(
            new_entry.incident_type,
            new_entry.zone_id,
            time_window=self.repetition_window
        )
        
        if len(similar) >= self.repetition_threshold:
            pattern = Pattern(
                pattern_type=PatternType.REPETITION,
                confidence=min(1.0, len(similar) / 5),
                description=f"Repeated {new_entry.incident_type} incidents in {new_entry.zone_id or 'area'}",
                related_memories=[m.memory_id for m in similar],
                first_occurrence=similar[-1].timestamp if similar else new_entry.timestamp,
                last_occurrence=new_entry.timestamp,
                frequency=len(similar),
                prediction=f"Similar incident likely in next {self.repetition_window // 60} minutes"
            )
            self._save_pattern(pattern)
    
    def _save_pattern(self, pattern: Pattern):
        """Save detected pattern to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pattern_id = f"PAT-{int(time.time())}-{pattern.pattern_type.value}"
        
        cursor.execute('''
            INSERT OR REPLACE INTO patterns
            (pattern_id, pattern_type, confidence, description, related_memories,
             first_occurrence, last_occurrence, frequency, prediction, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            pattern.pattern_type.value,
            pattern.confidence,
            pattern.description,
            json.dumps(pattern.related_memories),
            pattern.first_occurrence,
            pattern.last_occurrence,
            pattern.frequency,
            pattern.prediction,
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_patterns(self, min_confidence: float = 0.5) -> List[Pattern]:
        """Get currently active patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patterns from last 24 hours
        min_time = time.time() - 86400
        
        cursor.execute('''
            SELECT * FROM patterns 
            WHERE confidence >= ? AND last_occurrence > ?
            ORDER BY confidence DESC
        ''', (min_confidence, min_time))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append(Pattern(
                pattern_type=PatternType(row[1]),
                confidence=row[2],
                description=row[3],
                related_memories=json.loads(row[4]) if row[4] else [],
                first_occurrence=row[5],
                last_occurrence=row[6],
                frequency=row[7],
                prediction=row[8]
            ))
        
        conn.close()
        return patterns
    
    def compare_to_history(self, current_incident: Dict) -> Dict:
        """Compare current incident to historical patterns"""
        result = {
            'is_unusual': False,
            'similar_incidents_24h': 0,
            'similar_incidents_7d': 0,
            'patterns_triggered': [],
            'historical_context': ""
        }
        
        incident_type = current_incident.get('type')
        zone_id = current_incident.get('zone_id')
        
        # Count similar in last 24 hours
        similar_24h = self.recall_similar(incident_type, zone_id, 86400)
        result['similar_incidents_24h'] = len(similar_24h)
        
        # Count similar in last 7 days
        similar_7d = self.recall_similar(incident_type, zone_id, 86400 * 7)
        result['similar_incidents_7d'] = len(similar_7d)
        
        # Check if this is unusual
        if len(similar_24h) == 0 and len(similar_7d) <= 2:
            result['is_unusual'] = True
            result['historical_context'] = f"This is an unusual event. Only {len(similar_7d)} similar incidents in the past week."
        elif len(similar_24h) > 5:
            result['historical_context'] = f"This is a recurring issue. {len(similar_24h)} similar incidents today."
        else:
            result['historical_context'] = f"This type of incident occurs occasionally. {len(similar_7d)} times this week."
        
        # Check patterns
        for pattern in self.get_active_patterns():
            if pattern.pattern_type == PatternType.REPETITION:
                if incident_type in pattern.description:
                    result['patterns_triggered'].append(pattern.description)
        
        return result
    
    def get_statistics(self, days: int = 7) -> Dict:
        """Get memory statistics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        min_time = time.time() - (days * 86400)
        
        # Count by type
        cursor.execute('''
            SELECT incident_type, COUNT(*) as count
            FROM incidents 
            WHERE timestamp > ?
            GROUP BY incident_type
        ''', (min_time,))
        
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by zone
        cursor.execute('''
            SELECT zone_id, COUNT(*) as count
            FROM incidents 
            WHERE timestamp > ? AND zone_id IS NOT NULL
            GROUP BY zone_id
        ''', (min_time,))
        
        zone_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Total count
        cursor.execute('SELECT COUNT(*) FROM incidents WHERE timestamp > ?', (min_time,))
        total = cursor.fetchone()[0]
        
        # Active patterns
        patterns = self.get_active_patterns()
        
        conn.close()
        
        return {
            'total_incidents': total,
            'by_type': type_counts,
            'by_zone': zone_counts,
            'active_patterns': len(patterns),
            'pattern_descriptions': [p.description for p in patterns[:5]],
            'period_days': days
        }
