"""
LAYER 2: TEMPORAL MEMORY
Persistent storage for objects, behaviors, and events.
"""
import sqlite3
import json
import time
import os

class TemporalMemory:
    def __init__(self, db_path='system_memory.db'):
        print("🧠 Initializing Memory Layer...")
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Entity Table: Tracks people/objects
        c.execute('''CREATE TABLE IF NOT EXISTS entities
                     (id INTEGER PRIMARY KEY, track_id INTEGER, first_seen REAL, last_seen REAL, 
                      class TEXT, total_incidents INTEGER, threat_level REAL)''')
                      
        # Event Table: Discrete events
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (id INTEGER PRIMARY KEY, timestamp REAL, type TEXT, location TEXT, 
                      description TEXT, metadata JSON)''')
                      
        conn.commit()
        conn.close()

    def update_entity(self, track_id, class_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT id FROM entities WHERE track_id=?", (track_id,))
        row = c.fetchone()
        
        now = time.time()
        if row:
            c.execute("UPDATE entities SET last_seen=? WHERE track_id=?", (now, track_id))
        else:
            c.execute("INSERT INTO entities (track_id, first_seen, last_seen, class, total_incidents, threat_level) VALUES (?, ?, ?, ?, 0, 0.0)",
                      (track_id, now, now, class_name))
        
        conn.commit()
        conn.close()

    def log_event(self, event_type, location, description, metadata=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("INSERT INTO events (timestamp, type, location, description, metadata) VALUES (?, ?, ?, ?, ?)",
                  (time.time(), event_type, location, description, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()

    def get_recent_events(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
