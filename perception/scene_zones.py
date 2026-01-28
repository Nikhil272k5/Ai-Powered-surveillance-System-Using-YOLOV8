"""
AbnoGuard Perception - Scene Zone Analyzer
Defines and analyzes semantic zones in the scene (entry, exit, restricted, waiting)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class ZoneType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    WAITING = "waiting"
    RESTRICTED = "restricted"
    EMERGENCY = "emergency"
    TRANSIT = "transit"
    UNKNOWN = "unknown"


@dataclass
class Zone:
    """Represents a semantic zone in the scene"""
    zone_id: str
    zone_type: ZoneType
    polygon: List[Tuple[int, int]]  # List of (x, y) points
    name: str = ""
    rules: Dict = field(default_factory=dict)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside polygon using ray casting"""
        n = len(self.polygon)
        inside = False
        
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def contains_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box center is in zone"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return self.contains_point(center_x, center_y)
    
    def get_bbox_overlap(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate overlap percentage of bbox with zone"""
        x1, y1, x2, y2 = bbox
        total_points = 0
        inside_points = 0
        
        # Sample points in bbox
        for x in range(x1, x2, 10):
            for y in range(y1, y2, 10):
                total_points += 1
                if self.contains_point(x, y):
                    inside_points += 1
        
        return inside_points / max(1, total_points)


@dataclass
class ZoneEvent:
    """Event related to zone interaction"""
    track_id: int
    zone_id: str
    event_type: str  # enter, exit, dwell, violation
    timestamp: float
    duration: float = 0.0
    details: Dict = field(default_factory=dict)


class SceneZoneAnalyzer:
    """
    Analyzes object interactions with semantic zones.
    Zone-aware detection adjusts rules based on context.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.zones: Dict[str, Zone] = {}
        
        # Track state
        self.track_zones: Dict[int, Dict] = {}  # track_id -> {zone_id: enter_time}
        self.zone_occupancy: Dict[str, List[int]] = {}  # zone_id -> [track_ids]
        
        # Zone-specific rules
        self.default_rules = {
            ZoneType.ENTRY: {'max_dwell': 60, 'alert_on_exit_back': True},
            ZoneType.EXIT: {'max_dwell': 30, 'alert_on_entry_back': True},
            ZoneType.WAITING: {'max_dwell': 300, 'min_expected_dwell': 30},
            ZoneType.RESTRICTED: {'max_dwell': 0, 'immediate_alert': True},
            ZoneType.EMERGENCY: {'monitor_always': True, 'panic_detection': True},
            ZoneType.TRANSIT: {'max_dwell': 120, 'loitering_alert': True}
        }
        
        # Load zones from config if available
        zones_file = Path(self.config.get('zones_file', 'config/zones.yaml'))
        if zones_file.exists():
            self._load_zones(zones_file)
        else:
            self._create_default_zones()
        
        print(f"📍 Scene Zone Analyzer initialized with {len(self.zones)} zones")
    
    def _create_default_zones(self):
        """Create default zones for a typical surveillance scene"""
        # These are placeholder zones that should be configured per camera
        # Using relative coordinates (0-100) that get scaled to frame size
        
        # Entry zone - left side
        self.add_zone(Zone(
            zone_id="entry_main",
            zone_type=ZoneType.ENTRY,
            polygon=[(0, 0), (20, 0), (20, 100), (0, 100)],
            name="Main Entry"
        ))
        
        # Exit zone - right side
        self.add_zone(Zone(
            zone_id="exit_main",
            zone_type=ZoneType.EXIT,
            polygon=[(80, 0), (100, 0), (100, 100), (80, 100)],
            name="Main Exit"
        ))
        
        # Waiting area - center bottom
        self.add_zone(Zone(
            zone_id="waiting_lobby",
            zone_type=ZoneType.WAITING,
            polygon=[(30, 60), (70, 60), (70, 100), (30, 100)],
            name="Lobby Waiting Area"
        ))
        
        # Transit corridor - center
        self.add_zone(Zone(
            zone_id="transit_corridor",
            zone_type=ZoneType.TRANSIT,
            polygon=[(20, 30), (80, 30), (80, 60), (20, 60)],
            name="Main Corridor"
        ))
    
    def _load_zones(self, zones_file: Path):
        """Load zones from YAML configuration"""
        try:
            import yaml
            with open(zones_file, 'r') as f:
                data = yaml.safe_load(f)
            
            for zone_data in data.get('zones', []):
                zone = Zone(
                    zone_id=zone_data['id'],
                    zone_type=ZoneType(zone_data['type']),
                    polygon=[(p['x'], p['y']) for p in zone_data['polygon']],
                    name=zone_data.get('name', ''),
                    rules=zone_data.get('rules', {})
                )
                self.add_zone(zone)
        except Exception as e:
            print(f"⚠️ Error loading zones: {e}")
            self._create_default_zones()
    
    def add_zone(self, zone: Zone):
        """Add a zone"""
        self.zones[zone.zone_id] = zone
        self.zone_occupancy[zone.zone_id] = []
    
    def scale_zones_to_frame(self, frame_width: int, frame_height: int):
        """Scale zone coordinates from percentage to pixel coordinates"""
        for zone in self.zones.values():
            scaled_polygon = []
            for x, y in zone.polygon:
                scaled_x = int(x * frame_width / 100)
                scaled_y = int(y * frame_height / 100)
                scaled_polygon.append((scaled_x, scaled_y))
            zone.polygon = scaled_polygon
    
    def analyze(self, tracked_objects: List[Tuple], current_time: float) -> Dict:
        """
        Analyze zone interactions for all tracked objects
        
        Args:
            tracked_objects: List of (track_id, x1, y1, x2, y2, class, conf)
            current_time: Current timestamp
        
        Returns:
            Dict with zone events, violations, and context adjustments
        """
        events = []
        violations = []
        context_adjustments = {}
        
        current_tracks = set()
        
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, conf = obj
            current_tracks.add(track_id)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            
            # Initialize track if new
            if track_id not in self.track_zones:
                self.track_zones[track_id] = {}
            
            # Check each zone
            for zone_id, zone in self.zones.items():
                in_zone = zone.contains_bbox(bbox)
                was_in_zone = zone_id in self.track_zones[track_id]
                
                if in_zone and not was_in_zone:
                    # Entered zone
                    self.track_zones[track_id][zone_id] = current_time
                    self.zone_occupancy[zone_id].append(track_id)
                    
                    event = ZoneEvent(
                        track_id=track_id,
                        zone_id=zone_id,
                        event_type="enter",
                        timestamp=current_time,
                        details={'zone_type': zone.zone_type.value}
                    )
                    events.append(event)
                    
                    # Check for immediate violations
                    if zone.zone_type == ZoneType.RESTRICTED:
                        violations.append({
                            'track_id': track_id,
                            'zone_id': zone_id,
                            'violation_type': 'restricted_zone_entry',
                            'severity': 'high',
                            'description': f"Track {track_id} entered restricted zone {zone.name}"
                        })
                
                elif not in_zone and was_in_zone:
                    # Exited zone
                    enter_time = self.track_zones[track_id].pop(zone_id)
                    dwell_time = current_time - enter_time
                    
                    if track_id in self.zone_occupancy[zone_id]:
                        self.zone_occupancy[zone_id].remove(track_id)
                    
                    event = ZoneEvent(
                        track_id=track_id,
                        zone_id=zone_id,
                        event_type="exit",
                        timestamp=current_time,
                        duration=dwell_time
                    )
                    events.append(event)
                
                elif in_zone and was_in_zone:
                    # Still in zone - check dwell time
                    enter_time = self.track_zones[track_id][zone_id]
                    dwell_time = current_time - enter_time
                    
                    rules = zone.rules or self.default_rules.get(zone.zone_type, {})
                    max_dwell = rules.get('max_dwell', 300)
                    
                    if max_dwell > 0 and dwell_time > max_dwell:
                        violations.append({
                            'track_id': track_id,
                            'zone_id': zone_id,
                            'violation_type': 'excessive_dwell',
                            'severity': 'medium',
                            'dwell_time': dwell_time,
                            'description': f"Track {track_id} in {zone.name} for {dwell_time:.0f}s (max: {max_dwell}s)"
                        })
                
                # Generate context adjustments
                if in_zone:
                    context_adjustments[track_id] = self._get_context_adjustments(zone, class_name)
        
        # Clean up tracks that disappeared
        for track_id in list(self.track_zones.keys()):
            if track_id not in current_tracks:
                for zone_id in list(self.track_zones[track_id].keys()):
                    if track_id in self.zone_occupancy.get(zone_id, []):
                        self.zone_occupancy[zone_id].remove(track_id)
                del self.track_zones[track_id]
        
        return {
            'events': events,
            'violations': violations,
            'context_adjustments': context_adjustments,
            'zone_occupancy': {z: len(tracks) for z, tracks in self.zone_occupancy.items()}
        }
    
    def _get_context_adjustments(self, zone: Zone, class_name: str) -> Dict:
        """Get detection adjustments based on zone context"""
        adjustments = {
            'alert_threshold_modifier': 1.0,
            'expected_behavior': 'normal',
            'loitering_tolerance': 60
        }
        
        if zone.zone_type == ZoneType.WAITING:
            adjustments['loitering_tolerance'] = 300  # 5 min OK in waiting area
            adjustments['expected_behavior'] = 'stationary'
        
        elif zone.zone_type == ZoneType.TRANSIT:
            adjustments['loitering_tolerance'] = 60
            adjustments['expected_behavior'] = 'moving'
        
        elif zone.zone_type == ZoneType.RESTRICTED:
            adjustments['alert_threshold_modifier'] = 0.5  # Lower threshold = more sensitive
            adjustments['expected_behavior'] = 'none'
        
        elif zone.zone_type == ZoneType.ENTRY or zone.zone_type == ZoneType.EXIT:
            adjustments['expected_behavior'] = 'brief'
            adjustments['loitering_tolerance'] = 30
        
        return adjustments
    
    def get_zone_for_point(self, x: int, y: int) -> Optional[Zone]:
        """Get the zone containing a point"""
        for zone in self.zones.values():
            if zone.contains_point(x, y):
                return zone
        return None
    
    def get_zone_summary(self) -> Dict:
        """Get summary of all zones and their occupancy"""
        return {
            'zones': [
                {
                    'id': z.zone_id,
                    'name': z.name,
                    'type': z.zone_type.value,
                    'occupancy': len(self.zone_occupancy.get(z.zone_id, []))
                }
                for z in self.zones.values()
            ],
            'total_zones': len(self.zones)
        }
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones on frame for visualization"""
        import cv2
        
        overlay = frame.copy()
        
        zone_colors = {
            ZoneType.ENTRY: (0, 255, 0),      # Green
            ZoneType.EXIT: (0, 0, 255),        # Red
            ZoneType.WAITING: (255, 255, 0),   # Cyan
            ZoneType.RESTRICTED: (0, 0, 255),  # Red
            ZoneType.EMERGENCY: (0, 165, 255), # Orange
            ZoneType.TRANSIT: (255, 0, 255),   # Magenta
            ZoneType.UNKNOWN: (128, 128, 128)  # Gray
        }
        
        for zone in self.zones.values():
            color = zone_colors.get(zone.zone_type, (128, 128, 128))
            pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
            
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [pts], color)
            
            # Draw border
            cv2.polylines(frame, [pts], True, color, 2)
            
            # Draw label
            if zone.polygon:
                label_x = min(p[0] for p in zone.polygon) + 5
                label_y = min(p[1] for p in zone.polygon) + 20
                cv2.putText(frame, f"{zone.name} ({zone.zone_type.value})",
                           (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Blend overlay
        alpha = 0.2
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
