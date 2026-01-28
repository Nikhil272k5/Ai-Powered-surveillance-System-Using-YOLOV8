"""
AbnoGuard Perception - Optical Flow Analyzer
Detects motion patterns, speed, and direction using optical flow
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MotionPattern(Enum):
    STATIC = "static"
    WALKING = "walking"
    RUNNING = "running"
    ERRATIC = "erratic"
    COUNTERFLOW = "counterflow"
    SUDDEN_STOP = "sudden_stop"
    SUDDEN_START = "sudden_start"


@dataclass
class MotionVector:
    """Represents motion at a point"""
    x: float
    y: float
    magnitude: float
    angle: float
    
    @property
    def speed(self) -> float:
        return self.magnitude
    
    @property
    def direction(self) -> str:
        if self.angle < 45 or self.angle >= 315:
            return "right"
        elif self.angle < 135:
            return "down"
        elif self.angle < 225:
            return "left"
        else:
            return "up"


@dataclass
class RegionMotion:
    """Motion analysis for a region"""
    region_id: int
    avg_magnitude: float
    avg_angle: float
    motion_pattern: MotionPattern
    flow_consistency: float  # 0-1, how consistent is the flow direction
    anomaly_score: float


class OpticalFlowAnalyzer:
    """
    Analyzes motion using optical flow for behavior understanding.
    Provides motion vectors, speed estimation, and pattern detection.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Optical flow parameters
        self.pyr_scale = self.config.get('pyr_scale', 0.5)
        self.levels = self.config.get('levels', 3)
        self.winsize = self.config.get('winsize', 15)
        self.iterations = self.config.get('iterations', 3)
        self.poly_n = self.config.get('poly_n', 5)
        self.poly_sigma = self.config.get('poly_sigma', 1.2)
        
        # Motion thresholds
        self.static_threshold = self.config.get('static_threshold', 1.0)
        self.walking_threshold = self.config.get('walking_threshold', 5.0)
        self.running_threshold = self.config.get('running_threshold', 15.0)
        
        # State
        self.prev_gray = None
        self.motion_history = []
        self.global_flow_direction = None
        
        print("🔄 Optical Flow Analyzer initialized")
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """
        Analyze motion in frame using optical flow
        
        Returns:
            Dict with flow field, motion vectors, patterns, and anomalies
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                'flow': None,
                'motion_vectors': [],
                'global_motion': None,
                'regions': [],
                'anomalies': []
            }
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0
        )
        
        # Analyze flow
        motion_vectors = self._extract_motion_vectors(flow)
        global_motion = self._compute_global_motion(flow)
        regions = self._analyze_regions(flow, frame.shape)
        anomalies = self._detect_anomalies(regions, global_motion)
        
        # Update history
        self.motion_history.append(global_motion)
        if len(self.motion_history) > 30:
            self.motion_history.pop(0)
        
        self.prev_gray = gray
        
        return {
            'flow': flow,
            'motion_vectors': motion_vectors,
            'global_motion': global_motion,
            'regions': regions,
            'anomalies': anomalies
        }
    
    def _extract_motion_vectors(self, flow: np.ndarray, sample_step: int = 20) -> List[MotionVector]:
        """Extract motion vectors at sampled grid points"""
        vectors = []
        h, w = flow.shape[:2]
        
        for y in range(0, h, sample_step):
            for x in range(0, w, sample_step):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)
                angle = np.degrees(np.arctan2(fy, fx)) % 360
                
                if magnitude > self.static_threshold:
                    vectors.append(MotionVector(
                        x=float(x), y=float(y),
                        magnitude=float(magnitude),
                        angle=float(angle)
                    ))
        
        return vectors
    
    def _compute_global_motion(self, flow: np.ndarray) -> Dict:
        """Compute global motion statistics"""
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.degrees(np.arctan2(flow[..., 1], flow[..., 0])) % 360
        
        # Filter out static regions
        active_mask = magnitude > self.static_threshold
        
        if np.sum(active_mask) < 100:
            return {
                'avg_magnitude': 0,
                'avg_angle': 0,
                'motion_coverage': 0,
                'pattern': MotionPattern.STATIC
            }
        
        avg_magnitude = np.mean(magnitude[active_mask])
        
        # Compute dominant direction using histogram
        angle_hist, _ = np.histogram(angle[active_mask], bins=8, range=(0, 360))
        dominant_angle_idx = np.argmax(angle_hist)
        dominant_angle = dominant_angle_idx * 45 + 22.5
        
        # Determine pattern
        if avg_magnitude < self.static_threshold:
            pattern = MotionPattern.STATIC
        elif avg_magnitude < self.walking_threshold:
            pattern = MotionPattern.WALKING
        elif avg_magnitude < self.running_threshold:
            pattern = MotionPattern.RUNNING
        else:
            pattern = MotionPattern.ERRATIC
        
        # Motion coverage (what % of frame is moving)
        motion_coverage = np.sum(active_mask) / magnitude.size
        
        return {
            'avg_magnitude': float(avg_magnitude),
            'avg_angle': float(dominant_angle),
            'motion_coverage': float(motion_coverage),
            'pattern': pattern
        }
    
    def _analyze_regions(self, flow: np.ndarray, frame_shape: Tuple) -> List[RegionMotion]:
        """Analyze motion in grid regions"""
        regions = []
        h, w = flow.shape[:2]
        
        # Divide into 3x3 grid
        grid_h, grid_w = h // 3, w // 3
        
        for i in range(3):
            for j in range(3):
                region_id = i * 3 + j
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                
                region_flow = flow[y1:y2, x1:x2]
                magnitude = np.sqrt(region_flow[..., 0]**2 + region_flow[..., 1]**2)
                angle = np.degrees(np.arctan2(region_flow[..., 1], region_flow[..., 0])) % 360
                
                active_mask = magnitude > self.static_threshold
                
                if np.sum(active_mask) < 50:
                    avg_mag = 0
                    avg_ang = 0
                    consistency = 1.0
                    pattern = MotionPattern.STATIC
                else:
                    avg_mag = np.mean(magnitude[active_mask])
                    avg_ang = np.mean(angle[active_mask])
                    
                    # Flow consistency - how uniform is the direction
                    angle_std = np.std(angle[active_mask])
                    consistency = max(0, 1 - angle_std / 180)
                    
                    if avg_mag < self.walking_threshold:
                        pattern = MotionPattern.WALKING
                    elif avg_mag < self.running_threshold:
                        pattern = MotionPattern.RUNNING
                    else:
                        pattern = MotionPattern.ERRATIC
                
                regions.append(RegionMotion(
                    region_id=region_id,
                    avg_magnitude=float(avg_mag),
                    avg_angle=float(avg_ang),
                    motion_pattern=pattern,
                    flow_consistency=float(consistency),
                    anomaly_score=0.0
                ))
        
        return regions
    
    def _detect_anomalies(self, regions: List[RegionMotion], global_motion: Dict) -> List[Dict]:
        """Detect motion anomalies"""
        anomalies = []
        
        if not regions or global_motion['pattern'] == MotionPattern.STATIC:
            return anomalies
        
        global_angle = global_motion['avg_angle']
        
        for region in regions:
            if region.motion_pattern == MotionPattern.STATIC:
                continue
            
            # Check for counterflow
            angle_diff = abs(region.avg_angle - global_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > 120:  # Moving opposite to global flow
                anomalies.append({
                    'type': 'counterflow',
                    'region_id': region.region_id,
                    'severity': min(1.0, angle_diff / 180),
                    'description': f"Region {region.region_id} moving against global flow"
                })
                region.anomaly_score = min(1.0, angle_diff / 180)
            
            # Check for erratic motion
            if region.flow_consistency < 0.3:
                anomalies.append({
                    'type': 'erratic_motion',
                    'region_id': region.region_id,
                    'severity': 1 - region.flow_consistency,
                    'description': f"Erratic motion detected in region {region.region_id}"
                })
        
        return anomalies
    
    def get_speed_for_bbox(self, flow: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Get average motion speed within a bounding box"""
        if flow is None:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        h, w = flow.shape[:2]
        
        # Clamp to frame bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region_flow = flow[y1:y2, x1:x2]
        magnitude = np.sqrt(region_flow[..., 0]**2 + region_flow[..., 1]**2)
        
        return float(np.mean(magnitude))
    
    def visualize_flow(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Visualize optical flow on frame"""
        if flow is None:
            return frame
        
        vis = frame.copy()
        
        # Draw flow vectors
        step = 20
        h, w = flow.shape[:2]
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)
                
                if magnitude > self.static_threshold:
                    end_x = int(x + fx * 2)
                    end_y = int(y + fy * 2)
                    
                    # Color by magnitude
                    intensity = min(255, int(magnitude * 20))
                    color = (0, 255 - intensity, intensity)
                    
                    cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return vis
    
    def reset(self):
        """Reset analyzer state"""
        self.prev_gray = None
        self.motion_history = []
