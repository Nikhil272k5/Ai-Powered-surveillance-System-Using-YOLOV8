"""
Self-Learning Normality Engine
Learns what "normal" looks like for each camera/context without labeled data.

This module observes:
- Movement speed distributions
- Crowd density trends
- Dwell time patterns
- Object presence durations
- Spatial occupancy heatmaps

Uses unsupervised learning (GMM, clustering, density estimation) to build
adaptive baselines that evolve over time.
"""

import numpy as np
import time
import json
import pickle
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import DBSCAN
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Normality Engine will use basic statistics.")


@dataclass
class NormalityObservation:
    """Single observation of scene state"""
    timestamp: float
    motion_speeds: List[float]  # Speeds of all tracked objects
    crowd_density: int  # Number of people in frame
    dwell_times: Dict[int, float]  # track_id -> time in scene
    object_counts: Dict[str, int]  # class_name -> count
    spatial_positions: List[Tuple[float, float]]  # Center positions of objects


@dataclass
class NormalityBaseline:
    """Learned baseline for a specific context"""
    context_id: str
    created_at: float
    last_updated: float
    observation_count: int = 0
    
    # Speed statistics
    speed_mean: float = 0.0
    speed_std: float = 1.0
    speed_distribution: Optional[Any] = None  # GMM model
    
    # Crowd density statistics
    density_mean: float = 0.0
    density_std: float = 1.0
    density_percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Dwell time statistics
    dwell_mean: float = 0.0
    dwell_std: float = 1.0
    dwell_threshold: float = 30.0  # Seconds
    
    # Spatial distribution (heatmap)
    spatial_heatmap: Optional[np.ndarray] = None
    spatial_grid_size: Tuple[int, int] = (10, 10)
    
    # Object presence patterns
    object_presence_mean: Dict[str, float] = field(default_factory=dict)
    object_presence_std: Dict[str, float] = field(default_factory=dict)


class NormalityEngine:
    """
    Self-Learning Normality Engine
    
    Learns what is "normal" for each camera/video context without human labels.
    Detects anomalies as deviations from the learned baseline.
    """
    
    def __init__(self, 
                 context_id: str = "default",
                 learning_window_minutes: int = 30,
                 adaptation_rate: float = 0.1,
                 anomaly_threshold: float = 2.5,
                 min_observations: int = 100,
                 profiles_dir: str = "profiles"):
        """
        Initialize the Normality Engine.
        
        Args:
            context_id: Unique identifier for this camera/video context
            learning_window_minutes: Rolling window for learning baselines
            adaptation_rate: How quickly to adapt to new patterns (0.0-1.0)
            anomaly_threshold: Standard deviations for anomaly detection
            min_observations: Minimum observations before baseline is trusted
            profiles_dir: Directory to store learned profiles
        """
        self.context_id = context_id
        self.learning_window_minutes = learning_window_minutes
        self.adaptation_rate = adaptation_rate
        self.anomaly_threshold = anomaly_threshold
        self.min_observations = min_observations
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Observation buffer (rolling window)
        self.observations: deque = deque(maxlen=10000)
        
        # Current baseline
        self.baseline = NormalityBaseline(
            context_id=context_id,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        # Load existing profile if available
        self._load_profile()
        
        # Runtime stats
        self.anomalies_detected = 0
        self.last_anomaly_check = time.time()
        
        # Feature buffers for incremental learning
        self._speed_buffer: deque = deque(maxlen=1000)
        self._density_buffer: deque = deque(maxlen=1000)
        self._dwell_buffer: deque = deque(maxlen=1000)
        self._position_buffer: deque = deque(maxlen=5000)
        
        print(f"🧠 Normality Engine initialized for context: {context_id}")
        print(f"   Learning window: {learning_window_minutes} minutes")
        print(f"   Anomaly threshold: {anomaly_threshold} std devs")
    
    def observe(self, tracked_objects: List, track_history: Dict) -> None:
        """
        Add a new observation from the current frame.
        
        Args:
            tracked_objects: List of [track_id, x1, y1, x2, y2, class_name, confidence]
            track_history: Dictionary of track_id -> track info with timestamps
        """
        current_time = time.time()
        
        # Extract features from current frame
        motion_speeds = []
        spatial_positions = []
        dwell_times = {}
        object_counts = defaultdict(int)
        
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            # Calculate center position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            spatial_positions.append((center_x, center_y))
            
            # Count object types
            object_counts[class_name] += 1
            
            # Get track info for dwell time and speed
            if track_id in track_history:
                track_info = track_history[track_id]
                
                # Dwell time
                if 'start_time' in track_info:
                    dwell_time = current_time - track_info['start_time']
                    dwell_times[track_id] = dwell_time
                    self._dwell_buffer.append(dwell_time)
                
                # Speed (if we have previous position)
                if 'bbox' in track_info:
                    prev_bbox = track_info['bbox']
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    
                    distance = np.sqrt(
                        (center_x - prev_center_x) ** 2 + 
                        (center_y - prev_center_y) ** 2
                    )
                    motion_speeds.append(distance)
                    self._speed_buffer.append(distance)
        
        # Crowd density (number of people)
        crowd_density = object_counts.get('person', 0)
        self._density_buffer.append(crowd_density)
        
        # Store positions for spatial distribution
        for pos in spatial_positions:
            self._position_buffer.append(pos)
        
        # Create observation
        observation = NormalityObservation(
            timestamp=current_time,
            motion_speeds=motion_speeds,
            crowd_density=crowd_density,
            dwell_times=dwell_times,
            object_counts=dict(object_counts),
            spatial_positions=spatial_positions
        )
        
        self.observations.append(observation)
        self.baseline.observation_count += 1
        
        # Update baseline periodically
        if len(self.observations) % 100 == 0:
            self._update_baseline()
    
    def _update_baseline(self) -> None:
        """Update the learned baseline with recent observations"""
        if len(self._speed_buffer) < 10:
            return
        
        current_time = time.time()
        
        # Update speed statistics
        if self._speed_buffer:
            speeds = np.array(list(self._speed_buffer))
            new_mean = np.mean(speeds)
            new_std = max(np.std(speeds), 0.1)  # Prevent zero std
            
            # Exponential moving average update
            self.baseline.speed_mean = (
                (1 - self.adaptation_rate) * self.baseline.speed_mean +
                self.adaptation_rate * new_mean
            )
            self.baseline.speed_std = (
                (1 - self.adaptation_rate) * self.baseline.speed_std +
                self.adaptation_rate * new_std
            )
            
            # Fit GMM if sklearn available
            if SKLEARN_AVAILABLE and len(speeds) >= 50:
                try:
                    gmm = GaussianMixture(n_components=min(3, len(speeds) // 20), 
                                          random_state=42)
                    gmm.fit(speeds.reshape(-1, 1))
                    self.baseline.speed_distribution = gmm
                except Exception:
                    pass
        
        # Update density statistics
        if self._density_buffer:
            densities = np.array(list(self._density_buffer))
            new_mean = np.mean(densities)
            new_std = max(np.std(densities), 0.1)
            
            self.baseline.density_mean = (
                (1 - self.adaptation_rate) * self.baseline.density_mean +
                self.adaptation_rate * new_mean
            )
            self.baseline.density_std = (
                (1 - self.adaptation_rate) * self.baseline.density_std +
                self.adaptation_rate * new_std
            )
            
            # Calculate percentiles
            self.baseline.density_percentiles = {
                'p50': float(np.percentile(densities, 50)),
                'p75': float(np.percentile(densities, 75)),
                'p90': float(np.percentile(densities, 90)),
                'p95': float(np.percentile(densities, 95))
            }
        
        # Update dwell time statistics
        if self._dwell_buffer:
            dwells = np.array(list(self._dwell_buffer))
            new_mean = np.mean(dwells)
            new_std = max(np.std(dwells), 0.1)
            
            self.baseline.dwell_mean = (
                (1 - self.adaptation_rate) * self.baseline.dwell_mean +
                self.adaptation_rate * new_mean
            )
            self.baseline.dwell_std = (
                (1 - self.adaptation_rate) * self.baseline.dwell_std +
                self.adaptation_rate * new_std
            )
            
            # Dynamic dwell threshold (95th percentile)
            self.baseline.dwell_threshold = float(np.percentile(dwells, 95))
        
        # Update spatial heatmap
        if self._position_buffer and len(self._position_buffer) >= 100:
            self._update_spatial_heatmap()
        
        self.baseline.last_updated = current_time
    
    def _update_spatial_heatmap(self) -> None:
        """Update the spatial distribution heatmap"""
        positions = np.array(list(self._position_buffer))
        
        if len(positions) < 10:
            return
        
        # Create or update heatmap
        grid_h, grid_w = self.baseline.spatial_grid_size
        
        # Normalize positions to grid (assuming 0-1920 width, 0-1080 height)
        # This will be scaled based on actual frame dimensions
        max_x = max(positions[:, 0].max(), 1920)
        max_y = max(positions[:, 1].max(), 1080)
        
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        for x, y in positions:
            grid_x = min(int(x / max_x * grid_w), grid_w - 1)
            grid_y = min(int(y / max_y * grid_h), grid_h - 1)
            heatmap[grid_y, grid_x] += 1
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Blend with existing heatmap
        if self.baseline.spatial_heatmap is not None:
            self.baseline.spatial_heatmap = (
                (1 - self.adaptation_rate) * self.baseline.spatial_heatmap +
                self.adaptation_rate * heatmap
            )
        else:
            self.baseline.spatial_heatmap = heatmap
    
    def check_anomaly(self, tracked_objects: List, track_history: Dict) -> Dict[str, Any]:
        """
        Check if the current scene state is anomalous.
        
        Returns:
            Dictionary with anomaly results:
            - is_anomaly: bool
            - anomaly_score: float (0-1, higher = more anomalous)
            - anomaly_types: List of detected anomaly types
            - explanation: Human-readable explanation
            - details: Detailed breakdown by feature
        """
        if self.baseline.observation_count < self.min_observations:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'anomaly_types': [],
                'explanation': 'Insufficient observations for anomaly detection',
                'details': {'learning_progress': self.baseline.observation_count / self.min_observations}
            }
        
        anomaly_scores = {}
        anomaly_types = []
        explanations = []
        
        # Extract current state
        motion_speeds = []
        spatial_positions = []
        dwell_times = {}
        object_counts = defaultdict(int)
        
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            spatial_positions.append((center_x, center_y))
            object_counts[class_name] += 1
            
            if track_id in track_history:
                track_info = track_history[track_id]
                if 'start_time' in track_info:
                    dwell_times[track_id] = time.time() - track_info['start_time']
                if 'bbox' in track_info:
                    prev = track_info['bbox']
                    distance = np.sqrt(
                        (center_x - (prev[0] + prev[2]) / 2) ** 2 +
                        (center_y - (prev[1] + prev[3]) / 2) ** 2
                    )
                    motion_speeds.append(distance)
        
        crowd_density = object_counts.get('person', 0)
        
        # Check speed anomaly
        if motion_speeds and self.baseline.speed_std > 0:
            max_speed = max(motion_speeds)
            speed_z = (max_speed - self.baseline.speed_mean) / self.baseline.speed_std
            anomaly_scores['speed'] = min(abs(speed_z) / self.anomaly_threshold, 1.0)
            
            if abs(speed_z) > self.anomaly_threshold:
                anomaly_types.append('abnormal_speed')
                if speed_z > 0:
                    explanations.append(
                        f"Unusually fast movement detected ({max_speed:.1f} vs normal {self.baseline.speed_mean:.1f})"
                    )
                else:
                    explanations.append(
                        f"Unusually slow movement detected ({max_speed:.1f} vs normal {self.baseline.speed_mean:.1f})"
                    )
        
        # Check crowd density anomaly
        if self.baseline.density_std > 0:
            density_z = (crowd_density - self.baseline.density_mean) / self.baseline.density_std
            anomaly_scores['density'] = min(abs(density_z) / self.anomaly_threshold, 1.0)
            
            if abs(density_z) > self.anomaly_threshold:
                anomaly_types.append('abnormal_crowd_density')
                if density_z > 0:
                    explanations.append(
                        f"Unusually high crowd density ({crowd_density} vs normal {self.baseline.density_mean:.1f})"
                    )
                else:
                    explanations.append(
                        f"Unusually low crowd density ({crowd_density} vs normal {self.baseline.density_mean:.1f})"
                    )
        
        # Check dwell time anomaly
        if dwell_times and self.baseline.dwell_std > 0:
            max_dwell = max(dwell_times.values())
            dwell_z = (max_dwell - self.baseline.dwell_mean) / self.baseline.dwell_std
            anomaly_scores['dwell_time'] = min(abs(dwell_z) / self.anomaly_threshold, 1.0)
            
            if max_dwell > self.baseline.dwell_threshold:
                anomaly_types.append('extended_dwell')
                explanations.append(
                    f"Extended presence detected ({max_dwell:.1f}s vs typical {self.baseline.dwell_mean:.1f}s)"
                )
        
        # Check spatial anomaly (unusual location)
        if spatial_positions and self.baseline.spatial_heatmap is not None:
            spatial_score = self._check_spatial_anomaly(spatial_positions)
            anomaly_scores['spatial'] = spatial_score
            
            if spatial_score > 0.8:
                anomaly_types.append('unusual_location')
                explanations.append("Activity detected in unusual location")
        
        # Calculate overall anomaly score
        if anomaly_scores:
            overall_score = max(anomaly_scores.values())
        else:
            overall_score = 0.0
        
        is_anomaly = len(anomaly_types) > 0
        
        if is_anomaly:
            self.anomalies_detected += 1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': overall_score,
            'anomaly_types': anomaly_types,
            'explanation': ' | '.join(explanations) if explanations else 'Normal scene state',
            'details': {
                'scores': anomaly_scores,
                'baseline': {
                    'speed_mean': self.baseline.speed_mean,
                    'speed_std': self.baseline.speed_std,
                    'density_mean': self.baseline.density_mean,
                    'density_std': self.baseline.density_std,
                    'dwell_threshold': self.baseline.dwell_threshold
                },
                'current': {
                    'max_speed': max(motion_speeds) if motion_speeds else 0,
                    'crowd_density': crowd_density,
                    'max_dwell': max(dwell_times.values()) if dwell_times else 0
                }
            }
        }
    
    def _check_spatial_anomaly(self, positions: List[Tuple[float, float]]) -> float:
        """Check if positions are in unusual locations"""
        if self.baseline.spatial_heatmap is None:
            return 0.0
        
        grid_h, grid_w = self.baseline.spatial_grid_size
        heatmap = self.baseline.spatial_heatmap
        
        anomaly_scores = []
        max_x = 1920  # Assume HD resolution
        max_y = 1080
        
        for x, y in positions:
            grid_x = min(int(x / max_x * grid_w), grid_w - 1)
            grid_y = min(int(y / max_y * grid_h), grid_h - 1)
            
            # Low heatmap value = unusual location
            location_score = 1.0 - heatmap[grid_y, grid_x]
            anomaly_scores.append(location_score)
        
        return np.mean(anomaly_scores) if anomaly_scores else 0.0
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get information about learning progress"""
        return {
            'context_id': self.context_id,
            'observation_count': self.baseline.observation_count,
            'min_required': self.min_observations,
            'is_baseline_ready': self.baseline.observation_count >= self.min_observations,
            'progress_percentage': min(100, (self.baseline.observation_count / self.min_observations) * 100),
            'anomalies_detected': self.anomalies_detected,
            'baseline_stats': {
                'speed_mean': self.baseline.speed_mean,
                'speed_std': self.baseline.speed_std,
                'density_mean': self.baseline.density_mean,
                'density_std': self.baseline.density_std,
                'dwell_threshold': self.baseline.dwell_threshold
            },
            'last_updated': self.baseline.last_updated
        }
    
    def save_profile(self) -> None:
        """Save learned profile to disk"""
        profile_path = self.profiles_dir / f"{self.context_id}_profile.pkl"
        
        try:
            # Prepare data for serialization
            save_data = {
                'context_id': self.context_id,
                'baseline': {
                    'created_at': self.baseline.created_at,
                    'last_updated': self.baseline.last_updated,
                    'observation_count': self.baseline.observation_count,
                    'speed_mean': self.baseline.speed_mean,
                    'speed_std': self.baseline.speed_std,
                    'density_mean': self.baseline.density_mean,
                    'density_std': self.baseline.density_std,
                    'density_percentiles': self.baseline.density_percentiles,
                    'dwell_mean': self.baseline.dwell_mean,
                    'dwell_std': self.baseline.dwell_std,
                    'dwell_threshold': self.baseline.dwell_threshold,
                    'spatial_heatmap': self.baseline.spatial_heatmap,
                    'spatial_grid_size': self.baseline.spatial_grid_size,
                    'object_presence_mean': self.baseline.object_presence_mean,
                    'object_presence_std': self.baseline.object_presence_std
                },
                'anomalies_detected': self.anomalies_detected
            }
            
            with open(profile_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"✅ Normality profile saved: {profile_path}")
            
        except Exception as e:
            print(f"❌ Error saving normality profile: {e}")
    
    def _load_profile(self) -> None:
        """Load existing profile from disk"""
        profile_path = self.profiles_dir / f"{self.context_id}_profile.pkl"
        
        if not profile_path.exists():
            return
        
        try:
            with open(profile_path, 'rb') as f:
                save_data = pickle.load(f)
            
            baseline_data = save_data.get('baseline', {})
            self.baseline = NormalityBaseline(
                context_id=self.context_id,
                created_at=baseline_data.get('created_at', time.time()),
                last_updated=baseline_data.get('last_updated', time.time()),
                observation_count=baseline_data.get('observation_count', 0),
                speed_mean=baseline_data.get('speed_mean', 0.0),
                speed_std=baseline_data.get('speed_std', 1.0),
                density_mean=baseline_data.get('density_mean', 0.0),
                density_std=baseline_data.get('density_std', 1.0),
                density_percentiles=baseline_data.get('density_percentiles', {}),
                dwell_mean=baseline_data.get('dwell_mean', 0.0),
                dwell_std=baseline_data.get('dwell_std', 1.0),
                dwell_threshold=baseline_data.get('dwell_threshold', 30.0),
                spatial_heatmap=baseline_data.get('spatial_heatmap'),
                spatial_grid_size=baseline_data.get('spatial_grid_size', (10, 10)),
                object_presence_mean=baseline_data.get('object_presence_mean', {}),
                object_presence_std=baseline_data.get('object_presence_std', {})
            )
            
            self.anomalies_detected = save_data.get('anomalies_detected', 0)
            
            print(f"✅ Loaded normality profile: {profile_path}")
            print(f"   Observations: {self.baseline.observation_count}")
            
        except Exception as e:
            print(f"⚠️ Error loading normality profile: {e}")
    
    def reset(self) -> None:
        """Reset the engine to initial state"""
        self.observations.clear()
        self._speed_buffer.clear()
        self._density_buffer.clear()
        self._dwell_buffer.clear()
        self._position_buffer.clear()
        
        self.baseline = NormalityBaseline(
            context_id=self.context_id,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.anomalies_detected = 0
        print(f"🔄 Normality Engine reset for context: {self.context_id}")
