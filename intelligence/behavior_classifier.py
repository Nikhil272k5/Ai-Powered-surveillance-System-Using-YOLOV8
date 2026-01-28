"""
Behavior & Intent Inference Layer
Classifies behavior intent, not just objects.

Intent Classes:
- NORMAL_TRANSIT: Walking through area
- WAITING: Stationary, normal context
- LOITERING: Extended presence, no clear purpose
- PANIC_MOVEMENT: Rapid, erratic motion
- EVASIVE_BEHAVIOR: Avoiding detection patterns
- CARELESS_ABANDONMENT: Object left accidentally
- SUSPICIOUS_ABANDONMENT: Deliberate placement

Uses:
- Motion history features (velocity, acceleration, trajectory)
- Lightweight ML models (RandomForest / XGBoost / Isolation Forest)
- Works on top of existing tracking
"""

import numpy as np
import time
import pickle
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available. Behavior Classifier will use rule-based fallback.")


class IntentClass(Enum):
    """Behavior intent classes"""
    NORMAL_TRANSIT = "normal_transit"
    WAITING = "waiting"
    LOITERING = "loitering"
    PANIC_MOVEMENT = "panic_movement"
    EVASIVE_BEHAVIOR = "evasive_behavior"
    CARELESS_ABANDONMENT = "careless_abandonment"
    SUSPICIOUS_ABANDONMENT = "suspicious_abandonment"
    UNKNOWN = "unknown"


@dataclass
class BehaviorClassification:
    """Classification result for a tracked entity"""
    track_id: int
    intent_class: IntentClass
    confidence: float  # 0-1
    features: Dict[str, float]
    explanation: str
    timestamp: float
    secondary_intents: List[Tuple[IntentClass, float]] = field(default_factory=list)


@dataclass
class MotionFeatures:
    """Extracted motion features from track history"""
    # Speed features
    mean_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    speed_variance: float = 0.0
    speed_acceleration: float = 0.0
    
    # Direction features
    mean_direction: float = 0.0
    direction_variance: float = 0.0
    direction_changes: int = 0
    
    # Trajectory features
    path_straightness: float = 0.0
    area_covered: float = 0.0
    net_displacement: float = 0.0
    total_distance: float = 0.0
    
    # Temporal features
    dwell_time: float = 0.0
    time_stationary: float = 0.0
    time_moving: float = 0.0
    
    # Behavior features
    stops_count: int = 0
    reversals_count: int = 0
    proximity_to_objects: float = 0.0


class BehaviorClassifier:
    """
    Behavior & Intent Inference Layer
    
    Classifies the behavioral intent of tracked entities using motion
    history features and lightweight ML models.
    """
    
    def __init__(self,
                 model_type: str = "random_forest",
                 classification_threshold: float = 0.6,
                 history_length: int = 30,
                 update_interval: int = 5,
                 models_dir: str = "models"):
        """
        Initialize the Behavior Classifier.
        
        Args:
            model_type: ML model type ("random_forest", "isolation_forest")
            classification_threshold: Minimum confidence for classification
            history_length: Number of past frames to consider
            update_interval: Frames between classification updates
            models_dir: Directory for model storage
        """
        self.model_type = model_type
        self.classification_threshold = classification_threshold
        self.history_length = history_length
        self.update_interval = update_interval
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Track histories for feature extraction
        self.track_histories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        
        # Classification cache
        self.classifications: Dict[int, BehaviorClassification] = {}
        
        # Frame counter for update interval
        self.frame_counter = 0
        
        # ML model (if available)
        self.model = None
        self.scaler = None
        self._initialize_model()
        
        # Statistics
        self.total_classifications = 0
        self.classification_counts: Dict[str, int] = defaultdict(int)
        
        print(f"🎭 Behavior Classifier initialized")
        print(f"   Model type: {model_type}")
        print(f"   Classification threshold: {classification_threshold}")
        print(f"   History length: {history_length} frames")
    
    def _initialize_model(self) -> None:
        """Initialize or load the ML model"""
        if not SKLEARN_AVAILABLE:
            print("   Using rule-based classification (sklearn not available)")
            return
        
        model_path = self.models_dir / f"behavior_{self.model_type}.pkl"
        scaler_path = self.models_dir / "behavior_scaler.pkl"
        
        if model_path.exists() and scaler_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"   Loaded trained model: {model_path}")
                return
            except Exception as e:
                print(f"   Error loading model: {e}")
        
        # Initialize new model with synthetic training data
        print("   Initializing model with synthetic training data...")
        self._train_on_synthetic_data()
    
    def _train_on_synthetic_data(self) -> None:
        """Train model on synthetic behavior patterns"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Generate synthetic training data for each behavior class
        X_train = []
        y_train = []
        
        np.random.seed(42)
        n_samples_per_class = 200
        
        # Normal Transit: moderate speed, straight path, low variance
        for _ in range(n_samples_per_class):
            features = self._generate_synthetic_features(
                mean_speed=(15, 5), speed_var=(5, 2),
                direction_var=(0.1, 0.05), path_straightness=(0.8, 0.1),
                dwell_time=(5, 2), direction_changes=(1, 0.5)
            )
            X_train.append(features)
            y_train.append(IntentClass.NORMAL_TRANSIT.value)
        
        # Waiting: very low speed, small area, moderate dwell
        for _ in range(n_samples_per_class):
            features = self._generate_synthetic_features(
                mean_speed=(2, 1), speed_var=(1, 0.5),
                direction_var=(0.5, 0.2), path_straightness=(0.3, 0.2),
                dwell_time=(30, 10), direction_changes=(2, 1)
            )
            X_train.append(features)
            y_train.append(IntentClass.WAITING.value)
        
        # Loitering: low speed, small area, high dwell, some movement
        for _ in range(n_samples_per_class):
            features = self._generate_synthetic_features(
                mean_speed=(5, 2), speed_var=(3, 1),
                direction_var=(0.8, 0.3), path_straightness=(0.2, 0.1),
                dwell_time=(60, 20), direction_changes=(5, 2)
            )
            X_train.append(features)
            y_train.append(IntentClass.LOITERING.value)
        
        # Panic Movement: very high speed, erratic, high variance
        for _ in range(n_samples_per_class):
            features = self._generate_synthetic_features(
                mean_speed=(50, 15), speed_var=(30, 10),
                direction_var=(1.2, 0.3), path_straightness=(0.3, 0.2),
                dwell_time=(3, 1), direction_changes=(8, 3)
            )
            X_train.append(features)
            y_train.append(IntentClass.PANIC_MOVEMENT.value)
        
        # Evasive Behavior: variable speed, sudden direction changes, reversals
        for _ in range(n_samples_per_class):
            features = self._generate_synthetic_features(
                mean_speed=(20, 8), speed_var=(15, 5),
                direction_var=(1.5, 0.4), path_straightness=(0.15, 0.1),
                dwell_time=(10, 5), direction_changes=(10, 4),
                reversals=(3, 1)
            )
            X_train.append(features)
            y_train.append(IntentClass.EVASIVE_BEHAVIOR.value)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Create and train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            # Fallback to random forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        self.model.fit(X_scaled, y_train)
        
        # Save model
        try:
            joblib.dump(self.model, self.models_dir / f"behavior_{self.model_type}.pkl")
            joblib.dump(self.scaler, self.models_dir / "behavior_scaler.pkl")
            print("   Model trained and saved")
        except Exception as e:
            print(f"   Error saving model: {e}")
    
    def _generate_synthetic_features(self, 
                                      mean_speed=(10, 5),
                                      speed_var=(5, 2),
                                      direction_var=(0.5, 0.2),
                                      path_straightness=(0.5, 0.2),
                                      dwell_time=(10, 5),
                                      direction_changes=(2, 1),
                                      reversals=(0, 0)) -> np.ndarray:
        """Generate synthetic feature vector for training"""
        return np.array([
            max(0, np.random.normal(*mean_speed)),  # mean_speed
            max(0, np.random.normal(mean_speed[0] * 1.5, speed_var[0])),  # max_speed
            max(0, np.random.normal(mean_speed[0] * 0.3, 2)),  # min_speed
            max(0, np.random.normal(*speed_var)),  # speed_variance
            np.random.normal(0, 5),  # speed_acceleration
            np.random.normal(0, 1),  # mean_direction
            max(0, np.random.normal(*direction_var)),  # direction_variance
            max(0, int(np.random.normal(*direction_changes))),  # direction_changes
            min(1, max(0, np.random.normal(*path_straightness))),  # path_straightness
            max(0, np.random.normal(1000, 500)),  # area_covered
            max(0, np.random.normal(100, 50)),  # net_displacement
            max(0, np.random.normal(200, 100)),  # total_distance
            max(0, np.random.normal(*dwell_time)),  # dwell_time
            max(0, np.random.normal(dwell_time[0] * 0.3, 2)),  # time_stationary
            max(0, np.random.normal(dwell_time[0] * 0.7, 3)),  # time_moving
            max(0, int(np.random.normal(1, 0.5))),  # stops_count
            max(0, int(np.random.normal(*reversals))),  # reversals_count
        ])
    
    def update(self, tracked_objects: List, track_history: Dict) -> List[BehaviorClassification]:
        """
        Update classifier with new tracking data and return classifications.
        
        Args:
            tracked_objects: List of [track_id, x1, y1, x2, y2, class_name, confidence]
            track_history: Dictionary of track info from tracker
        
        Returns:
            List of BehaviorClassification for updated tracks
        """
        self.frame_counter += 1
        current_time = time.time()
        
        # Update track histories
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            # Only classify people
            if class_name != 'person':
                continue
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            self.track_histories[track_id].append({
                'position': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'timestamp': current_time,
                'class_name': class_name,
                'confidence': confidence
            })
        
        # Classify at update interval
        classifications = []
        if self.frame_counter % self.update_interval == 0:
            for obj in tracked_objects:
                track_id = obj[0]
                class_name = obj[5]
                
                if class_name != 'person':
                    continue
                
                if len(self.track_histories[track_id]) >= 10:
                    classification = self._classify_track(track_id, track_history)
                    if classification:
                        classifications.append(classification)
                        self.classifications[track_id] = classification
        
        return classifications
    
    def _classify_track(self, track_id: int, 
                        track_history: Dict) -> Optional[BehaviorClassification]:
        """Classify the behavior intent of a specific track"""
        history = list(self.track_histories[track_id])
        
        if len(history) < 10:
            return None
        
        # Extract features
        features = self._extract_features(history, track_id, track_history)
        
        # Classify using ML model or rules
        if SKLEARN_AVAILABLE and self.model is not None:
            classification = self._classify_with_ml(track_id, features)
        else:
            classification = self._classify_with_rules(track_id, features)
        
        if classification:
            self.total_classifications += 1
            self.classification_counts[classification.intent_class.value] += 1
        
        return classification
    
    def _extract_features(self, history: List[Dict], track_id: int,
                          track_history: Dict) -> MotionFeatures:
        """Extract motion features from track history"""
        features = MotionFeatures()
        
        if len(history) < 2:
            return features
        
        positions = [h['position'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        # Calculate speeds
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
                speeds.append(speed)
        
        if speeds:
            features.mean_speed = float(np.mean(speeds))
            features.max_speed = float(np.max(speeds))
            features.min_speed = float(np.min(speeds))
            features.speed_variance = float(np.var(speeds))
            
            # Acceleration (change in speed)
            if len(speeds) >= 2:
                accelerations = np.diff(speeds)
                features.speed_acceleration = float(np.mean(np.abs(accelerations)))
        
        # Calculate directions and direction changes
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                direction = np.arctan2(dy, dx)
                directions.append(direction)
        
        if directions:
            features.mean_direction = float(np.mean(directions))
            features.direction_variance = float(np.var(directions))
            
            # Count significant direction changes
            direction_changes = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff > np.pi / 4:  # More than 45 degrees
                    direction_changes += 1
            features.direction_changes = direction_changes
            
            # Count reversals (more than 135 degrees)
            reversals = 0
            for i in range(1, len(directions)):
                angle_diff = abs(directions[i] - directions[i-1])
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff > 3 * np.pi / 4:
                    reversals += 1
            features.reversals_count = reversals
        
        # Path trajectory features
        if len(positions) >= 2:
            # Total distance traveled
            total_dist = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_dist += np.sqrt(dx**2 + dy**2)
            features.total_distance = float(total_dist)
            
            # Net displacement (start to end)
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            features.net_displacement = float(np.sqrt(dx**2 + dy**2))
            
            # Path straightness (net displacement / total distance)
            if total_dist > 0:
                features.path_straightness = min(1.0, features.net_displacement / total_dist)
            
            # Area covered (bounding box of all positions)
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            features.area_covered = float((max(xs) - min(xs)) * (max(ys) - min(ys)))
        
        # Temporal features
        features.dwell_time = float(timestamps[-1] - timestamps[0])
        
        # Time stationary vs moving
        stationary_time = 0
        moving_time = 0
        stationary_threshold = 5  # pixels/second
        
        for i in range(1, len(speeds)):
            dt = timestamps[i] - timestamps[i-1]
            if speeds[i-1] < stationary_threshold:
                stationary_time += dt
            else:
                moving_time += dt
        
        features.time_stationary = float(stationary_time)
        features.time_moving = float(moving_time)
        
        # Stops count (transitions from moving to stationary)
        stops = 0
        was_moving = False
        for speed in speeds:
            is_moving = speed >= stationary_threshold
            if was_moving and not is_moving:
                stops += 1
            was_moving = is_moving
        features.stops_count = stops
        
        return features
    
    def _classify_with_ml(self, track_id: int, 
                          features: MotionFeatures) -> BehaviorClassification:
        """Classify using the trained ML model"""
        # Convert features to array
        feature_array = np.array([
            features.mean_speed,
            features.max_speed,
            features.min_speed,
            features.speed_variance,
            features.speed_acceleration,
            features.mean_direction,
            features.direction_variance,
            features.direction_changes,
            features.path_straightness,
            features.area_covered,
            features.net_displacement,
            features.total_distance,
            features.dwell_time,
            features.time_stationary,
            features.time_moving,
            features.stops_count,
            features.reversals_count
        ]).reshape(1, -1)
        
        # Scale features
        try:
            feature_scaled = self.scaler.transform(feature_array)
        except Exception:
            feature_scaled = feature_array
        
        # Predict
        prediction = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]
        
        # Get confidence
        class_idx = list(self.model.classes_).index(prediction)
        confidence = probabilities[class_idx]
        
        # Get secondary intents
        secondary = []
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices[1:3]:
            if probabilities[idx] > 0.1:
                intent = IntentClass(self.model.classes_[idx])
                secondary.append((intent, float(probabilities[idx])))
        
        # Determine intent class
        try:
            intent_class = IntentClass(prediction)
        except ValueError:
            intent_class = IntentClass.UNKNOWN
        
        # Generate explanation
        explanation = self._generate_explanation(intent_class, features, confidence)
        
        return BehaviorClassification(
            track_id=track_id,
            intent_class=intent_class,
            confidence=float(confidence),
            features=self._features_to_dict(features),
            explanation=explanation,
            timestamp=time.time(),
            secondary_intents=secondary
        )
    
    def _classify_with_rules(self, track_id: int,
                             features: MotionFeatures) -> BehaviorClassification:
        """Rule-based classification fallback"""
        scores = {intent: 0.0 for intent in IntentClass}
        
        # Normal Transit: moderate speed, straight path
        if 10 < features.mean_speed < 30 and features.path_straightness > 0.6:
            scores[IntentClass.NORMAL_TRANSIT] += 0.4
        if features.direction_variance < 0.5:
            scores[IntentClass.NORMAL_TRANSIT] += 0.3
        if features.dwell_time < 20:
            scores[IntentClass.NORMAL_TRANSIT] += 0.3
        
        # Waiting: nearly stationary
        if features.mean_speed < 5:
            scores[IntentClass.WAITING] += 0.4
        if features.area_covered < 500:
            scores[IntentClass.WAITING] += 0.3
        if 10 < features.dwell_time < 60:
            scores[IntentClass.WAITING] += 0.3
        
        # Loitering: slow movement in small area for long time
        if features.mean_speed < 10:
            scores[IntentClass.LOITERING] += 0.2
        if features.area_covered < 2000:
            scores[IntentClass.LOITERING] += 0.2
        if features.dwell_time > 30:
            scores[IntentClass.LOITERING] += 0.3
        if features.direction_changes > 3:
            scores[IntentClass.LOITERING] += 0.3
        
        # Panic Movement: very fast, erratic
        if features.mean_speed > 40:
            scores[IntentClass.PANIC_MOVEMENT] += 0.4
        if features.speed_variance > 20:
            scores[IntentClass.PANIC_MOVEMENT] += 0.3
        if features.direction_variance > 1.0:
            scores[IntentClass.PANIC_MOVEMENT] += 0.3
        
        # Evasive Behavior: reversals, direction changes
        if features.reversals_count >= 2:
            scores[IntentClass.EVASIVE_BEHAVIOR] += 0.4
        if features.direction_changes > 6:
            scores[IntentClass.EVASIVE_BEHAVIOR] += 0.3
        if features.path_straightness < 0.2:
            scores[IntentClass.EVASIVE_BEHAVIOR] += 0.3
        
        # Find best match
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        if confidence < self.classification_threshold:
            best_intent = IntentClass.UNKNOWN
            confidence = 1.0 - max(scores.values())
        
        # Secondary intents
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary = [(intent, score) for intent, score in sorted_intents[1:3] if score > 0.2]
        
        explanation = self._generate_explanation(best_intent, features, confidence)
        
        return BehaviorClassification(
            track_id=track_id,
            intent_class=best_intent,
            confidence=float(confidence),
            features=self._features_to_dict(features),
            explanation=explanation,
            timestamp=time.time(),
            secondary_intents=secondary
        )
    
    def _generate_explanation(self, intent: IntentClass, 
                             features: MotionFeatures, 
                             confidence: float) -> str:
        """Generate human-readable explanation for classification"""
        explanations = {
            IntentClass.NORMAL_TRANSIT: 
                f"Walking through area at {features.mean_speed:.1f} px/s, "
                f"path straightness {features.path_straightness:.0%}",
            IntentClass.WAITING:
                f"Stationary/minimal movement ({features.mean_speed:.1f} px/s) "
                f"for {features.dwell_time:.0f}s",
            IntentClass.LOITERING:
                f"Extended presence ({features.dwell_time:.0f}s) in small area "
                f"({features.area_covered:.0f}px²) with {features.direction_changes} direction changes",
            IntentClass.PANIC_MOVEMENT:
                f"Rapid movement ({features.mean_speed:.1f} px/s, max {features.max_speed:.1f}) "
                f"with high variance ({features.speed_variance:.1f})",
            IntentClass.EVASIVE_BEHAVIOR:
                f"Evasive pattern: {features.reversals_count} reversals, "
                f"{features.direction_changes} direction changes, "
                f"path straightness {features.path_straightness:.0%}",
            IntentClass.CARELESS_ABANDONMENT:
                f"Object left behind during normal transit",
            IntentClass.SUSPICIOUS_ABANDONMENT:
                f"Deliberate object placement detected",
            IntentClass.UNKNOWN:
                f"Behavior pattern unclear (confidence: {confidence:.0%})"
        }
        
        return explanations.get(intent, f"Classified as {intent.value}")
    
    def _features_to_dict(self, features: MotionFeatures) -> Dict[str, float]:
        """Convert MotionFeatures to dictionary"""
        return {
            'mean_speed': features.mean_speed,
            'max_speed': features.max_speed,
            'speed_variance': features.speed_variance,
            'direction_variance': features.direction_variance,
            'direction_changes': features.direction_changes,
            'path_straightness': features.path_straightness,
            'area_covered': features.area_covered,
            'dwell_time': features.dwell_time,
            'reversals_count': features.reversals_count
        }
    
    def get_classification(self, track_id: int) -> Optional[BehaviorClassification]:
        """Get the most recent classification for a track"""
        return self.classifications.get(track_id)
    
    def get_all_classifications(self) -> Dict[int, BehaviorClassification]:
        """Get all current classifications"""
        return self.classifications.copy()
    
    def classify_abandonment(self, track_id: int, object_track_id: int,
                            person_behavior: Optional[BehaviorClassification]) -> IntentClass:
        """
        Classify whether an abandonment is careless or suspicious.
        
        Uses the person's behavior leading up to abandonment.
        """
        if person_behavior is None:
            return IntentClass.CARELESS_ABANDONMENT
        
        # Suspicious indicators:
        # - Evasive behavior before leaving
        # - Looking around (direction changes)
        # - Quick departure after placing
        
        features = person_behavior.features
        
        suspicion_score = 0.0
        
        # Direction changes suggest looking around
        if features.get('direction_changes', 0) > 4:
            suspicion_score += 0.3
        
        # Low path straightness suggests intentional positioning
        if features.get('path_straightness', 1.0) < 0.3:
            suspicion_score += 0.2
        
        # Recent evasive behavior
        if person_behavior.intent_class == IntentClass.EVASIVE_BEHAVIOR:
            suspicion_score += 0.4
        
        # Reversals suggest deliberate action
        if features.get('reversals_count', 0) > 1:
            suspicion_score += 0.2
        
        if suspicion_score > 0.5:
            return IntentClass.SUSPICIOUS_ABANDONMENT
        else:
            return IntentClass.CARELESS_ABANDONMENT
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return {
            'total_classifications': self.total_classifications,
            'classification_counts': dict(self.classification_counts),
            'active_tracks': len(self.classifications),
            'model_type': self.model_type,
            'sklearn_available': SKLEARN_AVAILABLE
        }
    
    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset classifier state"""
        if track_id is not None:
            self.track_histories.pop(track_id, None)
            self.classifications.pop(track_id, None)
        else:
            self.track_histories.clear()
            self.classifications.clear()
            self.frame_counter = 0
