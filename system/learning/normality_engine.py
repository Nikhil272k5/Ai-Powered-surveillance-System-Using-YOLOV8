"""
LAYER 3: NORMALITY & BEHAVIOR LEARNING
Unsupervised learning of what is "normal" in the scene.
Uses online statistical updating (Welford's algorithm) to build
distributions of features like speed, dwell time, and size.
"""
import numpy as np
import pickle
import os

class NormalityEngine:
    def __init__(self, model_path='normality_model.pkl'):
        print("🎓 Initializing Learning Layer...")
        self.model_path = model_path
        self.stats = {
            'speed': {'mean': 0.0, 'variance': 0.0, 'n': 0},
            'dwell': {'mean': 0.0, 'variance': 0.0, 'n': 0},
            'size': {'mean': 0.0, 'variance': 0.0, 'n': 0}
        }
        self.load_model()
        
    def update(self, features):
        """
        Update normalcy model with new observation.
        features: dict with keys 'speed', 'dwell', 'size'
        """
        for key, value in features.items():
            if key in self.stats:
                self._update_stat(key, value)
        
        # Periodic save
        if self.stats['speed']['n'] % 100 == 0:
            self.save_model()

    def _update_stat(self, key, x):
        """Online variance update (Welford's Algorithm)"""
        stat = self.stats[key]
        n_prev = stat['n']
        n_new = n_prev + 1
        
        # Mean update
        delta = x - stat['mean']
        new_mean = stat['mean'] + delta / n_new
        
        # Variance update
        delta2 = x - new_mean
        new_var_sum = (stat['variance'] * n_prev) + (delta * delta2)
        new_var = new_var_sum / n_new if n_new > 0 else 0
        
        self.stats[key] = {'mean': new_mean, 'variance': new_var, 'n': n_new}

    def get_anomaly_score(self, features):
        """
        Calculate Z-Score based anomaly.
        Returns: 0.0 (normal) to 1.0 (highly anomalous)
        """
        max_z = 0.0
        
        for key, value in features.items():
            if key in self.stats and self.stats[key]['n'] > 10:
                mean = self.stats[key]['mean']
                std = np.sqrt(self.stats[key]['variance']) + 1e-5
                z = abs(value - mean) / std
                max_z = max(max_z, z)
        
        # Map Z-score to 0-1 probability (Sigmoid-ish)
        # Z=2 -> 0.6, Z=3 -> 0.8, Z=5 -> 0.99
        score = 1 - np.exp(-0.2 * max_z**2)
        return float(score)

    def save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.stats, f)
        except:
            pass

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.stats = pickle.load(f)
            except:
                pass
