"""
Autonomous Improvement & Self-Adaptation Loop
System evolves without retraining based on feedback and outcomes.

Features:
- Track alert outcomes (acknowledged/dismissed/timed out)
- Measure false positive rates per alert type
- Auto-adjust detection thresholds
- Reweight confidence fusion signals
- Maintain performance history
"""

import time
import json
import pickle
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np


@dataclass
class AlertOutcome:
    """Recorded outcome of an alert"""
    alert_id: str
    alert_type: str
    trust_score: float
    outcome: str  # 'acknowledged', 'dismissed', 'timed_out', 'pending'
    timestamp: float
    response_time: Optional[float] = None  # Time to response
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific period"""
    period_start: float
    period_end: float
    total_alerts: int = 0
    acknowledged: int = 0
    dismissed: int = 0
    timed_out: int = 0
    false_positive_rate: float = 0.0
    avg_trust_score: float = 0.0
    avg_response_time: float = 0.0
    adjustments_made: List[Dict] = field(default_factory=list)


@dataclass
class ThresholdAdjustment:
    """Record of a threshold adjustment"""
    timestamp: float
    parameter: str
    old_value: float
    new_value: float
    reason: str
    metrics_snapshot: Dict[str, float]


class SelfImprovementEngine:
    """
    Autonomous Improvement & Self-Adaptation Loop
    
    Tracks alert outcomes and automatically adjusts system parameters
    to improve reliability over time without human retraining.
    """
    
    def __init__(self,
                 learning_rate: float = 0.05,
                 min_alerts_for_adjustment: int = 20,
                 performance_window_hours: int = 24,
                 auto_dismiss_timeout: int = 300,
                 max_adjustment_per_cycle: float = 0.1,
                 profiles_dir: str = "profiles"):
        """
        Initialize the Self-Improvement Engine.
        
        Args:
            learning_rate: How aggressively to adjust parameters (0-1)
            min_alerts_for_adjustment: Minimum alerts before auto-adjusting
            performance_window_hours: Window for calculating performance
            auto_dismiss_timeout: Seconds before auto-dismissing unacknowledged alert
            max_adjustment_per_cycle: Maximum parameter change per adjustment cycle
            profiles_dir: Directory for storing state
        """
        self.learning_rate = learning_rate
        self.min_alerts_for_adjustment = min_alerts_for_adjustment
        self.performance_window_hours = performance_window_hours
        self.auto_dismiss_timeout = auto_dismiss_timeout
        self.max_adjustment_per_cycle = max_adjustment_per_cycle
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert tracking
        self.pending_alerts: Dict[str, AlertOutcome] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Per-type statistics
        self.type_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'total': 0,
                'acknowledged': 0,
                'dismissed': 0,
                'timed_out': 0,
                'trust_scores': [],
                'response_times': []
            }
        )
        
        # Current thresholds and weights (these will be adjusted)
        self.current_thresholds: Dict[str, float] = {
            'trust_threshold': 60.0,
            'speed_threshold': 30.0,
            'loitering_threshold': 15.0,
            'abandonment_threshold': 5.0,
        }
        
        self.current_weights: Dict[str, float] = {
            'vision_confidence': 0.25,
            'temporal_persistence': 0.20,
            'motion_stability': 0.15,
            'crowd_context': 0.15,
            'zone_rules': 0.10,
            'historical_accuracy': 0.15,
        }
        
        # Adjustment history
        self.adjustments: List[ThresholdAdjustment] = []
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        
        # Last adjustment check time
        self.last_adjustment_time = time.time()
        self.adjustment_check_interval = 300  # 5 minutes
        
        # Load saved state
        self._load_state()
        
        print(f"🔄 Self-Improvement Engine initialized")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Min alerts for adjustment: {min_alerts_for_adjustment}")
        print(f"   Performance window: {performance_window_hours}h")
    
    def register_alert(self, alert_id: str, alert_type: str, 
                      trust_score: float, metadata: Optional[Dict] = None) -> None:
        """
        Register a new alert for tracking.
        
        Args:
            alert_id: Unique alert identifier
            alert_type: Type of alert (e.g., 'speed_spike')
            trust_score: Trust score assigned by confidence fusion
            metadata: Additional alert data
        """
        outcome = AlertOutcome(
            alert_id=alert_id,
            alert_type=alert_type,
            trust_score=trust_score,
            outcome='pending',
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.pending_alerts[alert_id] = outcome
        self.type_stats[alert_type]['total'] += 1
        self.type_stats[alert_type]['trust_scores'].append(trust_score)
    
    def record_feedback(self, alert_id: str, outcome: str) -> bool:
        """
        Record feedback for an alert.
        
        Args:
            alert_id: Alert identifier
            outcome: 'acknowledged' or 'dismissed'
        
        Returns:
            True if feedback recorded, False if alert not found
        """
        if alert_id not in self.pending_alerts:
            return False
        
        alert = self.pending_alerts[alert_id]
        alert.outcome = outcome
        alert.response_time = time.time() - alert.timestamp
        
        # Update type statistics
        stats = self.type_stats[alert.alert_type]
        if outcome == 'acknowledged':
            stats['acknowledged'] += 1
        elif outcome == 'dismissed':
            stats['dismissed'] += 1
        
        if alert.response_time:
            stats['response_times'].append(alert.response_time)
        
        # Move to history
        self.alert_history.append(alert)
        del self.pending_alerts[alert_id]
        
        # Check if we should adjust
        self._check_for_adjustment()
        
        return True
    
    def process_timeouts(self) -> List[str]:
        """
        Process timed-out alerts.
        
        Returns:
            List of timed-out alert IDs
        """
        if self.auto_dismiss_timeout < 0:
            return []
        
        current_time = time.time()
        timed_out = []
        
        for alert_id, alert in list(self.pending_alerts.items()):
            if current_time - alert.timestamp > self.auto_dismiss_timeout:
                alert.outcome = 'timed_out'
                alert.response_time = self.auto_dismiss_timeout
                
                self.type_stats[alert.alert_type]['timed_out'] += 1
                
                self.alert_history.append(alert)
                del self.pending_alerts[alert_id]
                timed_out.append(alert_id)
        
        return timed_out
    
    def _check_for_adjustment(self) -> None:
        """Check if conditions are met for parameter adjustment"""
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.last_adjustment_time < self.adjustment_check_interval:
            return
        
        # Check if we have enough data
        total_recent = sum(
            stats['total'] for stats in self.type_stats.values()
        )
        
        if total_recent < self.min_alerts_for_adjustment:
            return
        
        # Perform adjustment
        self._perform_adjustment()
        self.last_adjustment_time = current_time
    
    def _perform_adjustment(self) -> None:
        """Perform automatic parameter adjustment based on performance"""
        adjustments_made = []
        
        for alert_type, stats in self.type_stats.items():
            if stats['total'] < 5:
                continue
            
            # Calculate false positive rate
            total = stats['total']
            false_positives = stats['dismissed'] + stats['timed_out'] * 0.5
            fpr = false_positives / total if total > 0 else 0
            
            # Calculate true positive rate
            tpr = stats['acknowledged'] / total if total > 0 else 0
            
            # Adjust thresholds based on performance
            threshold_key = f"{alert_type}_threshold"
            
            if fpr > 0.5:
                # Too many false positives - increase threshold
                adjustment = self._adjust_threshold(
                    threshold_key,
                    direction='increase',
                    reason=f"High false positive rate ({fpr:.0%}) for {alert_type}"
                )
                if adjustment:
                    adjustments_made.append(adjustment)
            
            elif fpr < 0.1 and tpr > 0.8:
                # Very good performance - slightly decrease threshold to catch more
                adjustment = self._adjust_threshold(
                    threshold_key,
                    direction='decrease',
                    reason=f"Excellent performance for {alert_type}, relaxing threshold"
                )
                if adjustment:
                    adjustments_made.append(adjustment)
        
        # Adjust signal weights based on correlation with accuracy
        self._adjust_signal_weights()
        
        # Record performance metrics
        if adjustments_made:
            metrics = self._calculate_current_metrics()
            metrics.adjustments_made = [asdict(a) if hasattr(a, '__dict__') else a for a in adjustments_made]
            self.performance_history.append(metrics)
            
            # Save state
            self._save_state()
    
    def _adjust_threshold(self, threshold_key: str, direction: str, 
                         reason: str) -> Optional[ThresholdAdjustment]:
        """Adjust a specific threshold"""
        if threshold_key not in self.current_thresholds:
            # Check trust threshold
            if 'trust' in threshold_key:
                threshold_key = 'trust_threshold'
            else:
                return None
        
        old_value = self.current_thresholds[threshold_key]
        
        # Calculate adjustment
        adjustment_amount = old_value * self.learning_rate
        adjustment_amount = min(adjustment_amount, old_value * self.max_adjustment_per_cycle)
        
        if direction == 'increase':
            new_value = old_value + adjustment_amount
        else:
            new_value = old_value - adjustment_amount
        
        # Apply bounds
        if 'trust' in threshold_key:
            new_value = max(30, min(90, new_value))
        elif 'threshold' in threshold_key:
            new_value = max(old_value * 0.5, min(old_value * 2, new_value))
        
        # Only adjust if significant change
        if abs(new_value - old_value) < 0.1:
            return None
        
        self.current_thresholds[threshold_key] = new_value
        
        adjustment = ThresholdAdjustment(
            timestamp=time.time(),
            parameter=threshold_key,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            metrics_snapshot=self._get_metrics_snapshot()
        )
        
        self.adjustments.append(adjustment)
        
        print(f"🔧 Auto-adjusted {threshold_key}: {old_value:.2f} → {new_value:.2f}")
        print(f"   Reason: {reason}")
        
        return adjustment
    
    def _adjust_signal_weights(self) -> None:
        """Adjust confidence fusion signal weights based on performance"""
        # Calculate how well each signal correlates with correct outcomes
        if len(self.alert_history) < 50:
            return
        
        # Group alerts by outcome
        acknowledged = [a for a in self.alert_history if a.outcome == 'acknowledged']
        dismissed = [a for a in self.alert_history if a.outcome in ['dismissed', 'timed_out']]
        
        if not acknowledged or not dismissed:
            return
        
        # Compare average trust scores
        ack_avg = np.mean([a.trust_score for a in acknowledged[-50:]])
        dis_avg = np.mean([a.trust_score for a in dismissed[-50:]])
        
        # If acknowledged alerts have lower trust scores, we need to recalibrate
        if ack_avg < dis_avg:
            # Increase temporal and historical weights (more reliable signals)
            self.current_weights['temporal_persistence'] = min(
                0.35, self.current_weights['temporal_persistence'] * 1.1
            )
            self.current_weights['historical_accuracy'] = min(
                0.30, self.current_weights['historical_accuracy'] * 1.1
            )
            
            # Normalize weights
            total = sum(self.current_weights.values())
            self.current_weights = {k: v / total for k, v in self.current_weights.items()}
    
    def _calculate_current_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics for the current period"""
        current_time = time.time()
        window_start = current_time - (self.performance_window_hours * 3600)
        
        # Filter to window
        recent_alerts = [
            a for a in self.alert_history
            if a.timestamp >= window_start
        ]
        
        if not recent_alerts:
            return PerformanceMetrics(
                period_start=window_start,
                period_end=current_time
            )
        
        total = len(recent_alerts)
        acknowledged = sum(1 for a in recent_alerts if a.outcome == 'acknowledged')
        dismissed = sum(1 for a in recent_alerts if a.outcome == 'dismissed')
        timed_out = sum(1 for a in recent_alerts if a.outcome == 'timed_out')
        
        fpr = (dismissed + timed_out * 0.5) / total if total > 0 else 0
        avg_trust = np.mean([a.trust_score for a in recent_alerts])
        
        response_times = [a.response_time for a in recent_alerts if a.response_time]
        avg_response = np.mean(response_times) if response_times else 0
        
        return PerformanceMetrics(
            period_start=window_start,
            period_end=current_time,
            total_alerts=total,
            acknowledged=acknowledged,
            dismissed=dismissed,
            timed_out=timed_out,
            false_positive_rate=fpr,
            avg_trust_score=avg_trust,
            avg_response_time=avg_response
        )
    
    def _get_metrics_snapshot(self) -> Dict[str, float]:
        """Get a snapshot of current metrics"""
        metrics = self._calculate_current_metrics()
        return {
            'total_alerts': metrics.total_alerts,
            'false_positive_rate': metrics.false_positive_rate,
            'avg_trust_score': metrics.avg_trust_score
        }
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current threshold values"""
        return self.current_thresholds.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current signal weights"""
        return self.current_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_metrics = self._calculate_current_metrics()
        
        # Calculate improvement over time
        improvement = None
        if len(self.performance_history) >= 2:
            first_fpr = self.performance_history[0].false_positive_rate
            current_fpr = current_metrics.false_positive_rate
            if first_fpr > 0:
                improvement = (first_fpr - current_fpr) / first_fpr
        
        return {
            'current_metrics': {
                'total_alerts': current_metrics.total_alerts,
                'acknowledged': current_metrics.acknowledged,
                'dismissed': current_metrics.dismissed,
                'timed_out': current_metrics.timed_out,
                'false_positive_rate': current_metrics.false_positive_rate,
                'avg_trust_score': current_metrics.avg_trust_score,
                'avg_response_time': current_metrics.avg_response_time
            },
            'current_thresholds': self.current_thresholds,
            'current_weights': self.current_weights,
            'total_adjustments': len(self.adjustments),
            'improvement_since_start': improvement,
            'pending_alerts': len(self.pending_alerts),
            'history_size': len(self.alert_history),
            'per_type_stats': {
                k: {
                    'total': v['total'],
                    'acknowledged': v['acknowledged'],
                    'dismissed': v['dismissed'],
                    'timed_out': v['timed_out'],
                    'fpr': (v['dismissed'] + v['timed_out'] * 0.5) / v['total'] if v['total'] > 0 else 0
                }
                for k, v in self.type_stats.items()
            }
        }
    
    def get_adjustment_history(self, count: int = 10) -> List[Dict]:
        """Get recent adjustment history"""
        return [
            {
                'timestamp': a.timestamp,
                'parameter': a.parameter,
                'old_value': a.old_value,
                'new_value': a.new_value,
                'reason': a.reason
            }
            for a in self.adjustments[-count:]
        ]
    
    def _save_state(self) -> None:
        """Save current state to disk"""
        state_path = self.profiles_dir / "self_improvement_state.pkl"
        
        try:
            state = {
                'thresholds': self.current_thresholds,
                'weights': self.current_weights,
                'type_stats': dict(self.type_stats),
                'adjustments': [(
                    a.timestamp, a.parameter, a.old_value, 
                    a.new_value, a.reason
                ) for a in self.adjustments[-100:]],
                'performance_history': [
                    (p.period_start, p.period_end, p.total_alerts,
                     p.acknowledged, p.dismissed, p.timed_out,
                     p.false_positive_rate)
                    for p in self.performance_history[-50:]
                ]
            }
            
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            print(f"⚠️ Error saving self-improvement state: {e}")
    
    def _load_state(self) -> None:
        """Load saved state from disk"""
        state_path = self.profiles_dir / "self_improvement_state.pkl"
        
        if not state_path.exists():
            return
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.current_thresholds.update(state.get('thresholds', {}))
            self.current_weights.update(state.get('weights', {}))
            
            for k, v in state.get('type_stats', {}).items():
                self.type_stats[k].update(v)
            
            print(f"✅ Loaded self-improvement state")
            print(f"   Previous adjustments: {len(state.get('adjustments', []))}")
            
        except Exception as e:
            print(f"⚠️ Error loading self-improvement state: {e}")
    
    def reset(self) -> None:
        """Reset to initial state"""
        self.pending_alerts.clear()
        self.alert_history.clear()
        self.type_stats.clear()
        self.adjustments.clear()
        self.performance_history.clear()
        
        # Reset thresholds to defaults
        self.current_thresholds = {
            'trust_threshold': 60.0,
            'speed_threshold': 30.0,
            'loitering_threshold': 15.0,
            'abandonment_threshold': 5.0,
        }
        
        self.current_weights = {
            'vision_confidence': 0.25,
            'temporal_persistence': 0.20,
            'motion_stability': 0.15,
            'crowd_context': 0.15,
            'zone_rules': 0.10,
            'historical_accuracy': 0.15,
        }
        
        print("🔄 Self-improvement engine reset to initial state")
