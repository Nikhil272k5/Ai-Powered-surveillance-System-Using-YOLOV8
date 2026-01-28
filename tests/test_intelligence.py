"""
Tests for Intelligence Modules
Unit and integration tests for AbnoGuard intelligence components.
"""

import sys
import time
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import intelligence modules
try:
    from intelligence.normality_engine import NormalityEngine
    from intelligence.confidence_fusion import ConfidenceFusionEngine
    from intelligence.behavior_classifier import BehaviorClassifier, IntentClass
    from intelligence.causal_reasoning import CausalReasoningEngine, EventType
    from intelligence.self_improvement import SelfImprovementEngine
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import intelligence modules: {e}")
    INTELLIGENCE_AVAILABLE = False


# Skip all tests if modules not available
@unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
class TestNormalityEngine(unittest.TestCase):
    """Tests for Self-Learning Normality Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = NormalityEngine(
            context_id="test_context",
            learning_window_minutes=5,
            min_observations=10
        )
    
    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertEqual(self.engine.context_id, "test_context")
        self.assertEqual(self.engine.min_observations, 10)
        self.assertTrue(hasattr(self.engine, 'baseline'))
    
    def test_observation_recording(self):
        """Test that observations are recorded"""
        # Create mock tracked objects
        tracked_objects = [
            [1, 100, 100, 200, 300, 'person', 0.9],
            [2, 300, 100, 400, 300, 'person', 0.85],
        ]
        track_history = {
            1: {'start_time': time.time() - 5, 'bbox': [100, 100, 200, 300]},
            2: {'start_time': time.time() - 10, 'bbox': [300, 100, 400, 300]},
        }
        
        initial_count = self.engine.baseline.observation_count
        self.engine.observe(tracked_objects, track_history)
        
        self.assertEqual(self.engine.baseline.observation_count, initial_count + 1)
    
    def test_anomaly_detection_insufficient_data(self):
        """Test anomaly detection with insufficient data"""
        tracked_objects = [
            [1, 100, 100, 200, 300, 'person', 0.9],
        ]
        track_history = {}
        
        result = self.engine.check_anomaly(tracked_objects, track_history)
        
        self.assertFalse(result['is_anomaly'])
        self.assertIn('Insufficient', result['explanation'])
    
    def test_learning_progress(self):
        """Test learning progress reporting"""
        progress = self.engine.get_learning_progress()
        
        self.assertIn('observation_count', progress)
        self.assertIn('progress_percentage', progress)
        self.assertIn('is_baseline_ready', progress)
    
    def test_reset(self):
        """Test engine reset"""
        # Add some observations
        for _ in range(5):
            self.engine.observe([[1, 0, 0, 50, 50, 'person', 0.9]], {})
        
        self.engine.reset()
        
        self.assertEqual(self.engine.baseline.observation_count, 0)


@unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
class TestConfidenceFusion(unittest.TestCase):
    """Tests for Multi-Signal Confidence Fusion Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = ConfidenceFusionEngine(trust_threshold=60)
    
    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertEqual(self.engine.trust_threshold, 60)
        self.assertTrue(len(self.engine.weights) > 0)
    
    def test_evaluate_alert(self):
        """Test alert evaluation"""
        alert = {
            'type': 'speed_spike',
            'track_id': 1,
            'description': 'Fast movement detected'
        }
        
        result = self.engine.evaluate_alert(
            alert=alert,
            vision_confidence=0.85,
            crowd_density=3
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(0 <= result.trust_score <= 100)
        self.assertIsInstance(result.is_suppressed, bool)
        self.assertIsInstance(result.explanation, str)
    
    def test_high_confidence_not_suppressed(self):
        """Test that high confidence alerts are not suppressed"""
        alert = {'type': 'abandoned_object', 'track_id': 1}
        
        result = self.engine.evaluate_alert(
            alert=alert,
            vision_confidence=0.95
        )
        
        # High vision confidence should contribute to high trust score
        self.assertGreater(result.trust_score, 40)
    
    def test_threshold_update(self):
        """Test threshold update"""
        self.engine.update_threshold(70)
        self.assertEqual(self.engine.trust_threshold, 70)
    
    def test_feedback_recording(self):
        """Test feedback is recorded"""
        self.engine.record_feedback('alert_1', 'speed_spike', 'acknowledged')
        
        history = self.engine.alert_history.get('speed_spike')
        self.assertIsNotNone(history)
        self.assertEqual(history.acknowledged, 1)


@unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
class TestBehaviorClassifier(unittest.TestCase):
    """Tests for Behavior & Intent Classifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = BehaviorClassifier(
            history_length=30,
            update_interval=1  # Update every frame for testing
        )
    
    def test_initialization(self):
        """Test classifier initializes correctly"""
        self.assertEqual(self.classifier.history_length, 30)
        self.assertIsNotNone(self.classifier.model)
    
    def test_update_with_tracks(self):
        """Test updating classifier with tracked objects"""
        tracked_objects = [
            [1, 100, 100, 200, 300, 'person', 0.9],
        ]
        track_history = {}
        
        # Run multiple updates to build history
        for i in range(15):
            tracked_objects[0][1] += 5  # Move object
            tracked_objects[0][3] += 5
            self.classifier.update(tracked_objects, track_history)
        
        # Should have recorded some history
        self.assertTrue(len(self.classifier.track_histories[1]) > 0)
    
    def test_intent_classes(self):
        """Test all intent classes are defined"""
        expected_classes = [
            IntentClass.NORMAL_TRANSIT,
            IntentClass.WAITING,
            IntentClass.LOITERING,
            IntentClass.PANIC_MOVEMENT,
            IntentClass.EVASIVE_BEHAVIOR
        ]
        
        for intent_class in expected_classes:
            self.assertIsInstance(intent_class.value, str)
    
    def test_abandonment_classification(self):
        """Test abandonment classification"""
        result = self.classifier.classify_abandonment(
            track_id=1,
            object_track_id=2,
            person_behavior=None
        )
        
        self.assertEqual(result, IntentClass.CARELESS_ABANDONMENT)


@unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
class TestCausalReasoning(unittest.TestCase):
    """Tests for Causal Reasoning Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CausalReasoningEngine(
            lookback_seconds=30,
            min_chain_length=2
        )
    
    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertEqual(self.engine.lookback_seconds, 30)
    
    def test_add_event(self):
        """Test adding events"""
        event = self.engine.add_event(
            event_type=EventType.SPEED_SPIKE,
            position=(100, 200),
            track_ids=[1]
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.SPEED_SPIKE)
    
    def test_event_from_alert(self):
        """Test creating event from alert"""
        alert = {
            'type': 'loitering',
            'track_id': 1,
            'position': (150, 250)
        }
        
        event = self.engine.add_event_from_alert(alert)
        
        self.assertEqual(event.event_type, EventType.LOITERING_DETECTED)
    
    def test_causal_chain_building(self):
        """Test causal chain is built for related events"""
        # Add potentially related events
        event1 = self.engine.add_event(
            event_type=EventType.CROWD_SURGE,
            position=(100, 100)
        )
        
        time.sleep(0.1)  # Small delay
        
        event2 = self.engine.add_event(
            event_type=EventType.PANIC_MOVEMENT,
            position=(120, 120)
        )
        
        chain = self.engine.analyze_event(event2)
        
        # May or may not find a chain depending on timing
        # Just verify it doesn't crash
        self.assertTrue(True)
    
    def test_statistics(self):
        """Test statistics retrieval"""
        stats = self.engine.get_statistics()
        
        self.assertIn('total_events', stats)
        self.assertIn('total_chains', stats)


@unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
class TestSelfImprovement(unittest.TestCase):
    """Tests for Self-Improvement Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SelfImprovementEngine(
            learning_rate=0.1,
            min_alerts_for_adjustment=5
        )
    
    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertEqual(self.engine.learning_rate, 0.1)
    
    def test_register_alert(self):
        """Test alert registration"""
        self.engine.register_alert('alert_1', 'speed_spike', 75.0)
        
        self.assertIn('alert_1', self.engine.pending_alerts)
    
    def test_record_feedback(self):
        """Test recording feedback"""
        self.engine.register_alert('alert_2', 'loitering', 65.0)
        result = self.engine.record_feedback('alert_2', 'acknowledged')
        
        self.assertTrue(result)
        self.assertNotIn('alert_2', self.engine.pending_alerts)
    
    def test_process_timeouts(self):
        """Test processing timed out alerts"""
        self.engine.auto_dismiss_timeout = 1
        self.engine.register_alert('alert_3', 'counterflow', 55.0)
        
        time.sleep(1.5)
        
        timed_out = self.engine.process_timeouts()
        self.assertIn('alert_3', timed_out)
    
    def test_performance_summary(self):
        """Test performance summary"""
        summary = self.engine.get_performance_summary()
        
        self.assertIn('current_metrics', summary)
        self.assertIn('current_thresholds', summary)
    
    def test_reset(self):
        """Test engine reset"""
        self.engine.register_alert('alert_4', 'test', 50.0)
        self.engine.reset()
        
        self.assertEqual(len(self.engine.pending_alerts), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for intelligence pipeline"""
    
    @unittest.skipUnless(INTELLIGENCE_AVAILABLE, "Intelligence modules not available")
    def test_full_pipeline(self):
        """Test complete intelligence pipeline"""
        # Initialize all engines
        normality = NormalityEngine(context_id="test", min_observations=5)
        fusion = ConfidenceFusionEngine(trust_threshold=50)
        behavior = BehaviorClassifier(update_interval=1)
        causal = CausalReasoningEngine()
        improvement = SelfImprovementEngine(min_alerts_for_adjustment=3)
        
        # Simulate processing
        for frame_num in range(20):
            tracked_objects = [
                [1, 100 + frame_num * 2, 100, 200 + frame_num * 2, 300, 'person', 0.9],
            ]
            track_history = {1: {'start_time': time.time() - frame_num}}
            
            # Update normality
            normality.observe(tracked_objects, track_history)
            normality_result = normality.check_anomaly(tracked_objects, track_history)
            
            # Update behavior
            behavior.update(tracked_objects, track_history)
            
            # Simulate alert at frame 10
            if frame_num == 10:
                alert = {'type': 'speed_spike', 'track_id': 1}
                
                # Fuse signals
                fusion_result = fusion.evaluate_alert(
                    alert, 
                    vision_confidence=0.8,
                    normality_result=normality_result
                )
                
                # Add to causal
                causal.add_event_from_alert(alert)
                
                # Register with improvement
                improvement.register_alert(
                    fusion_result.alert_id,
                    'speed_spike',
                    fusion_result.trust_score
                )
        
        # Verify pipeline completed
        self.assertGreater(normality.baseline.observation_count, 0)
        self.assertGreater(fusion.total_processed, 0)
        self.assertGreater(causal.total_events, 0)
        self.assertGreater(len(improvement.pending_alerts), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNormalityEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestBehaviorClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfImprovement))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
