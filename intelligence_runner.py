"""
Intelligence Runner - Enhanced Video Processing Pipeline
Extends VideoRunner with all intelligence modules while maintaining backward compatibility.

This module integrates:
- Self-Learning Normality Engine
- Multi-Signal Confidence Fusion
- Behavior & Intent Classification
- Causal Reasoning & Event Chains
- Autonomous Self-Improvement
- Multi-Modal Audio Intelligence

Usage:
    # Enhanced mode (with all intelligence)
    runner = IntelligenceRunner(video_path)
    runner.run()
    
    # Basic mode (original behavior)
    runner = IntelligenceRunner(video_path, enhanced_mode=False)
    runner.run()
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import existing components
from video_runner import VideoRunner
from detector import YOLODetector
from tracker import ObjectTracker
from logic_abandonment import AbandonmentDetector
from logic_anomaly import AnomalyDetector
from utils import save_alert, draw_alerts_overlay

# Import intelligence modules
try:
    from intelligence.normality_engine import NormalityEngine
    from intelligence.confidence_fusion import ConfidenceFusionEngine
    from intelligence.behavior_classifier import BehaviorClassifier
    from intelligence.causal_reasoning import CausalReasoningEngine
    from intelligence.self_improvement import SelfImprovementEngine
    from intelligence.audio_analyzer import AudioAnalyzer
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Intelligence modules not available: {e}")
    INTELLIGENCE_AVAILABLE = False

# Import configuration
try:
    from config.config_loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class IntelligenceRunner(VideoRunner):
    """
    Enhanced Video Runner with Intelligence Modules
    
    Extends the existing VideoRunner with all intelligence capabilities
    while maintaining full backward compatibility.
    """
    
    def __init__(self, video_path: str, enhanced_mode: bool = True,
                 context_id: Optional[str] = None):
        """
        Initialize Intelligence Runner.
        
        Args:
            video_path: Path to video file
            enhanced_mode: Enable intelligence modules (True) or basic mode (False)
            context_id: Unique identifier for this camera/context (for profile persistence)
        """
        # Initialize base VideoRunner
        super().__init__(video_path)
        
        self.enhanced_mode = enhanced_mode and INTELLIGENCE_AVAILABLE
        self.context_id = context_id or Path(video_path).stem
        
        # Load configuration
        if CONFIG_AVAILABLE:
            self.config = get_config()
            # Check if config says to use enhanced mode
            if not self.config.is_enhanced_mode():
                self.enhanced_mode = False
        else:
            self.config = None
        
        # Intelligence modules (initialized if enhanced mode)
        self.normality_engine = None
        self.confidence_fusion = None
        self.behavior_classifier = None
        self.causal_reasoning = None
        self.self_improvement = None
        self.audio_analyzer = None
        
        # Enhanced features data
        self.enhanced_alerts: List[Dict] = []
        self.behavior_classifications: Dict[int, Any] = {}
        self.causal_chains: List[Any] = []
        self.current_trust_scores: Dict[str, float] = {}
        
        # Dashboard data buffer
        self.dashboard_data = {
            'alerts': [],
            'behaviors': {},
            'normality': {},
            'causal_explanations': [],
            'statistics': {}
        }
        
        if self.enhanced_mode:
            self._initialize_intelligence_modules()
        
        print(f"🧠 Intelligence Runner initialized")
        print(f"   Mode: {'Enhanced' if self.enhanced_mode else 'Basic'}")
        print(f"   Context ID: {self.context_id}")
    
    def _initialize_intelligence_modules(self) -> None:
        """Initialize all intelligence modules"""
        if not INTELLIGENCE_AVAILABLE:
            print("⚠️ Intelligence modules not available")
            return
        
        print("🔧 Initializing intelligence modules...")
        
        # Get configuration values
        if self.config:
            norm_cfg = self.config.normality_engine
            conf_cfg = self.config.confidence_fusion
            behav_cfg = self.config.behavior_classifier
            causal_cfg = self.config.causal_reasoning
            self_cfg = self.config.self_improvement
            audio_cfg = self.config.audio_analyzer
        else:
            # Use defaults
            norm_cfg = None
            conf_cfg = None
            behav_cfg = None
            causal_cfg = None
            self_cfg = None
            audio_cfg = None
        
        # 1. Normality Engine
        try:
            self.normality_engine = NormalityEngine(
                context_id=self.context_id,
                learning_window_minutes=getattr(norm_cfg, 'learning_window_minutes', 30) if norm_cfg else 30,
                adaptation_rate=getattr(norm_cfg, 'adaptation_rate', 0.1) if norm_cfg else 0.1,
                anomaly_threshold=getattr(norm_cfg, 'anomaly_threshold', 2.5) if norm_cfg else 2.5
            )
        except Exception as e:
            print(f"⚠️ Normality Engine init failed: {e}")
            self.normality_engine = None
        
        # 2. Confidence Fusion
        try:
            self.confidence_fusion = ConfidenceFusionEngine(
                trust_threshold=getattr(conf_cfg, 'trust_threshold', 60) if conf_cfg else 60,
                weights=getattr(conf_cfg, 'weights', None) if conf_cfg else None
            )
        except Exception as e:
            print(f"⚠️ Confidence Fusion init failed: {e}")
            self.confidence_fusion = None
        
        # 3. Behavior Classifier
        try:
            self.behavior_classifier = BehaviorClassifier(
                model_type=getattr(behav_cfg, 'model_type', 'random_forest') if behav_cfg else 'random_forest',
                classification_threshold=getattr(behav_cfg, 'classification_threshold', 0.6) if behav_cfg else 0.6
            )
        except Exception as e:
            print(f"⚠️ Behavior Classifier init failed: {e}")
            self.behavior_classifier = None
        
        # 4. Causal Reasoning
        try:
            self.causal_reasoning = CausalReasoningEngine(
                lookback_seconds=getattr(causal_cfg, 'lookback_seconds', 30) if causal_cfg else 30,
                spatial_proximity=getattr(causal_cfg, 'spatial_proximity', 200) if causal_cfg else 200
            )
        except Exception as e:
            print(f"⚠️ Causal Reasoning init failed: {e}")
            self.causal_reasoning = None
        
        # 5. Self-Improvement
        try:
            self.self_improvement = SelfImprovementEngine(
                learning_rate=getattr(self_cfg, 'learning_rate', 0.05) if self_cfg else 0.05,
                min_alerts_for_adjustment=getattr(self_cfg, 'min_alerts_for_adjustment', 20) if self_cfg else 20
            )
        except Exception as e:
            print(f"⚠️ Self-Improvement init failed: {e}")
            self.self_improvement = None
        
        # 6. Audio Analyzer (optional)
        audio_enabled = getattr(audio_cfg, 'enabled', False) if audio_cfg else False
        if audio_enabled:
            try:
                self.audio_analyzer = AudioAnalyzer()
            except Exception as e:
                print(f"⚠️ Audio Analyzer init failed: {e}")
                self.audio_analyzer = None
        
        print("✅ Intelligence modules initialized")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the enhanced detection pipeline.
        Overrides VideoRunner._process_frame for enhanced mode.
        """
        if not self.enhanced_mode:
            # Use original processing
            return super()._process_frame(frame)
        
        # Run base detection
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, frame)
        track_history = self.tracker.get_all_tracks()
        
        # Run abandonment detection
        abandonment_alerts = self.abandonment_detector.update(
            tracked_objects, track_history
        )
        
        # Run anomaly detection
        anomaly_alerts = self.anomaly_detector.update(
            tracked_objects, track_history
        )
        
        # Combine base alerts
        base_alerts = abandonment_alerts + anomaly_alerts
        
        # === INTELLIGENCE LAYER ===
        
        # 1. Update normality engine
        normality_result = None
        if self.normality_engine:
            self.normality_engine.observe(tracked_objects, track_history)
            normality_result = self.normality_engine.check_anomaly(tracked_objects, track_history)
            self.dashboard_data['normality'] = self.normality_engine.get_learning_progress()
        
        # 2. Update behavior classifier
        if self.behavior_classifier:
            classifications = self.behavior_classifier.update(tracked_objects, track_history)
            for classification in classifications:
                self.behavior_classifications[classification.track_id] = classification
            self.dashboard_data['behaviors'] = {
                k: {
                    'intent': v.intent_class.value,
                    'confidence': v.confidence,
                    'explanation': v.explanation
                }
                for k, v in self.behavior_classifications.items()
            }
        
        # 3. Process alerts through confidence fusion
        enhanced_alerts = []
        crowd_density = sum(1 for obj in tracked_objects if obj[5] == 'person')
        
        for alert in base_alerts:
            if self.confidence_fusion:
                # Get vision confidence from detection
                vision_conf = 0.5
                for det in detections:
                    if det[5] == alert.get('track_id'):
                        vision_conf = det[4]
                        break
                
                # Evaluate alert
                fusion_result = self.confidence_fusion.evaluate_alert(
                    alert=alert,
                    vision_confidence=vision_conf,
                    tracked_objects=tracked_objects,
                    crowd_density=crowd_density,
                    normality_result=normality_result
                )
                
                # Store trust score
                self.current_trust_scores[fusion_result.alert_id] = fusion_result.trust_score
                
                if not fusion_result.is_suppressed:
                    # Add fusion data to alert
                    alert['trust_score'] = fusion_result.trust_score
                    alert['fusion_explanation'] = fusion_result.explanation
                    alert['alert_id'] = fusion_result.alert_id
                    
                    # 4. Add causal explanation
                    if self.causal_reasoning:
                        causal_explanation = self.causal_reasoning.get_explanation_for_alert(alert)
                        alert['causal_explanation'] = causal_explanation
                    
                    enhanced_alerts.append(alert)
                    
                    # 5. Register with self-improvement
                    if self.self_improvement:
                        self.self_improvement.register_alert(
                            fusion_result.alert_id,
                            alert.get('type', 'unknown'),
                            fusion_result.trust_score
                        )
            else:
                enhanced_alerts.append(alert)
        
        # Use enhanced alerts
        self.enhanced_alerts = enhanced_alerts
        
        # Process alerts (save to CSV, snapshots)
        for alert in enhanced_alerts:
            self._process_enhanced_alert(alert, frame)
        
        # Draw detections and tracks
        frame_with_detections = self.detector.draw_detections(frame, detections)
        frame_with_tracks = self._draw_tracks(frame_with_detections, tracked_objects)
        
        # Draw behavior classifications
        frame_with_behaviors = self._draw_behavior_overlay(frame_with_tracks, tracked_objects)
        
        # Draw alerts overlay with trust scores
        frame_with_alerts = self._draw_enhanced_alerts(frame_with_behaviors, enhanced_alerts)
        
        # Add enhanced info overlay
        frame_with_info = self._add_enhanced_info_overlay(frame_with_alerts)
        
        return frame_with_info
    
    def _process_enhanced_alert(self, alert: Dict, frame: np.ndarray) -> None:
        """Process and save an enhanced alert"""
        timestamp_str = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
        trust_score = alert.get('trust_score', 'N/A')
        
        print(f"🚨 [{timestamp_str}] {alert['type'].upper()}: {alert['description']}")
        print(f"   Trust Score: {trust_score}{'%' if isinstance(trust_score, (int, float)) else ''}")
        
        if 'fusion_explanation' in alert:
            print(f"   Explanation: {alert['fusion_explanation']}")
        
        if 'causal_explanation' in alert:
            print(f"   Cause: {alert['causal_explanation']}")
        
        # Save alert
        try:
            save_alert(alert, frame)
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
        
        # Update dashboard data
        self.dashboard_data['alerts'].append({
            'timestamp': alert['timestamp'],
            'type': alert['type'],
            'description': alert['description'],
            'trust_score': trust_score,
            'explanation': alert.get('fusion_explanation', ''),
            'causal_explanation': alert.get('causal_explanation', '')
        })
        
        # Keep only recent alerts
        if len(self.dashboard_data['alerts']) > 100:
            self.dashboard_data['alerts'] = self.dashboard_data['alerts'][-100:]
    
    def _draw_behavior_overlay(self, frame: np.ndarray, 
                                tracked_objects: List) -> np.ndarray:
        """Draw behavior classification labels on tracked objects"""
        if not self.behavior_classifications:
            return frame
        
        overlay = frame.copy()
        
        for obj in tracked_objects:
            track_id = obj[0]
            if track_id not in self.behavior_classifications:
                continue
            
            classification = self.behavior_classifications[track_id]
            x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            
            # Determine color based on intent
            intent = classification.intent_class.value
            if intent in ['panic_movement', 'evasive_behavior', 'suspicious_abandonment']:
                color = (0, 0, 255)  # Red for concerning
            elif intent in ['loitering']:
                color = (0, 165, 255)  # Orange for attention
            else:
                color = (0, 255, 0)  # Green for normal
            
            # Draw behavior label below bounding box
            label = f"{intent.replace('_', ' ').title()}"
            conf_label = f"({classification.confidence:.0%})"
            
            cv2.putText(overlay, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(overlay, conf_label, (x1, y2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def _draw_enhanced_alerts(self, frame: np.ndarray, 
                               alerts: List[Dict]) -> np.ndarray:
        """Draw alerts with trust scores"""
        if not alerts:
            return frame
        
        overlay = frame.copy()
        
        # Draw alert banner at top
        banner_height = 80
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (0, 0, 180), -1)
        
        # Alert count and average trust
        trust_scores = [a.get('trust_score', 0) for a in alerts if isinstance(a.get('trust_score'), (int, float))]
        avg_trust = np.mean(trust_scores) if trust_scores else 0
        
        cv2.putText(overlay, f"🚨 {len(alerts)} ALERTS | Avg Trust: {avg_trust:.0f}%",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Show latest alert
        if alerts:
            latest = alerts[-1]
            alert_text = f"{latest['type'].upper()}: {latest.get('description', '')[:50]}..."
            cv2.putText(overlay, alert_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def _add_enhanced_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add enhanced information overlay with intelligence data"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Left panel: Stats
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps:.1f}",
            f"Tracks: {len(self.tracker.get_all_tracks())}",
            f"Mode: Enhanced" if self.enhanced_mode else "Mode: Basic"
        ]
        
        # Add normality learning progress
        if self.normality_engine:
            progress = self.normality_engine.get_learning_progress()
            info_text.append(f"Normality: {progress['progress_percentage']:.0f}%")
        
        # Add behavior counts
        if self.behavior_classifications:
            intent_counts = {}
            for bc in self.behavior_classifications.values():
                intent = bc.intent_class.value
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            most_common = max(intent_counts, key=intent_counts.get) if intent_counts else "N/A"
            info_text.append(f"Behaviors: {most_common}")
        
        y_offset = 100  # Start below alert banner if present
        for text in info_text:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 22
        
        # Right panel: Recent alerts with trust
        if self.enhanced_alerts:
            x_start = width - 350
            y_offset = 100
            
            cv2.putText(overlay, "Recent Alerts:", (x_start, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            for alert in self.enhanced_alerts[-3:]:
                trust = alert.get('trust_score', 'N/A')
                trust_str = f"{trust:.0f}%" if isinstance(trust, (int, float)) else trust
                text = f"{alert['type']}: {trust_str}"
                cv2.putText(overlay, text, (x_start, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 18
        
        # Bottom: Instructions
        instructions = [
            "Press 'q' to quit | 's' to save | 'm' to toggle mode"
        ]
        y_offset = height - 30
        for text in instructions:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return overlay
    
    def run(self) -> None:
        """Run the enhanced video processing pipeline"""
        try:
            # Open video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"\n📊 Video Info:")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Duration: {total_frames/fps:.1f}s")
            print(f"\n🚀 Starting {'enhanced' if self.enhanced_mode else 'basic'} analysis...")
            print("   Press 'q' to quit, 's' to save frame, 'm' to toggle mode")
            
            # Create window
            cv2.namedWindow('AbnoGuard Intelligence', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AbnoGuard Intelligence', 1200, 800)
            
            # Main processing loop
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow('AbnoGuard Intelligence', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord('m'):
                    # Toggle mode
                    self.enhanced_mode = not self.enhanced_mode
                    mode_str = "Enhanced" if self.enhanced_mode else "Basic"
                    print(f"🔄 Switched to {mode_str} mode")
                
                # Update stats
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed_time = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed_time
                
                # Show progress
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"📈 Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - FPS: {self.fps:.1f}")
        
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._enhanced_cleanup()
    
    def _enhanced_cleanup(self) -> None:
        """Enhanced cleanup with profile saving"""
        # Save normality profile
        if self.normality_engine:
            self.normality_engine.save_profile()
        
        # Process timeouts in self-improvement
        if self.self_improvement:
            self.self_improvement.process_timeouts()
            summary = self.self_improvement.get_performance_summary()
            print(f"\n📊 Self-Improvement Summary:")
            print(f"   Total alerts: {summary['current_metrics']['total_alerts']}")
            print(f"   False positive rate: {summary['current_metrics']['false_positive_rate']:.0%}")
        
        # Call parent cleanup
        super()._cleanup()
        
        print(f"\n🧠 Intelligence Analysis Complete!")
        if self.enhanced_mode:
            print(f"   Alerts processed: {len(self.dashboard_data['alerts'])}")
            print(f"   Behaviors classified: {len(self.behavior_classifications)}")
            if self.normality_engine:
                progress = self.normality_engine.get_learning_progress()
                print(f"   Normality learning: {progress['progress_percentage']:.0f}%")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        data = self.dashboard_data.copy()
        
        # Add current statistics
        if self.confidence_fusion:
            data['fusion_stats'] = self.confidence_fusion.get_statistics()
        
        if self.self_improvement:
            data['improvement_stats'] = self.self_improvement.get_performance_summary()
        
        if self.causal_reasoning:
            data['causal_stats'] = self.causal_reasoning.get_statistics()
        
        return data
    
    def record_feedback(self, alert_id: str, outcome: str) -> bool:
        """
        Record user feedback for an alert.
        
        Args:
            alert_id: Alert identifier
            outcome: 'acknowledged' or 'dismissed'
        
        Returns:
            True if feedback recorded
        """
        if self.self_improvement:
            return self.self_improvement.record_feedback(alert_id, outcome)
        return False


def main():
    """Main entry point for Intelligence Runner"""
    import tkinter as tk
    from tkinter import filedialog
    from utils import setup_directories, check_dependencies
    
    print("🧠 AbnoGuard Intelligence Platform")
    print("🔍 Advanced Situational Awareness Engine")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    if not check_dependencies():
        print("❌ Dependencies check failed")
        return
    
    # File picker
    root = tk.Tk()
    root.withdraw()
    
    print("📁 Please select a video file...")
    video_path = filedialog.askopenfilename(
        title="Select Video for Intelligence Analysis",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*")
        ]
    )
    
    if not video_path:
        print("❌ No video selected")
        return
    
    print(f"✅ Selected: {video_path}")
    
    # Run intelligence analysis
    try:
        runner = IntelligenceRunner(video_path, enhanced_mode=True)
        runner.run()
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
