"""
AbnoGuard v3.0 - Intelligent Video Runner
Dual-brain processing pipeline with perception and cognition
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from detector import YOLODetector
from tracker import ObjectTracker
from utils import save_alert, draw_alerts_overlay, setup_directories


class IntelligentRunner:
    """
    Video processing pipeline with dual-brain intelligence.
    
    Brain-1 (Perception): Detection + Motion + Pose + Zones
    Brain-2 (Cognition): Narrative + Memory + Risk + Ethics + Explain
    """
    
    def __init__(self, video_path: str, config: Optional[dict] = None):
        self.video_path = video_path
        self.config = config or {}
        self.cap = None
        
        # Initialize detection
        self.detector = YOLODetector()
        self.tracker = ObjectTracker()
        
        # Try to initialize dual-brain system
        self.intelligence = None
        try:
            from core.fusion_engine import DualBrainFusion
            self.intelligence = DualBrainFusion(config)
            print("🧠 Dual-Brain Intelligence System active")
        except Exception as e:
            print(f"⚠️ Basic mode (intelligence modules not loaded): {e}")
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.alerts_generated = []
        
        print(f"🎬 Intelligent Runner initialized for: {Path(video_path).name}")
    
    def run(self, headless: bool = False, callback=None):
        """Run the complete intelligent video processing pipeline"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            
            # Get video properties
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📊 Video: {width}x{height} @ {video_fps:.1f}fps, {total_frames} frames")
            
            if not headless:
                cv2.namedWindow('AbnoGuard Intelligence', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('AbnoGuard Intelligence', 1200, 800)
            
            # Main processing loop
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self._process_frame(frame)
                
                # Callback for external systems (dashboard, etc.)
                if callback:
                    callback(result)
                
                # Display
                if not headless:
                    display_frame = result['display_frame']
                    cv2.imshow('AbnoGuard Intelligence', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._save_frame(frame)
                
                # Progress
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"📈 {progress:.1f}% | Frame {self.frame_count} | FPS: {self.fps:.1f}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame through the intelligence pipeline"""
        
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracked_objects = self.tracker.update(detections, frame)
        
        # Draw basic detections
        frame_vis = self.detector.draw_detections(frame, detections)
        frame_vis = self._draw_tracks(frame_vis, tracked_objects)
        
        # Intelligence processing
        alerts = []
        narratives = []
        briefs = []
        intelligence_output = None
        
        if self.intelligence is not None:
            try:
                intelligence_output = self.intelligence.process(
                    frame, 
                    detections, 
                    tracked_objects
                )
                
                alerts = intelligence_output.final_alerts
                narratives = intelligence_output.narratives
                briefs = intelligence_output.incident_briefs
                
                # Draw intelligence overlays
                frame_vis = self._draw_intelligence_overlay(
                    frame_vis, intelligence_output
                )
                
            except Exception as e:
                print(f"⚠️ Intelligence error: {e}")
        
        # Handle alerts
        for alert in alerts:
            self._process_alert(alert, frame)
        
        # Add info overlay
        frame_vis = self._add_info_overlay(frame_vis, len(alerts))
        
        return {
            'frame': frame,
            'display_frame': frame_vis,
            'detections': detections,
            'tracked_objects': tracked_objects,
            'alerts': alerts,
            'narratives': narratives,
            'briefs': briefs,
            'intelligence': intelligence_output,
            'frame_count': self.frame_count,
            'fps': self.fps
        }
    
    def _draw_tracks(self, frame: np.ndarray, tracked_objects: List) -> np.ndarray:
        """Draw tracking information"""
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, conf = obj
            
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"ID:{track_id} {class_name}"
            cv2.putText(frame, label, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _draw_intelligence_overlay(self, frame: np.ndarray, 
                                   intel: 'IntelligenceOutput') -> np.ndarray:
        """Draw intelligence analysis overlay"""
        # Risk indicators
        for entity in intel.perception.entities:
            x1, y1, x2, y2 = entity.bbox
            
            # Color by threat level
            if entity.threat_level.value == 'critical':
                color = (0, 0, 255)
            elif entity.threat_level.value == 'high':
                color = (0, 128, 255)
            elif entity.threat_level.value == 'medium':
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            
            # Draw threat indicator
            if entity.threat_score > 0.3:
                cv2.putText(frame, f"⚠️ {entity.threat_score:.0%}",
                           (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Alert banner
        if intel.final_alerts:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 200), -1)
            text = f"🚨 {len(intel.final_alerts)} ALERTS | {intel.narratives[0][:60]}..." if intel.narratives else f"🚨 {len(intel.final_alerts)} ALERTS"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _add_info_overlay(self, frame: np.ndarray, alert_count: int) -> np.ndarray:
        """Add status information overlay"""
        info = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps:.1f}",
            f"Tracks: {len(self.tracker.get_all_tracks())}",
            f"Alerts: {alert_count}"
        ]
        
        y = 30
        for text in info:
            cv2.putText(frame, text, (frame.shape[1] - 150, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y += 20
        
        # Intelligence status
        if self.intelligence:
            status = self.intelligence.get_status()
            cv2.putText(frame, f"🧠 Intelligence: ON",
                       (frame.shape[1] - 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def _process_alert(self, alert: Dict, frame: np.ndarray):
        """Process and save an alert"""
        timestamp_str = time.strftime('%H:%M:%S')
        alert_type = alert.get('type', 'unknown')
        
        print(f"🚨 [{timestamp_str}] {alert_type.upper()}: {alert.get('description', '')}")
        
        # Save alert
        try:
            save_alert({
                'timestamp': time.time(),
                'type': alert_type,
                'track_id': alert.get('track_id', 0),
                'description': alert.get('description', ''),
                'severity': alert.get('severity', 'medium')
            }, frame)
        except Exception as e:
            print(f"⚠️ Error saving alert: {e}")
        
        self.alerts_generated.append(alert)
    
    def _save_frame(self, frame: np.ndarray):
        """Save current frame"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"outputs/snaps/frame_{timestamp}_{self.frame_count:06d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved: {filename}")
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.intelligence:
            self.intelligence.cleanup()
        
        elapsed = time.time() - self.start_time
        print(f"\n✅ Complete! {self.frame_count} frames in {elapsed:.1f}s ({self.frame_count/elapsed:.1f} FPS)")
        print(f"📊 Alerts generated: {len(self.alerts_generated)}")


def main():
    """Main entry point"""
    import tkinter as tk
    from tkinter import filedialog
    
    setup_directories()
    
    print("🧠 AbnoGuard v3.0 - Dual-Brain Intelligence System")
    print("=" * 50)
    
    root = tk.Tk()
    root.withdraw()
    
    video_path = filedialog.askopenfilename(
        title="Select Video for Intelligence Analysis",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    
    if not video_path:
        print("❌ No video selected")
        return
    
    runner = IntelligentRunner(video_path)
    runner.run()


if __name__ == "__main__":
    main()
