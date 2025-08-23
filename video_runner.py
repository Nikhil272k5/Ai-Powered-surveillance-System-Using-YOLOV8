"""
Video Runner - Main video processing pipeline
Handles video loading, processing, and display with all detection components
"""

import cv2
import time
import numpy as np
from pathlib import Path

from detector import YOLODetector
from tracker import ObjectTracker
from logic_abandonment import AbandonmentDetector
from logic_anomaly import AnomalyDetector
from utils import save_alert, draw_alerts_overlay

class VideoRunner:
    def __init__(self, video_path):
        """
        Initialize video runner
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        
        # Initialize detection components
        self.detector = YOLODetector()
        self.tracker = ObjectTracker()
        self.abandonment_detector = AbandonmentDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Processing stats
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        print(f"🎬 Video Runner initialized for: {Path(video_path).name}")
    
    def run(self):
        """Run the complete video processing pipeline"""
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
            
            print(f"📊 Video Info:")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Total Frames: {total_frames}")
            print(f"   Duration: {total_frames/fps:.1f}s")
            print("\n🚀 Starting analysis... Press 'q' to quit, 's' to save frame")
            
            # Create window
            cv2.namedWindow('AbnoGuard - AI Surveillance', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AbnoGuard - AI Surveillance', 1200, 800)
            
            # Main processing loop
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow('AbnoGuard - AI Surveillance', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                
                # Update stats
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed_time = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed_time
                
                # Show progress
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"📈 Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - FPS: {self.fps:.1f}")
        
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame):
        """Process a single frame through the detection pipeline"""
        # Run object detection
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, frame)
        
        # Run abandonment detection
        abandonment_alerts = self.abandonment_detector.update(
            tracked_objects, self.tracker.get_all_tracks()
        )
        
        # Run anomaly detection
        anomaly_alerts = self.anomaly_detector.update(
            tracked_objects, self.tracker.get_all_tracks()
        )
        
        # Combine all alerts
        all_alerts = abandonment_alerts + anomaly_alerts
        
        # Process alerts
        for alert in all_alerts:
            self._process_alert(alert, frame)
        
        # Draw detections and tracks
        frame_with_detections = self.detector.draw_detections(frame, detections)
        frame_with_tracks = self._draw_tracks(frame_with_detections, tracked_objects)
        
        # Draw alerts overlay
        frame_with_alerts = draw_alerts_overlay(frame_with_tracks, all_alerts)
        
        # Add info overlay
        frame_with_info = self._add_info_overlay(frame_with_alerts)
        
        return frame_with_info
    
    def _draw_tracks(self, frame, tracked_objects):
        """Draw tracking information on frame"""
        frame_copy = frame.copy()
        
        for obj in tracked_objects:
            track_id, x1, y1, x2, y2, class_name, confidence = obj
            
            # Draw bounding box with track ID
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID and class
            label = f"ID:{track_id} {class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            cv2.putText(frame_copy, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def _process_alert(self, alert, frame):
        """Process and save an alert"""
        # Print alert to terminal
        timestamp_str = time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))
        print(f"🚨 [{timestamp_str}] {alert['type'].upper()}: {alert['description']}")
        
        # Save alert to CSV and snapshot
        try:
            save_alert(alert, frame)
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
    
    def _add_info_overlay(self, frame):
        """Add information overlay to frame"""
        overlay = frame.copy()
        
        # Add stats
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps:.1f}",
            f"Active Tracks: {len(self.tracker.get_all_tracks())}",
            f"Abandoned Objects: {len(self.abandonment_detector.get_abandoned_objects())}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(overlay, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
        
        # Add instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save frame"
        ]
        
        y_offset = frame.shape[0] - 60
        for text in instructions:
            cv2.putText(overlay, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_offset += 20
        
        return overlay
    
    def _save_frame(self, frame):
        """Save current frame"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"outputs/snaps/frame_{timestamp}_{self.frame_count:06d}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        elapsed_time = time.time() - self.start_time
        print(f"\n✅ Analysis Complete!")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Total Time: {elapsed_time:.1f}s")
        print(f"   Average FPS: {self.frame_count/elapsed_time:.1f}")
        print(f"   Alerts saved to: outputs/logs/alerts.csv")
        print(f"   Snapshots saved to: outputs/snaps/")

