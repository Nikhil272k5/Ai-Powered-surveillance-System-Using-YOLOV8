#!/usr/bin/env python3
"""
Demonstration of AbnoGuard working system
This script simulates the complete pipeline and shows alerts being generated
"""

import time
import cv2
import numpy as np
import os
from logic_abandonment import AbandonmentDetector
from logic_anomaly import AnomalyDetector
from utils import save_alert, setup_directories

def create_demo_frame(frame_num, detections=None):
    """Create a demo frame with visual elements"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add background
    cv2.rectangle(frame, (0, 0), (640, 480), (50, 50, 50), -1)
    cv2.rectangle(frame, (50, 50), (590, 430), (100, 100, 100), -1)
    
    # Add title
    cv2.putText(frame, "AbnoGuard - AI Surveillance Demo", (100, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add frame info
    cv2.putText(frame, f"Frame: {frame_num}", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add detections if provided
    if detections:
        for det in detections:
            track_id, x1, y1, x2, y2, class_name, confidence = det
            
            # Draw bounding box
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name} #{track_id}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def simulate_surveillance_scenario():
    """Simulate a complete surveillance scenario"""
    print("🎬 AbnoGuard - AI Surveillance System Demo")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Initialize detectors with very short thresholds for demo
    abandonment_detector = AbandonmentDetector(abandonment_threshold=3.0)
    anomaly_detector = AnomalyDetector(speed_threshold=25.0, loitering_threshold=8.0)
    
    print("✅ Detectors initialized")
    print("   - Abandonment threshold: 3.0s")
    print("   - Speed threshold: 25.0 px/frame")
    print("   - Loitering threshold: 8.0s")
    
    # Simulation parameters
    total_frames = 100
    frame_rate = 2  # 2 FPS for demo
    
    # Track objects and their states
    tracked_objects = []
    movement_history = {}
    current_time = time.time()
    
    print(f"\n🚀 Starting simulation: {total_frames} frames at {frame_rate} FPS")
    print("   Timeline:")
    print("   - Frames 0-20: Empty scene")
    print("   - Frames 20-40: Person enters with bag")
    print("   - Frames 40-60: Person leaves bag behind")
    print("   - Frames 60-80: Person moves quickly (speed spike)")
    print("   - Frames 80-100: Person loiters in small area")
    
    for frame_num in range(total_frames):
        print(f"\n📸 Frame {frame_num:3d}/100")
        
        # Create frame
        frame = create_demo_frame(frame_num, tracked_objects)
        
        # Simulate different scenarios based on frame number
        if frame_num == 20:
            print("   👤 Person enters with bag")
            tracked_objects = [
                [1, 100, 150, 180, 350, 'person', 0.9],  # Person
                [2, 200, 200, 260, 240, 'bag', 0.8],     # Bag
            ]
            # Initialize movement history
            movement_history[1] = [{'timestamp': current_time, 'position': np.array([140, 250])}]
            
        elif frame_num == 40:
            print("   🎒 Person leaves bag behind")
            tracked_objects = [
                [2, 200, 200, 260, 240, 'bag', 0.8],     # Bag stays
            ]
            # Person left, bag is now alone
            
        elif frame_num == 60:
            print("   🏃 Person returns and moves quickly")
            tracked_objects = [
                [1, 100, 150, 180, 350, 'person', 0.9],  # Person returns
                [2, 200, 200, 260, 240, 'bag', 0.8],     # Bag
            ]
            # Add fast movement history
            movement_history[1] = [
                {'timestamp': current_time - 2, 'position': np.array([140, 250])},
                {'timestamp': current_time - 1, 'position': np.array([200, 250])},
                {'timestamp': current_time, 'position': np.array([300, 250])},  # Fast movement
            ]
            
        elif frame_num == 80:
            print("   🚶 Person loiters in small area")
            tracked_objects = [
                [1, 300, 150, 380, 350, 'person', 0.9],  # Person in new position
                [2, 200, 200, 260, 240, 'bag', 0.8],     # Bag
            ]
            # Add loitering movement history
            movement_history[1] = [
                {'timestamp': current_time - 10, 'position': np.array([340, 250])},
                {'timestamp': current_time - 8, 'position': np.array([345, 255])},
                {'timestamp': current_time - 6, 'position': np.array([350, 250])},
                {'timestamp': current_time - 4, 'position': np.array([345, 245])},
                {'timestamp': current_time - 2, 'position': np.array([340, 250])},
                {'timestamp': current_time, 'position': np.array([342, 252])},
            ]
        
        # Update current time
        current_time = time.time()
        
        # Run detection algorithms
        abandonment_alerts = abandonment_detector.update(tracked_objects, {})
        anomaly_alerts = anomaly_detector.update(tracked_objects, {})
        
        # Update anomaly detector with movement history
        if movement_history:
            anomaly_detector.movement_history.update(movement_history)
        
        # Process alerts
        all_alerts = abandonment_alerts + anomaly_alerts
        
        if all_alerts:
            print(f"   🚨 {len(all_alerts)} alerts detected!")
            for alert in all_alerts:
                print(f"      - {alert['type']}: {alert['description']}")
                try:
                    save_alert(alert, frame)
                except Exception as e:
                    print(f"      ❌ Error saving alert: {e}")
        else:
            print("   ✅ No alerts")
        
        # Show progress
        if frame_num % 20 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   📈 Progress: {progress:.0f}%")
        
        # Simulate frame processing time
        time.sleep(1/frame_rate)
    
    # Final summary
    print(f"\n🎯 Simulation Complete!")
    print("=" * 60)
    
    # Check output files
    alerts_file = 'outputs/logs/alerts.csv'
    snaps_dir = 'outputs/snaps/'
    
    if os.path.exists(alerts_file):
        with open(alerts_file, 'r') as f:
            lines = f.readlines()
            print(f"📊 Alerts Generated: {len(lines) - 1} (excluding header)")
            if len(lines) > 1:
                print("   Sample alerts:")
                for line in lines[1:4]:  # Show first 3 alerts
                    print(f"     {line.strip()}")
    
    if os.path.exists(snaps_dir):
        snap_files = os.listdir(snaps_dir)
        print(f"📸 Snapshots Saved: {len(snap_files)} files")
        for snap in snap_files:
            print(f"     {snap}")
    
    print(f"\n✅ Demo Complete! AbnoGuard system is working correctly.")
    print("   Check the outputs/ directory for generated alerts and snapshots.")

if __name__ == "__main__":
    simulate_surveillance_scenario()
