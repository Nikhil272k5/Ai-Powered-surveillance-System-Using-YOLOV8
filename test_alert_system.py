#!/usr/bin/env python3
"""
Test the alert system directly to ensure it's working
"""

import time
import cv2
import numpy as np
from logic_abandonment import AbandonmentDetector
from logic_anomaly import AnomalyDetector
from utils import save_alert, setup_directories

def test_alert_system():
    """Test the alert system directly"""
    print("🧪 Testing Alert System Directly")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Initialize detectors
    abandonment_detector = AbandonmentDetector(abandonment_threshold=2.0)  # Very short threshold for testing
    anomaly_detector = AnomalyDetector(speed_threshold=20.0, loitering_threshold=5.0)
    
    print("✅ Detectors initialized")
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Test Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Test 1: Simulate abandoned object detection
    print("\n🔍 Test 1: Abandoned Object Detection")
    
    # Simulate tracked objects with a bag
    tracked_objects = [
        [1, 100, 100, 160, 140, 'bag', 0.9],  # bag at position (100, 100)
        [2, 200, 200, 280, 380, 'person', 0.8],  # person at position (200, 200)
    ]
    
    # Wait a bit to simulate time passing
    time.sleep(3)
    
    # Check for abandonment
    abandonment_alerts = abandonment_detector.update(tracked_objects, {})
    print(f"   Abandonment alerts: {len(abandonment_alerts)}")
    
    for alert in abandonment_alerts:
        print(f"   🚨 {alert['type']}: {alert['description']}")
        # Save the alert
        save_alert(alert, frame)
    
    # Test 2: Simulate anomaly detection
    print("\n🔍 Test 2: Anomaly Detection")
    
    # Simulate movement history for speed spike
    current_time = time.time()
    movement_history = {
        2: [  # Person 2
            {'timestamp': current_time - 2, 'position': np.array([200, 200])},
            {'timestamp': current_time - 1, 'position': np.array([250, 200])},
            {'timestamp': current_time, 'position': np.array([350, 200])},  # Fast movement
        ]
    }
    
    # Update anomaly detector with fake history
    anomaly_detector.movement_history = movement_history
    
    # Check for anomalies
    anomaly_alerts = anomaly_detector.update(tracked_objects, {})
    print(f"   Anomaly alerts: {len(anomaly_alerts)}")
    
    for alert in anomaly_alerts:
        print(f"   🚨 {alert['type']}: {alert['description']}")
        # Save the alert
        save_alert(alert, frame)
    
    # Test 3: Simulate loitering detection
    print("\n🔍 Test 3: Loitering Detection")
    
    # Create a person that stays in small area
    loitering_history = {
        3: [  # Person 3
            {'timestamp': current_time - 10, 'position': np.array([400, 300])},
            {'timestamp': current_time - 8, 'position': np.array([405, 305])},
            {'timestamp': current_time - 6, 'position': np.array([410, 300])},
            {'timestamp': current_time - 4, 'position': np.array([405, 295])},
            {'timestamp': current_time - 2, 'position': np.array([400, 300])},
            {'timestamp': current_time, 'position': np.array([402, 302])},
        ]
    }
    
    # Add person 3 to tracked objects
    tracked_objects.append([3, 400, 300, 480, 480, 'person', 0.8])
    
    # Update anomaly detector with loitering history
    anomaly_detector.movement_history.update(loitering_history)
    anomaly_detector.alerted_tracks.clear()  # Reset to allow new alerts
    
    # Check for anomalies again
    anomaly_alerts2 = anomaly_detector.update(tracked_objects, {})
    print(f"   Additional anomaly alerts: {len(anomaly_alerts2)}")
    
    for alert in anomaly_alerts2:
        print(f"   🚨 {alert['type']}: {alert['description']}")
        # Save the alert
        save_alert(alert, frame)
    
    # Test 4: Check output files
    print("\n🔍 Test 4: Checking Output Files")
    
    import os
    alerts_file = 'outputs/logs/alerts.csv'
    snaps_dir = 'outputs/snaps/'
    
    if os.path.exists(alerts_file):
        with open(alerts_file, 'r') as f:
            lines = f.readlines()
            print(f"   Alerts CSV: {len(lines)} lines (including header)")
            if len(lines) > 1:
                print("   Sample alerts:")
                for line in lines[1:4]:  # Show first 3 alerts
                    print(f"     {line.strip()}")
    else:
        print("   ❌ Alerts CSV not found")
    
    if os.path.exists(snaps_dir):
        snap_files = os.listdir(snaps_dir)
        print(f"   Snapshots: {len(snap_files)} files")
        for snap in snap_files[:3]:  # Show first 3
            print(f"     {snap}")
    else:
        print("   ❌ Snapshots directory not found")
    
    print("\n✅ Alert System Test Complete!")
    print("   Check outputs/logs/alerts.csv for alert logs")
    print("   Check outputs/snaps/ for alert snapshots")

if __name__ == "__main__":
    test_alert_system()
