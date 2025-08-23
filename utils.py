"""
Utility functions for AbnoGuard
Handles logging, drawing, timestamps, and file operations
"""

import os
import csv
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def setup_directories():
    """Create necessary output directories"""
    directories = [
        'outputs',
        'outputs/snaps',
        'outputs/logs',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create alerts CSV file if it doesn't exist
    alerts_file = 'outputs/logs/alerts.csv'
    if not os.path.exists(alerts_file):
        with open(alerts_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'type', 'track_id', 'description', 
                'snapshot_path', 'severity', 'additional_info'
            ])
        print(f"📝 Created alerts log: {alerts_file}")

def check_dependencies():
    """Check if required packages are available"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'cv2', 
        'numpy', 'pandas', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are available")
    return True

def save_alert(alert, frame):
    """
    Save alert to CSV and save snapshot
    
    Args:
        alert: Alert dictionary
        frame: Frame to save as snapshot
    """
    timestamp = alert['timestamp']
    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Save snapshot
    snapshot_filename = f"snap_{int(timestamp)}_{alert['track_id']}_{alert['type']}.jpg"
    snapshot_path = f"outputs/snaps/{snapshot_filename}"
    
    try:
        cv2.imwrite(snapshot_path, frame)
    except Exception as e:
        print(f"❌ Error saving snapshot: {e}")
        snapshot_path = "ERROR_SAVING"
    
    # Prepare additional info
    additional_info = {}
    if 'position' in alert:
        additional_info['position'] = alert['position']
    if 'speed' in alert:
        additional_info['speed'] = alert['speed']
    if 'avg_speed' in alert:
        additional_info['avg_speed'] = alert['avg_speed']
    if 'time_span' in alert:
        additional_info['time_span'] = alert['time_span']
    if 'area_covered' in alert:
        additional_info['area_covered'] = alert['area_covered']
    if 'cosine_similarity' in alert:
        additional_info['cosine_similarity'] = alert['cosine_similarity']
    
    # Save to CSV
    alerts_file = 'outputs/logs/alerts.csv'
    try:
        with open(alerts_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                alert['type'],
                alert['track_id'],
                alert['description'],
                snapshot_path,
                alert.get('severity', 'medium'),
                str(additional_info)
            ])
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")

def draw_alerts_overlay(frame, alerts):
    """
    Draw alerts overlay on frame
    
    Args:
        frame: Input frame
        alerts: List of alerts to display
        
    Returns:
        Frame with alerts overlay
    """
    if not alerts:
        return frame
    
    overlay = frame.copy()
    
    # Draw alert banner at top
    banner_height = 60
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (0, 0, 255), -1)
    
    # Add alert text
    alert_text = f"🚨 {len(alerts)} ALERTS DETECTED"
    cv2.putText(overlay, alert_text, (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Draw individual alert details
    y_offset = banner_height + 20
    for i, alert in enumerate(alerts[:5]):  # Show max 5 alerts
        # Alert background
        cv2.rectangle(overlay, (10, y_offset - 15), (600, y_offset + 15), (0, 0, 0), -1)
        
        # Alert text
        alert_line = f"{alert['type'].upper()}: {alert['description'][:50]}..."
        cv2.putText(overlay, alert_line, (15, y_offset + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 30
        
        if i >= 4:  # Max 5 alerts shown
            break
    
    return overlay

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

def calculate_fps(frame_count, start_time):
    """Calculate current FPS"""
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def resize_frame(frame, max_width=1200, max_height=800):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame

def draw_progress_bar(frame, progress, text="Progress"):
    """Draw progress bar on frame"""
    height, width = frame.shape[:2]
    
    # Progress bar dimensions
    bar_width = 300
    bar_height = 20
    bar_x = (width - bar_width) // 2
    bar_y = height - 50
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (100, 100, 100), -1)
    
    # Progress
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                 (0, 255, 0), -1)
    
    # Text
    cv2.putText(frame, f"{text}: {progress*100:.1f}%", 
               (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

