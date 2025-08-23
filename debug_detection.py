#!/usr/bin/env python3
"""
Debug Detection Script for AbnoGuard
This script helps debug why YOLOv8 might not be detecting objects in your video.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import sys

def debug_yolo_detection(video_path):
    """Debug YOLOv8 detection on a video file"""
    
    print("🔍 Starting YOLOv8 Detection Debug...")
    print(f"📹 Video file: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Load YOLOv8 model
    print("🤖 Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for faster processing
        print("✅ YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   - FPS: {fps}")
    print(f"   - Total Frames: {frame_count}")
    print(f"   - Resolution: {width}x{height}")
    
    # Process first few frames to debug
    frame_num = 0
    max_debug_frames = 10
    
    print(f"\n🔍 Processing first {max_debug_frames} frames for debugging...")
    
    while frame_num < max_debug_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ End of video reached at frame {frame_num}")
            break
        
        frame_num += 1
        print(f"\n📸 Frame {frame_num}:")
        
        # Run YOLOv8 detection
        try:
            results = model(frame, verbose=False, conf=0.25)  # Lower confidence for debugging
            
            # Check if we have any detections
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"   ✅ Detections found: {len(result.boxes)}")
                    
                    # Show first few detections
                    for i, box in enumerate(result.boxes[:5]):  # Show first 5 detections
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_names = model.names
                        class_name = class_names[class_id] if class_id in class_names else f"class_{class_id}"
                        
                        print(f"      - {class_name} (conf: {conf:.2f}) at ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                        
                        # Check if this is a target class
                        target_classes = ['person', 'backpack', 'handbag', 'suitcase', 'bag', 'book', 'cell phone', 'laptop']
                        if class_name in target_classes:
                            print(f"        🎯 TARGET CLASS DETECTED!")
                else:
                    print(f"   ❌ No detections in this frame")
            else:
                print(f"   ❌ No results from YOLOv8")
                
        except Exception as e:
            print(f"   ❌ Error during detection: {e}")
        
        # Show frame with detections for visual debugging
        try:
            # Draw detections on frame
            annotated_frame = results[0].plot() if results and len(results) > 0 else frame
            
            # Resize for display if too large
            display_frame = annotated_frame
            if annotated_frame.shape[1] > 800:
                scale = 800 / annotated_frame.shape[1]
                new_width = int(annotated_frame.shape[1] * scale)
                new_height = int(annotated_frame.shape[0] * scale)
                display_frame = cv2.resize(annotated_frame, (new_width, new_height))
            
            # Show frame
            cv2.imshow(f'Debug Frame {frame_num}', display_frame)
            cv2.waitKey(1000)  # Show for 1 second
            
        except Exception as e:
            print(f"   ❌ Error displaying frame: {e}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🔍 Debug complete! Processed {frame_num} frames.")
    print("\n💡 Troubleshooting Tips:")
    print("1. If no detections: Your video might not contain objects YOLOv8 can recognize")
    print("2. If detections but no target classes: YOLOv8 is working but not finding people/bags")
    print("3. Try running on a different video with clear people/objects")
    print("4. Check if your video has good lighting and clear objects")
    print("5. YOLOv8 is more sensitive than YOLOv5 - it should detect more objects")

def test_with_sample_images():
    """Test YOLOv8 with some sample images to verify it's working"""
    print("\n🧪 Testing YOLOv8 with sample images...")
    
    try:
        model = YOLO('yolov8n.pt')
        
        # Create a simple test image with a colored rectangle (simulating an object)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a blue rectangle (simulating a person)
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 0, 0), -1)
        
        # Draw a green rectangle (simulating a bag)
        cv2.rectangle(test_image, (450, 200), (550, 300), (0, 255, 0), -1)
        
        print("   📸 Created test image with colored rectangles")
        
        # Run detection
        results = model(test_image, verbose=False, conf=0.25)
        
        if len(results) > 0 and results[0].boxes is not None:
            print(f"   ✅ YOLOv8 processed test image successfully")
            print(f"   📊 Found {len(results[0].boxes)} detections")
        else:
            print(f"   ❌ YOLOv8 found no detections in test image")
            
    except Exception as e:
        print(f"   ❌ Error testing YOLOv8: {e}")

if __name__ == "__main__":
    print("🚀 AbnoGuard Detection Debug Tool")
    print("🔍 Now using YOLOv8 for Enhanced Detection")
    print("=" * 50)
    
    # Test YOLOv8 with sample images first
    test_with_sample_images()
    
    # Ask user for video path
    print("\n" + "=" * 50)
    video_path = input("Enter the path to your video file (or press Enter to use test video): ").strip()
    
    if not video_path:
        # Use one of the test videos
        test_videos = [
            "realistic_test.mp4",
            "test_surveillance.mp4", 
            "ultra_realistic_test.mp4"
        ]
        
        for video in test_videos:
            if os.path.exists(video):
                video_path = video
                print(f"📹 Using test video: {video}")
                break
    
    if video_path:
        debug_yolo_detection(video_path)
    else:
        print("❌ No video file found to debug")
    
    print("\n🎯 Debug complete! Check the output above for issues.")
