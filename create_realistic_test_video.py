#!/usr/bin/env python3
"""
Create a realistic test surveillance video that will trigger YOLO detections
"""

import cv2
import numpy as np
import os

def create_realistic_test_video():
    # Video settings
    width, height = 640, 480
    fps = 10
    duration = 30  # 30 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('realistic_test.mp4', fourcc, fps, (width, height))
    
    print(f"🎬 Creating realistic test video: {width}x{height}, {fps} FPS, {duration}s")
    
    # Create a more realistic background (indoor scene)
    background = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add some background elements
    cv2.rectangle(background, (0, 0), (width, height), (100, 100, 100), -1)
    cv2.rectangle(background, (50, 50), (width-50, height-50), (150, 150, 150), -1)
    
    # Add floor lines
    for i in range(0, width, 100):
        cv2.line(background, (i, height-50), (i, height), (80, 80, 80), 2)
    
    for frame_num in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Add frame number and timestamp
        time_sec = frame_num / fps
        cv2.putText(frame, f'Frame {frame_num}', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Time: {time_sec:.1f}s', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Person 1 - appears at frame 50, moves around (normal movement)
        if frame_num >= 50:
            # Person position changes over time
            person1_x = 100 + int(30 * np.sin(frame_num * 0.1))
            person1_y = 150 + int(20 * np.cos(frame_num * 0.15))
            
            # Draw person as a realistic humanoid shape
            # Head
            cv2.circle(frame, (person1_x + 40, person1_y + 20), 15, (255, 200, 150), -1)
            # Body
            cv2.rectangle(frame, (person1_x + 25, person1_y + 35), (person1_x + 55, person1_y + 120), (0, 100, 200), -1)
            # Arms
            cv2.rectangle(frame, (person1_x + 15, person1_y + 40), (person1_x + 25, person1_y + 80), (255, 200, 150), -1)
            cv2.rectangle(frame, (person1_x + 55, person1_y + 40), (person1_x + 65, person1_y + 80), (255, 200, 150), -1)
            # Legs
            cv2.rectangle(frame, (person1_x + 30, person1_y + 120), (person1_x + 40, person1_y + 180), (0, 50, 100), -1)
            cv2.rectangle(frame, (person1_x + 40, person1_y + 120), (person1_x + 50, person1_y + 180), (0, 50, 100), -1)
            
            cv2.putText(frame, 'Person 1', (person1_x, person1_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Bag - appears at frame 100, stays stationary (potential abandoned object)
        if frame_num >= 100:
            bag_x, bag_y = 400, 200
            # Draw bag as a realistic object
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 50, bag_y + 35), (139, 69, 19), -1)
            cv2.rectangle(frame, (bag_x + 5, bag_y + 5), (bag_x + 45, bag_y + 30), (160, 82, 45), -1)
            # Bag handle
            cv2.ellipse(frame, (bag_x + 25, bag_y - 5), (15, 8), 0, 0, 180, (139, 69, 19), 3)
            
            cv2.putText(frame, 'Bag', (bag_x, bag_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Person 2 - appears at frame 150, moves quickly (speed spike)
        if frame_num >= 150:
            # Fast movement for speed spike detection
            person2_x = 300 + int(80 * np.sin(frame_num * 0.4))
            person2_y = 200 + int(50 * np.cos(frame_num * 0.3))
            
            # Draw person 2
            # Head
            cv2.circle(frame, (person2_x + 40, person2_y + 20), 15, (255, 180, 140), -1)
            # Body
            cv2.rectangle(frame, (person2_x + 25, person2_y + 35), (person2_x + 55, person2_y + 120), (200, 0, 100), -1)
            # Arms
            cv2.rectangle(frame, (person2_x + 15, person2_y + 40), (person2_x + 25, person2_y + 80), (255, 180, 140), -1)
            cv2.rectangle(frame, (person2_x + 55, person2_y + 40), (person2_x + 65, person2_y + 80), (255, 180, 140), -1)
            # Legs
            cv2.rectangle(frame, (person2_x + 30, person2_y + 120), (person2_x + 40, person2_y + 180), (150, 0, 75), -1)
            cv2.rectangle(frame, (person2_x + 40, person2_y + 120), (person2_x + 50, person2_y + 180), (150, 0, 75), -1)
            
            cv2.putText(frame, 'Person 2', (person2_x, person2_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Suitcase - appears at frame 200, stays in one place (abandoned)
        if frame_num >= 200:
            suitcase_x, suitcase_y = 150, 300
            # Draw suitcase
            cv2.rectangle(frame, (suitcase_x, suitcase_y), (suitcase_x + 70, suitcase_y + 45), (105, 105, 105), -1)
            cv2.rectangle(frame, (suitcase_x + 5, suitcase_y + 5), (suitcase_x + 65, suitcase_y + 40), (128, 128, 128), -1)
            # Handle
            cv2.rectangle(frame, (suitcase_x + 25, suitcase_y - 8), (suitcase_x + 45, suitcase_y), (64, 64, 64), -1)
            # Wheels
            cv2.circle(frame, (suitcase_x + 15, suitcase_y + 45), 8, (50, 50, 50), -1)
            cv2.circle(frame, (suitcase_x + 55, suitcase_y + 45), 8, (50, 50, 50), -1)
            
            cv2.putText(frame, 'Suitcase', (suitcase_x, suitcase_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Person 3 - appears at frame 250, loiters in small area
        if frame_num >= 250:
            # Small movement in confined area (loitering behavior)
            person3_x = 500 + int(20 * np.sin(frame_num * 0.05))
            person3_y = 100 + int(15 * np.cos(frame_num * 0.08))
            
            # Draw person 3
            # Head
            cv2.circle(frame, (person3_x + 40, person3_y + 20), 15, (200, 150, 100), -1)
            # Body
            cv2.rectangle(frame, (person3_x + 25, person3_y + 35), (person3_x + 55, person3_y + 120), (100, 100, 0), -1)
            # Arms
            cv2.rectangle(frame, (person3_x + 15, person3_y + 40), (person3_x + 25, person3_y + 80), (200, 150, 100), -1)
            cv2.rectangle(frame, (person3_x + 55, person3_y + 40), (person3_x + 65, person3_y + 80), (200, 150, 100), -1)
            # Legs
            cv2.rectangle(frame, (person3_x + 30, person3_y + 120), (person3_x + 40, person3_y + 180), (75, 75, 0), -1)
            cv2.rectangle(frame, (person3_x + 40, person3_y + 120), (person3_x + 50, person3_y + 180), (75, 75, 0), -1)
            
            cv2.putText(frame, 'Person 3', (person3_x, person3_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add some realistic objects that YOLO might detect
        if frame_num >= 120:
            # Add a backpack
            backpack_x, backpack_y = 450, 350
            cv2.rectangle(frame, (backpack_x, backpack_y), (backpack_x + 40, backpack_y + 30), (34, 139, 34), -1)
            cv2.rectangle(frame, (backpack_x + 5, backpack_y + 5), (backpack_x + 35, backpack_y + 25), (50, 205, 50), -1)
            cv2.putText(frame, 'Backpack', (backpack_x, backpack_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📈 Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # Release video writer
    out.release()
    print(f"✅ Realistic test video created: realistic_test.mp4")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Frames: {total_frames}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print("\n🎯 This video contains realistic objects:")
    print("   - Person 1: Normal movement (humanoid shape)")
    print("   - Bag: Stationary object (realistic bag shape)")
    print("   - Person 2: Fast movement (speed spike detection)")
    print("   - Suitcase: Stationary object (realistic suitcase)")
    print("   - Person 3: Small area movement (loitering detection)")
    print("   - Backpack: Additional object for detection")

if __name__ == "__main__":
    create_realistic_test_video()
