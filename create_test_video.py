#!/usr/bin/env python3
"""
Create a test surveillance video with people and objects
"""

import cv2
import numpy as np

def create_test_video():
    # Video settings
    width, height = 640, 480
    fps = 10
    duration = 30  # 30 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_surveillance.mp4', fourcc, fps, (width, height))
    
    print(f"🎬 Creating test surveillance video: {width}x{height}, {fps} FPS, {duration}s")
    
    for frame_num in range(total_frames):
        # Create base frame (dark background)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add frame number
        cv2.putText(frame, f'Frame {frame_num}', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        time_sec = frame_num / fps
        cv2.putText(frame, f'Time: {time_sec:.1f}s', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Person 1 - appears at frame 50, moves around
        if frame_num >= 50:
            # Person position changes over time
            person1_x = 100 + int(20 * np.sin(frame_num * 0.1))
            person1_y = 100 + int(10 * np.cos(frame_num * 0.15))
            
            cv2.rectangle(frame, (person1_x, person1_y), (person1_x + 80, person1_y + 200), 
                         (0, 255, 0), 2)
            cv2.putText(frame, 'Person 1', (person1_x, person1_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Bag - appears at frame 100, stays stationary (potential abandoned object)
        if frame_num >= 100:
            bag_x, bag_y = 400, 150
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 60, bag_y + 40), 
                         (255, 0, 0), 2)
            cv2.putText(frame, 'Bag', (bag_x, bag_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Person 2 - appears at frame 150, moves quickly (speed spike)
        if frame_num >= 150:
            # Fast movement for speed spike detection
            person2_x = 300 + int(50 * np.sin(frame_num * 0.3))
            person2_y = 200 + int(30 * np.cos(frame_num * 0.2))
            
            cv2.rectangle(frame, (person2_x, person2_y), (person2_x + 80, person2_y + 200), 
                         (0, 255, 0), 2)
            cv2.putText(frame, 'Person 2', (person2_x, person2_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Suitcase - appears at frame 200, stays in one place (abandoned)
        if frame_num >= 200:
            suitcase_x, suitcase_y = 150, 300
            cv2.rectangle(frame, (suitcase_x, suitcase_y), (suitcase_x + 80, suitcase_y + 60), 
                         (255, 0, 0), 2)
            cv2.putText(frame, 'Suitcase', (suitcase_x, suitcase_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Person 3 - appears at frame 250, loiters in small area
        if frame_num >= 250:
            # Small movement in confined area (loitering behavior)
            person3_x = 500 + int(15 * np.sin(frame_num * 0.05))
            person3_y = 100 + int(10 * np.cos(frame_num * 0.08))
            
            cv2.rectangle(frame, (person3_x, person3_y), (person3_x + 80, person3_y + 200), 
                         (0, 255, 0), 2)
            cv2.putText(frame, 'Person 3', (person3_x, person3_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add some background elements
        cv2.line(frame, (0, height//2), (width, height//2), (50, 50, 50), 1)
        cv2.line(frame, (width//2, 0), (width//2, height), (50, 50, 50), 1)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📈 Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # Release video writer
    out.release()
    print(f"✅ Test video created: test_surveillance.mp4")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Frames: {total_frames}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print("\n🎯 This video contains:")
    print("   - Person 1: Normal movement")
    print("   - Bag: Stationary object (potential abandoned)")
    print("   - Person 2: Fast movement (speed spike detection)")
    print("   - Suitcase: Stationary object (potential abandoned)")
    print("   - Person 3: Small area movement (loitering detection)")

if __name__ == "__main__":
    create_test_video()
