#!/usr/bin/env python3
"""
Create an ultra-realistic test surveillance video that will definitely trigger YOLO detections
"""

import cv2
import numpy as np
import os

def create_ultra_realistic_video():
    # Video settings
    width, height = 640, 480
    fps = 10
    duration = 30  # 30 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ultra_realistic_test.mp4', fourcc, fps, (width, height))
    
    print(f"🎬 Creating ultra-realistic test video: {width}x{height}, {fps} FPS, {duration}s")
    
    # Create a realistic indoor background
    background = np.ones((height, width, 3), dtype=np.uint8) * 180
    
    # Add realistic background elements
    # Floor
    cv2.rectangle(background, (0, height-100), (width, height), (120, 120, 120), -1)
    
    # Walls
    cv2.rectangle(background, (0, 0), (width, 100), (200, 200, 200), -1)
    cv2.rectangle(background, (0, 0), (50, height), (200, 200, 200), -1)
    cv2.rectangle(background, (width-50, 0), (width, height), (200, 200, 200), -1)
    
    # Add some texture
    for i in range(0, width, 20):
        cv2.line(background, (i, height-100), (i, height), (100, 100, 100), 1)
    
    for frame_num in range(total_frames):
        # Start with background
        frame = background.copy()
        
        # Add frame number and timestamp
        time_sec = frame_num / fps
        cv2.putText(frame, f'Frame {frame_num}', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f'Time: {time_sec:.1f}s', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Person 1 - appears at frame 50, moves around (normal movement)
        if frame_num >= 50:
            # Person position changes over time
            person1_x = 100 + int(25 * np.sin(frame_num * 0.1))
            person1_y = 150 + int(15 * np.cos(frame_num * 0.15))
            
            # Draw person as a very realistic humanoid shape
            # Head (more realistic)
            cv2.circle(frame, (person1_x + 40, person1_y + 25), 18, (255, 200, 150), -1)
            cv2.circle(frame, (person1_x + 35, person1_y + 20), 3, (0, 0, 0), -1)  # Left eye
            cv2.circle(frame, (person1_x + 45, person1_y + 20), 3, (0, 0, 0), -1)  # Right eye
            
            # Body (more realistic proportions)
            cv2.rectangle(frame, (person1_x + 25, person1_y + 43), (person1_x + 55, person1_y + 130), (0, 100, 200), -1)
            
            # Arms (more realistic)
            cv2.rectangle(frame, (person1_x + 15, person1_y + 50), (person1_x + 25, person1_y + 90), (255, 200, 150), -1)
            cv2.rectangle(frame, (person1_x + 55, person1_y + 50), (person1_x + 65, person1_y + 90), (255, 200, 150), -1)
            
            # Legs (more realistic)
            cv2.rectangle(frame, (person1_x + 30, person1_y + 130), (person1_x + 40, person1_y + 190), (0, 50, 100), -1)
            cv2.rectangle(frame, (person1_x + 40, person1_y + 130), (person1_x + 50, person1_y + 190), (0, 50, 100), -1)
            
            # Add some clothing details
            cv2.rectangle(frame, (person1_x + 30, person1_y + 43), (person1_x + 50, person1_y + 60), (255, 255, 255), -1)  # Shirt
            
            cv2.putText(frame, 'Person 1', (person1_x, person1_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Bag - appears at frame 100, stays stationary (potential abandoned object)
        if frame_num >= 100:
            bag_x, bag_y = 400, 200
            # Draw bag as a very realistic object
            # Main bag body
            cv2.rectangle(frame, (bag_x, bag_y), (bag_x + 60, bag_y + 45), (139, 69, 19), -1)
            cv2.rectangle(frame, (bag_x + 3, bag_y + 3), (bag_x + 57, bag_y + 42), (160, 82, 45), -1)
            
            # Bag handle (more realistic)
            cv2.ellipse(frame, (bag_x + 30, bag_y - 8), (18, 10), 0, 0, 180, (139, 69, 19), 4)
            cv2.ellipse(frame, (bag_x + 30, bag_y - 8), (18, 10), 0, 0, 180, (160, 82, 45), 2)
            
            # Add some bag details
            cv2.rectangle(frame, (bag_x + 10, bag_y + 10), (bag_x + 20, bag_y + 20), (255, 255, 255), -1)  # Zipper
            cv2.rectangle(frame, (bag_x + 35, bag_y + 15), (bag_x + 45, bag_y + 25), (255, 255, 255), -1)  # Pocket
            
            cv2.putText(frame, 'Bag', (bag_x, bag_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Person 2 - appears at frame 150, moves quickly (speed spike)
        if frame_num >= 150:
            # Fast movement for speed spike detection
            person2_x = 300 + int(100 * np.sin(frame_num * 0.5))
            person2_y = 200 + int(60 * np.cos(frame_num * 0.4))
            
            # Draw person 2 (very realistic)
            # Head
            cv2.circle(frame, (person2_x + 40, person2_y + 25), 18, (255, 180, 140), -1)
            cv2.circle(frame, (person2_x + 35, person2_y + 20), 3, (0, 0, 0), -1)  # Left eye
            cv2.circle(frame, (person2_x + 45, person2_y + 20), 3, (0, 0, 0), -1)  # Right eye
            
            # Body
            cv2.rectangle(frame, (person2_x + 25, person2_y + 43), (person2_x + 55, person2_y + 130), (200, 0, 100), -1)
            
            # Arms
            cv2.rectangle(frame, (person2_x + 15, person2_y + 50), (person2_x + 25, person2_y + 90), (255, 180, 140), -1)
            cv2.rectangle(frame, (person2_x + 55, person2_y + 50), (person2_x + 65, person2_y + 90), (255, 180, 140), -1)
            
            # Legs
            cv2.rectangle(frame, (person2_x + 30, person2_y + 130), (person2_x + 40, person2_y + 190), (150, 0, 75), -1)
            cv2.rectangle(frame, (person2_x + 40, person2_y + 130), (person2_x + 50, person2_y + 190), (150, 0, 75), -1)
            
            # Clothing details
            cv2.rectangle(frame, (person2_x + 30, person2_y + 43), (person2_x + 50, person2_y + 60), (255, 255, 255), -1)  # Shirt
            
            cv2.putText(frame, 'Person 2', (person2_x, person2_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Suitcase - appears at frame 200, stays in one place (abandoned)
        if frame_num >= 200:
            suitcase_x, suitcase_y = 150, 300
            # Draw suitcase (very realistic)
            # Main body
            cv2.rectangle(frame, (suitcase_x, suitcase_y), (suitcase_x + 80, suitcase_y + 55), (105, 105, 105), -1)
            cv2.rectangle(frame, (suitcase_x + 3, suitcase_y + 3), (suitcase_x + 77, suitcase_y + 52), (128, 128, 128), -1)
            
            # Handle (more realistic)
            cv2.rectangle(frame, (suitcase_x + 25, suitcase_y - 12), (suitcase_x + 55, suitcase_y), (64, 64, 64), -1)
            cv2.rectangle(frame, (suitcase_x + 30, suitcase_y - 10), (suitcase_x + 50, suitcase_y - 2), (80, 80, 80), -1)
            
            # Wheels (more realistic)
            cv2.circle(frame, (suitcase_x + 18, suitcase_y + 55), 10, (50, 50, 50), -1)
            cv2.circle(frame, (suitcase_x + 62, suitcase_y + 55), 10, (50, 50, 50), -1)
            cv2.circle(frame, (suitcase_x + 18, suitcase_y + 55), 8, (70, 70, 70), -1)
            cv2.circle(frame, (suitcase_x + 62, suitcase_y + 55), 8, (70, 70, 70), -1)
            
            # Add some suitcase details
            cv2.rectangle(frame, (suitcase_x + 15, suitcase_y + 15), (suitcase_x + 35, suitcase_y + 25), (255, 255, 255), -1)  # Logo
            cv2.rectangle(frame, (suitcase_x + 45, suitcase_y + 20), (suitcase_x + 65, suitcase_y + 30), (255, 255, 255), -1)  # Brand
            
            cv2.putText(frame, 'Suitcase', (suitcase_x, suitcase_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Person 3 - appears at frame 250, loiters in small area
        if frame_num >= 250:
            # Small movement in confined area (loitering behavior)
            person3_x = 500 + int(25 * np.sin(frame_num * 0.08))
            person3_y = 100 + int(20 * np.cos(frame_num * 0.12))
            
            # Draw person 3 (very realistic)
            # Head
            cv2.circle(frame, (person3_x + 40, person3_y + 25), 18, (200, 150, 100), -1)
            cv2.circle(frame, (person3_x + 35, person3_y + 20), 3, (0, 0, 0), -1)  # Left eye
            cv2.circle(frame, (person3_x + 45, person3_y + 20), 3, (0, 0, 0), -1)  # Right eye
            
            # Body
            cv2.rectangle(frame, (person3_x + 25, person3_y + 43), (person3_x + 55, person3_y + 130), (100, 100, 0), -1)
            
            # Arms
            cv2.rectangle(frame, (person3_x + 15, person3_y + 50), (person3_x + 25, person3_y + 90), (200, 150, 100), -1)
            cv2.rectangle(frame, (person3_x + 55, person3_y + 50), (person3_x + 65, person3_y + 90), (200, 150, 100), -1)
            
            # Legs
            cv2.rectangle(frame, (person3_x + 30, person3_y + 130), (person3_x + 40, person3_y + 190), (75, 75, 0), -1)
            cv2.rectangle(frame, (person3_x + 40, person3_y + 130), (person3_x + 50, person3_y + 190), (75, 75, 0), -1)
            
            # Clothing details
            cv2.rectangle(frame, (person3_x + 30, person3_y + 43), (person3_x + 50, person3_y + 60), (255, 255, 255), -1)  # Shirt
            
            cv2.putText(frame, 'Person 3', (person3_x, person3_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add some realistic objects that YOLO might detect
        if frame_num >= 120:
            # Add a backpack (very realistic)
            backpack_x, backpack_y = 450, 350
            # Main body
            cv2.rectangle(frame, (backpack_x, backpack_y), (backpack_x + 45, backpack_y + 35), (34, 139, 34), -1)
            cv2.rectangle(frame, (backpack_x + 2, backpack_y + 2), (backpack_x + 43, backpack_y + 33), (50, 205, 50), -1)
            
            # Straps
            cv2.rectangle(frame, (backpack_x + 15, backpack_y - 15), (backpack_x + 20, backpack_y), (34, 139, 34), -1)
            cv2.rectangle(frame, (backpack_x + 25, backpack_y - 15), (backpack_x + 30, backpack_y), (34, 139, 34), -1)
            
            # Pockets
            cv2.rectangle(frame, (backpack_x + 5, backpack_y + 5), (backpack_x + 15, backpack_y + 15), (255, 255, 255), -1)
            cv2.rectangle(frame, (backpack_x + 30, backpack_y + 8), (backpack_x + 40, backpack_y + 18), (255, 255, 255), -1)
            
            cv2.putText(frame, 'Backpack', (backpack_x, backpack_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add some realistic shadows and lighting effects
        if frame_num >= 50:
            # Add shadows under people
            shadow_offset = 5
            cv2.ellipse(frame, (person1_x + 40, person1_y + 190 + shadow_offset), (25, 8), 0, 0, 180, (50, 50, 50), -1)
            
            if frame_num >= 150:
                cv2.ellipse(frame, (person2_x + 40, person2_y + 190 + shadow_offset), (25, 8), 0, 0, 180, (50, 50, 50), -1)
            
            if frame_num >= 250:
                cv2.ellipse(frame, (person3_x + 40, person3_y + 190 + shadow_offset), (25, 8), 0, 0, 180, (50, 50, 50), -1)
        
        # Write frame
        out.write(frame)
        
        # Show progress
        if frame_num % 50 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"📈 Progress: {progress:.1f}% ({frame_num}/{total_frames})")
    
    # Release video writer
    out.release()
    print(f"✅ Ultra-realistic test video created: ultra_realistic_test.mp4")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Frames: {total_frames}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print("\n🎯 This video contains ultra-realistic objects:")
    print("   - Person 1: Normal movement (detailed humanoid)")
    print("   - Bag: Stationary object (detailed bag with handle)")
    print("   - Person 2: Fast movement (detailed humanoid)")
    print("   - Suitcase: Stationary object (detailed with wheels)")
    print("   - Person 3: Small area movement (detailed humanoid)")
    print("   - Backpack: Additional detailed object")
    print("   - Realistic shadows and lighting effects")

if __name__ == "__main__":
    create_ultra_realistic_video()
