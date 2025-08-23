#!/usr/bin/env python3
"""
Basic YOLOv8 Test Script
Tests if YOLOv8 can detect basic objects in a simple image.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_yolo_basic():
    """Test YOLOv8 with a simple generated image"""
    
    print("🧪 Testing YOLOv8 Basic Detection...")
    
    # Check if model exists or download it
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print(f"📥 YOLOv8 model not found. Will download: {model_path}")
    else:
        print(f"✅ Found existing YOLOv8 model: {model_path}")
    
    try:
        # Load model
        print("🤖 Loading YOLOv8 model...")
        model = YOLO(model_path)  # This will auto-download if needed
        print("✅ Model loaded successfully")
        
        # Create a test image
        print("📸 Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some shapes that might be detected
        # Blue rectangle (person-like)
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 0, 0), -1)
        
        # Green rectangle (bag-like)
        cv2.rectangle(test_image, (450, 200), (550, 300), (0, 255, 0), -1)
        
        # Red circle (object-like)
        cv2.circle(test_image, (100, 100), 50, (0, 0, 255), -1)
        
        print("✅ Test image created with shapes")
        
        # Save test image
        cv2.imwrite('test_debug_image.jpg', test_image)
        print("💾 Test image saved as 'test_debug_image.jpg'")
        
        # Run detection with lower confidence for testing
        print("🔍 Running YOLOv8 detection...")
        results = model(test_image, verbose=False, conf=0.25)
        
        print(f"📊 Detection Results:")
        print(f"   - Number of results: {len(results)}")
        
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                print(f"   - Number of detections: {len(result.boxes)}")
                
                if len(result.boxes) > 0:
                    print(f"   - Detections found:")
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_names = model.names
                        class_name = class_names[class_id] if class_id in class_names else f"class_{class_id}"
                        
                        print(f"      {i+1}. {class_name} (confidence: {conf:.2f})")
                        print(f"         Bounding box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
                        
                        # Check if this is a target class
                        target_classes = ['person', 'backpack', 'handbag', 'suitcase', 'bag', 'book', 'cell phone', 'laptop']
                        if class_name in target_classes:
                            print(f"         🎯 TARGET CLASS DETECTED!")
                else:
                    print("   ❌ No detections found")
            else:
                print("   ❌ No detection boxes")
        else:
            print("   ❌ No results from model")
        
        # Show the image with detections
        print("\n🖼️  Displaying image with detections...")
        try:
            if len(results) > 0 and results[0].boxes is not None:
                annotated_frame = results[0].plot()
                cv2.imshow('YOLOv8 Test Results', annotated_frame)
                print("   ✅ Press any key to close the image window")
                cv2.waitKey(0)
            else:
                cv2.imshow('Test Image (No Detections)', test_image)
                print("   ✅ Press any key to close the image window")
                cv2.waitKey(0)
        except Exception as e:
            print(f"   ❌ Error displaying image: {e}")
        
        cv2.destroyAllWindows()
        
        print("\n🎯 Test complete!")
        print("💡 YOLOv8 should provide better detection than YOLOv5")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_basic()
