#!/usr/bin/env python3
"""
YOLOv8 Object Detector for AbnoGuard
Handles object detection using YOLOv8 model
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', device='auto', conf_threshold=0.25):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model file
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Auto-download YOLOv8 model if not present
        if not os.path.exists(model_path):
            print(f"📥 Downloading YOLOv8 model: {model_path}")
            try:
                model = YOLO(model_path)  # This will auto-download
                print(f"✅ YOLOv8 model downloaded successfully: {model_path}")
            except Exception as e:
                print(f"❌ Error downloading YOLOv8 model: {e}")
                # Fallback to nano model
                model_path = 'yolov8n.pt'
                model = YOLO(model_path)
                print(f"✅ Using fallback YOLOv8n model")
        else:
            print(f"🤖 Loading existing YOLOv8 model: {model_path}")
            model = YOLO(model_path)
        
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🚀 YOLOv8 running on: {self.device}")
        
        # Target classes for surveillance
        self.target_classes = ['person', 'backpack', 'handbag', 'suitcase', 'bag', 'book', 'cell phone', 'laptop']
        self.target_class_ids = []
        
        # Get class names from model
        self.class_names = self.model.names
        
        # Map target class names to IDs
        for class_name in self.target_classes:
            for class_id, name in self.class_names.items():
                if name == class_name:
                    self.target_class_ids.append(class_id)
                    break
        
        print(f"🎯 Target classes: {self.target_classes}")
        print(f"🎯 Target class IDs: {self.target_class_ids}")
        
        # Warm up the model
        print("🔥 Warming up YOLOv8 model...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_input, verbose=False)
        print("✅ Model warmed up and ready!")
    
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections: [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False, conf=self.conf_threshold)
            
            detections = []
            all_detections = []  # For debugging
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.class_names[class_id] if class_id in self.class_names else f"class_{class_id}"
                        all_detections.append([x1, y1, x2, y2, conf, class_id, class_name])
                        
                        if class_id in self.target_class_ids:
                            detections.append([x1, y1, x2, y2, conf, class_id, class_name])
            
            # Debug output
            if all_detections:
                print(f"🔍 Frame detections: {len(all_detections)} total, {len(detections)} target classes")
                for det in all_detections[:3]:  # Show first 3 detections
                    x1, y1, x2, y2, conf, class_id, class_name = det
                    print(f"   - {class_name} (conf: {conf:.2f}) at ({x1:.0f}, {y1:.0f})")
            elif len(detections) == 0:
                print(f"🔍 Frame: No detections")
                
            return detections
            
        except Exception as e:
            print(f"❌ Error in YOLOv8 detection: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id, class_name = det
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green for person
            elif class_name in ['backpack', 'handbag', 'suitcase', 'bag']:
                color = (255, 0, 0)  # Blue for bags
            else:
                color = (0, 0, 255)  # Red for other objects
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'YOLOv8',
            'device': self.device,
            'target_classes': self.target_classes,
            'confidence_threshold': self.conf_threshold,
            'total_classes': len(self.class_names)
        }

