"""
LAYER 1: PERCEPTION
Extends YOLOv8 detection with tracking, motion, and pose estimation.
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

class VisionEngine:
    def __init__(self, model_path='yolov8n.pt'):
        print("👁️ Initializing Perception Layer...")
        # Face detection model
        self.model = YOLO(model_path)
        
        # Pose Estimator
        self.pose = None
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("✅ Pose Estimation Initialized")
        except Exception as e:
            print(f"⚠️ Pose Estimation failed to load (Optional): {e}")

        # Feature Extractors
        self.prev_gray = None
        self.tracks = {}  # ID -> {history, last_seen, velocity}
        
    def process_frame(self, frame):
        """
        Process a single frame through the perception stack.
        Returns: Analyzed Perception Object
        """
        t0 = time.time()
        height, width = frame.shape[:2]
        
        # 1. YOLO Detection & Tracking
        results = self.model.track(frame, persist=True, verbose=False)[0]
        
        entities = []
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            classes = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                class_name = self.model.names[cls]
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2)//2, (y1 + y2)//2)
                
                # 2. Motion Analysis (Velocity)
                velocity = (0, 0)
                if track_id in self.tracks:
                    prev_center = self.tracks[track_id]['last_pos']
                    velocity = (center[0] - prev_center[0], center[1] - prev_center[1])
                
                # Update track history
                if track_id not in self.tracks:
                    self.tracks[track_id] = {'history': [], 'last_pos': center, 'start_time': time.time()}
                self.tracks[track_id]['history'].append(center)
                self.tracks[track_id]['last_pos'] = center
                self.tracks[track_id]['last_seen'] = time.time()
                
                # 3. Pose Estimation (If Person)
                pose_data = None
                if class_name == 'person':
                    pose_data = self._analyze_pose(frame[y1:y2, x1:x2])
                
                entities.append({
                    'id': int(track_id),
                    'class': class_name,
                    'bbox': [x1, y1, x2, y2],
                    'center': center,
                    'confidence': float(conf),
                    'velocity': velocity,
                    'speed': np.linalg.norm(velocity),
                    'pose': pose_data
                })
                
        return {
            'frame_time': time.time(),
            'processing_time': time.time() - t0,
            'entities': entities,
            'frame_dim': (width, height)
        }

    def _analyze_pose(self, crop):
        """Lightweight pose analysis on crop"""
        if crop.shape[0] < 10 or crop.shape[1] < 10: return None
        if self.pose is None: return None
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            results = self.pose.process(rgb)
        except:
            return None
        
        if not results.pose_landmarks:
            return None
            
        # Basic heuristic: Hands up?
        landmarks = results.pose_landmarks.landmark
        # 15=left wrist, 16=right wrist, 11=left shoulder, 12=right shoulder
        wrists_y = (landmarks[15].y + landmarks[16].y) / 2
        shoulders_y = (landmarks[11].y + landmarks[12].y) / 2
        
        state = 'neutral'
        if wrists_y < shoulders_y: # Y is inverted in image coords
            state = 'hands_up'
            
        return {'state': state, 'stress_score': 0.0}
