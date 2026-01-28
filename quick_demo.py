"""Quick demo of dual-brain system"""
import cv2
import sys
import time

sys.path.insert(0, '.')

video_path = 'Avenue_Dataset/Avenue Dataset/testing_videos/01.avi'

print('=' * 60)
print('🧠 AbnoGuard v3.0 - Dual-Brain Intelligence System')
print('=' * 60)

# Initialize
from detector import YOLODetector
from tracker import ObjectTracker
from utils import setup_directories

setup_directories()
detector = YOLODetector()
tracker = ObjectTracker()

# Try loading intelligence
intelligence = None
try:
    from core.fusion_engine import DualBrainFusion
    intelligence = DualBrainFusion({})
    print('✅ Dual-Brain System ACTIVE')
except Exception as e:
    print(f'⚠️ Intelligence load error: {e}')
    import traceback
    traceback.print_exc()

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('❌ Could not open video')
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'📹 Video: {video_path}')
print(f'📊 Total Frames: {total}')
print('=' * 60)

# Process frames
start = time.time()
alert_count = 0
frame_count = 0
max_frames = 200

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detection + Tracking
    detections = detector.detect(frame)
    tracked = tracker.update(detections, frame)
    
    # Intelligence processing
    if intelligence:
        try:
            result = intelligence.process(frame, detections, tracked)
            
            for alert in result.final_alerts:
                alert_count += 1
                alert_type = alert.get('type', 'unknown')
                desc = alert.get('description', '')[:60]
                print(f'🚨 [{frame_count}] ALERT: {alert_type}')
                print(f'   📝 {desc}')
                
                if result.narratives:
                    narr = result.narratives[0][:100]
                    print(f'   📖 {narr}...')
                print()
                
        except Exception as e:
            if frame_count == 0:
                print(f'⚠️ Processing error: {e}')
                import traceback
                traceback.print_exc()
    
    if frame_count % 50 == 0:
        tracks = len(tracked)
        print(f'📈 Frame {frame_count}/{max_frames} | Tracks: {tracks}')
    
    frame_count += 1

elapsed = time.time() - start
print('=' * 60)
print(f'✅ Processed {frame_count} frames in {elapsed:.1f}s')
print(f'⚡ Speed: {frame_count/elapsed:.1f} FPS')
print(f'🚨 Alerts generated: {alert_count}')

if intelligence:
    status = intelligence.get_status()
    print(f'🧠 Active patterns: {status.get("active_patterns", 0)}')
    print(f'📊 Alerts suppressed: {status.get("alerts_suppressed", 0)}')
    print(f'⏱️ Avg processing: {status.get("avg_processing_time_ms", 0):.1f}ms')

cap.release()
print('=' * 60)
print('🎉 Live demo complete!')
