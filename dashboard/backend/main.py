"""
DASHBOARD BACKEND - Waits for User Video Upload
No default video - only processes what user uploads
"""
from fastapi import FastAPI, WebSocket, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import asyncio
import os
import sys
import cv2
import base64
import time
import threading
import shutil
from datetime import datetime
from collections import deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")

# ============================================
# GLOBAL STATE
# ============================================
class State:
    current_frame = None
    frame_lock = threading.Lock()
    fps = 0
    tracks = 0
    anomaly = 0.0
    risk = 0.0
    situation = ""
    briefings = deque(maxlen=50)
    is_processing = False
    current_video = None
    processor_thread = None

state = State()

# ============================================
# VIDEO PROCESSOR (runs in separate thread)
# ============================================
def process_video(video_path):
    """Process uploaded video - runs in background thread"""
    from system.perception.vision_engine import VisionEngine
    from system.memory.temporal_store import TemporalMemory
    from system.learning.normality_engine import NormalityEngine
    from system.reasoning.cognitive_engine import CognitiveEngine
    
    print(f"🎬 Starting analysis: {video_path}")
    
    vision = VisionEngine()
    memory = TemporalMemory()
    learning = NormalityEngine()
    cognition = CognitiveEngine()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        state.is_processing = False
        return
    
    state.is_processing = True
    frame_count = 0
    fps_start = time.time()
    fps_counter = 0
    last_screenshot_time = 0
    briefing_count = 0
    
    print("▶ Processing...")
    
    try:
        while state.is_processing:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                state.fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
            perception = vision.process_frame(frame)
            
            state.tracks = len(perception['entities'])
            
            lines = [f"📊 FRAME #{frame_count}", f"Time: {datetime.now().strftime('%H:%M:%S')}", f"Objects: {len(perception['entities'])}", "-" * 35]
            
            max_anomaly = 0.0
            max_risk = 0.0
            
            for entity in perception['entities']:
                memory.update_entity(entity['id'], entity['class'])
                
                bbox = entity['bbox']
                features = {'speed': entity['speed'], 'size': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])}
                learning.update(features)
                anomaly = learning.get_anomaly_score(features)
                
                context = {'history': memory.get_recent_events(5)}
                cognitive = cognition.analyze_situation(entity, context, anomaly)
                
                lines.append(f"\n🔹 ID:{entity['id']} ({entity['class']})")
                lines.append(f"   Behavior: {cognitive['intent'].upper()}")
                lines.append(f"   Confidence: {cognitive['intent_confidence']:.0%}")
                lines.append(f"   Anomaly: {anomaly:.2f}")
                lines.append(f"   Risk: {cognitive['risk_score']:.2f}")
                lines.append(f"   Reason: {cognitive['justification']}")
                
                max_anomaly = max(max_anomaly, anomaly)
                max_risk = max(max_risk, cognitive['risk_score'])
                
                # Generate detailed briefing for significant events
                if anomaly > 0.3 or cognitive['risk_score'] > 0.3:
                    briefing_count += 1
                    
                    # Determine severity level
                    if cognitive['risk_score'] > 0.7:
                        level = 'high'
                        action = 'IMMEDIATE REVIEW REQUIRED - Potential security threat detected'
                    elif cognitive['risk_score'] > 0.4:
                        level = 'warning'
                        action = 'Monitor closely - Unusual behavior pattern observed'
                    else:
                        level = 'info'
                        action = 'Continue monitoring - Minor deviation from normal'
                    
                    # Build comprehensive briefing content
                    content = f"""
<strong>🎯 Subject Information</strong><br>
- ID: {entity['id']}<br>
- Type: {entity['class'].upper()}<br>
- Location: Frame region ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})<br><br>

<strong>🧠 Behavioral Analysis</strong><br>
- Detected Behavior: <span style="color: {'#f85149' if cognitive['risk_score'] > 0.5 else '#d29922'}">{cognitive['intent'].upper()}</span><br>
- Confidence: {cognitive['intent_confidence']:.0%}<br>
- Reasoning: {cognitive['justification']}<br><br>

<strong>📊 Risk Assessment</strong><br>
- Anomaly Score: {anomaly:.2f} / 1.00<br>
- Risk Level: {cognitive['risk_score']:.2f} / 1.00<br>
- Threat Classification: {level.upper()}<br><br>

<strong>⚡ Recommended Action</strong><br>
{action}
"""
                    
                    state.briefings.appendleft({
                        'id': briefing_count,
                        'level': level,
                        'title': f"#{briefing_count}: {cognitive['intent'].upper()} - Subject {entity['id']}",
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'content': content.strip()
                    })
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                color = (0,0,255) if cognitive['risk_score'] > 0.7 else ((0,165,255) if cognitive['risk_score'] > 0.4 else (0,255,0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{entity['id']} {entity['class']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, cognitive['intent'][:10].upper(), (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            state.anomaly = max_anomaly
            state.risk = max_risk
            state.situation = "\n".join(lines)
            
            # Auto-screenshot on anomaly (separate from briefings)
            if (max_anomaly > 0.5 or max_risk > 0.5) and (time.time() - last_screenshot_time > 5):
                reason = "high-risk" if max_risk > 0.7 else "anomaly"
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{reason}.jpg"
                cv2.imwrite(os.path.join(SCREENSHOTS_DIR, filename), frame)
                last_screenshot_time = time.time()
                print(f"📸 Auto-screenshot: {filename}")
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with state.frame_lock:
                state.current_frame = buffer.tobytes()
            
            frame_count += 1
            time.sleep(0.02)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        state.is_processing = False
        print("✅ Processing complete")

# ============================================
# MJPEG STREAMING
# ============================================
def gen_mjpeg():
    while True:
        with state.frame_lock:
            frame = state.current_frame
        if frame:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        else:
            # Black placeholder when no video
            placeholder = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f\xff\xd9'
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n'
        time.sleep(0.1)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_mjpeg(), media_type='multipart/x-mixed-replace; boundary=frame')

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/", response_class=HTMLResponse)
async def root():
    html = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html):
        with open(html, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Loading...</h1>")

@app.post("/api/upload")
async def upload_and_start(file: UploadFile = File(...)):
    """Upload video and START processing immediately"""
    # Stop any current processing
    state.is_processing = False
    if state.processor_thread and state.processor_thread.is_alive():
        time.sleep(0.5)
    
    # Save uploaded file
    filepath = os.path.join(UPLOADS_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    state.current_video = filepath
    print(f"📁 Uploaded: {filepath}")
    
    # Clear old frame
    with state.frame_lock:
        state.current_frame = None
    
    # Start processing in background thread
    state.processor_thread = threading.Thread(target=process_video, args=(filepath,), daemon=True)
    state.processor_thread.start()
    
    return {"status": "ok", "path": filepath, "message": "Processing started"}

@app.post("/api/stop")
async def stop_processing():
    """Stop current processing"""
    state.is_processing = False
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    return {
        "processing": state.is_processing,
        "video": state.current_video,
        "fps": state.fps,
        "tracks": state.tracks,
        "anomaly": round(state.anomaly, 2),
        "risk": round(state.risk, 2)
    }

@app.get("/api/screenshots")
async def get_screenshots():
    shots = []
    if os.path.exists(SCREENSHOTS_DIR):
        for f in sorted(os.listdir(SCREENSHOTS_DIR), reverse=True)[:30]:
            if f.endswith(('.jpg', '.png')):
                fpath = os.path.join(SCREENSHOTS_DIR, f)
                parts = f.replace('.jpg', '').split('_')
                reason = parts[-1] if len(parts) > 3 else 'anomaly'
                shots.append({
                    "url": f"/screenshots/{f}",
                    "time": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%H:%M:%S'),
                    "reason": reason.replace('-', ' ').title(),
                    "auto": True
                })
    return shots

@app.get("/api/briefings")
async def get_briefings():
    return list(state.briefings)

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(0.5)
            await ws.send_json({
                "type": "stats",
                "fps": state.fps,
                "tracks": state.tracks,
                "anomaly": state.anomaly,
                "risk": state.risk,
                "situation": state.situation,
                "processing": state.is_processing
            })
    except:
        pass

def start_server(host="127.0.0.1", port=8000):
    print(f"🌐 Dashboard ready at http://{host}:{port}")
    print("⏳ Waiting for video upload...")
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    start_server()
