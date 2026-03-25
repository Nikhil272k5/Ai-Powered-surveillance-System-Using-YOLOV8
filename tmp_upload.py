"""Upload video to AbnoGuard dashboard and check status"""
import urllib.request
import json
import time

# 1. Check server is alive
print("Checking server...")
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5) as r:
        status = json.loads(r.read())
        print("Server OK:", status)
except Exception as e:
    print("Server not reachable:", e)
    exit(1)

# 2. Upload video using multipart
print("\nUploading video...")
video_path = r'uploads/1130(1).mp4'
url = 'http://127.0.0.1:8000/api/upload'
boundary = 'WebKitFormBoundary7MA4YWxkTrZu0gW'

with open(video_path, 'rb') as f:
    file_data = f.read()

part_header = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="file"; filename="1130(1).mp4"\r\n'
    'Content-Type: video/mp4\r\n'
    '\r\n'
).encode()
part_footer = ('\r\n--' + boundary + '--\r\n').encode()
body = part_header + file_data + part_footer

req = urllib.request.Request(url, data=body, method='POST')
req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)
req.add_header('Content-Length', str(len(body)))

try:
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
        print("Upload response:", resp)
except Exception as e:
    print("Upload failed:", e)
    exit(1)

# 3. Poll status a few times to show live data
print("\n--- LIVE STATUS (every 2s for 20s) ---")
for i in range(10):
    time.sleep(2)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5) as r:
            s = json.loads(r.read())
            print(f"[{i*2:2d}s] Processing={s['processing']} | FPS={s['fps']} | Tracks={s['tracks']} | Anomaly={s['anomaly']} | Risk={s['risk']}")
    except Exception as e:
        print(f"[{i*2:2d}s] Error:", e)

print("\nDone. Server is live at http://127.0.0.1:8000")
