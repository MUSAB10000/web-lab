# person_ppe_voice.py
import os
import cv2
import gc
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from gtts import gTTS
from norfair import Detection, Tracker
import subprocess

from supabase_helper import (
    login_admin, save_video, save_person,
    save_detection, save_alert, save_clip
)

# --- Feature Toggles ---
ENABLE_PPE = True
ENABLE_REDZONE = True

# --- Login ---
email = input("Enter admin email: ")
password = input("Enter password: ")
admin_id = login_admin(email, password)
if not admin_id:
    print("❌ Login failed.")
    exit()
print(f"🧑‍💻 Logged in as Admin ID: {admin_id}")

# --- Video Setup ---
VIDEO_PATH = "test_video3.mp4"
video_response = save_video("Lab Safety Video", VIDEO_PATH, admin_id)
if not video_response:
    print("❌ Exiting: video upload failed.")
    exit()
video_id = video_response.data[0]["id"]

# --- Model Paths ---
MODELS_DIR = "models"
person_model = YOLO(os.path.join(MODELS_DIR, "yolov8n.pt"))
mask_model = YOLO(os.path.join(MODELS_DIR, "best_mask_new.pt"))
gloves_model = YOLO(os.path.join(MODELS_DIR, "best_gloves.pt"))
labcoat_model = YOLO(os.path.join(MODELS_DIR, "best_labcoat.pt"))
glasses_model = YOLO(os.path.join(MODELS_DIR, "best_glasses.pt"))

# --- Red Zone ---
RED_ZONE = ((700, 800), (600, 700))

# --- Paths ---
ANNOTATED_VIDEO = "annotated_video3.1.mp4"
CLIPS_DIR = "clips"
VOICES_DIR = "voices"
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

# --- Trackers and State ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = 1280, 720
out = cv2.VideoWriter(ANNOTATED_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
tracker = Tracker(distance_function="euclidean", distance_threshold=30)
frame_count = 0

person_states, unsafe_start, red_start = {}, {}, {}
last_voice_time = {}

print("🎥 Starting...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    frame_count += 1
    timestamp = datetime.now().isoformat()

    # Detect persons
    results = person_model(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.7:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            detections.append(Detection(points=np.array([cx, cy]), scores=np.array([box.conf[0]]), data=(x1, y1, x2, y2)))

    tracked = tracker.update(detections)
    active_ids = set()

    for t in tracked:
        track_id = t.id
        active_ids.add(track_id)
        x1, y1, x2, y2 = map(int, t.last_detection.data)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # PPE Checks
        has_mask = len(mask_model(crop, verbose=False)[0].boxes) > 0 if ENABLE_PPE else True
        has_gloves = len(gloves_model(crop, verbose=False)[0].boxes) > 0 if ENABLE_PPE else True
        has_labcoat = len(labcoat_model(crop, verbose=False)[0].boxes) > 0 if ENABLE_PPE else True
        has_glasses = len(glasses_model(crop, verbose=False)[0].boxes) > 0 if ENABLE_PPE else True
        all_ok = has_mask and has_gloves and has_labcoat and has_glasses

        in_red = RED_ZONE[0][0] < cx < RED_ZONE[1][0] and RED_ZONE[0][1] < cy < RED_ZONE[1][1] if ENABLE_REDZONE else False

        if track_id not in person_states:
            person_resp = save_person(video_id, track_id, frame_count, has_mask, has_gloves, has_labcoat, has_glasses, in_red, "unsafe" if not all_ok else "safe")
            person_states[track_id] = person_resp.data[0]["id"]

        # Save detection
        save_detection(person_id=None, class_name="ppe_check", confidence=1.0, bbox_dict={"x1": x1, "y1": y1, "x2": x2, "y2": y2}, frame_path=VIDEO_PATH)

        now = datetime.now()

        # --- Unsafe Logic ---
        if not all_ok and ENABLE_PPE:
            if track_id not in unsafe_start:
                unsafe_start[track_id] = now
            elif (now - unsafe_start[track_id]).total_seconds() > 2:
                if track_id not in last_voice_time or (now - last_voice_time[track_id]).total_seconds() >= 1:
                    missing = []
                    if not has_mask: missing.append("mask")
                    if not has_gloves: missing.append("gloves")
                    if not has_labcoat: missing.append("lab coat")
                    if not has_glasses: missing.append("glasses")
                    msg = f"Alert: Missing {', '.join(missing)}"
                    voice_file = os.path.join(VOICES_DIR, f"frame_{frame_count}_ppe.mp3")
                    gTTS(text=msg).save(voice_file)
                    last_voice_time[track_id] = now

        # --- Red Zone Logic ---
        if in_red and ENABLE_REDZONE:
            if track_id not in red_start:
                red_start[track_id] = now
            elif (now - red_start[track_id]).total_seconds() > 5:
                if track_id not in last_voice_time or (now - last_voice_time[track_id]).total_seconds() >= 1:
                    msg = "Red zone violation"
                    voice_file = os.path.join(VOICES_DIR, f"frame_{frame_count}_red.mp3")
                    gTTS(text=msg).save(voice_file)
                    last_voice_time[track_id] = now

        # Draw box
        color = (0, 255, 0) if all_ok else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Status: {'Safe' if all_ok else 'Unsafe'}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw red zone
    if ENABLE_REDZONE:
        cv2.rectangle(frame, RED_ZONE[0], RED_ZONE[1], (0, 0, 255), 2)

    out.write(frame)
    print(f"🟢 Frame {frame_count}")

cap.release()
out.release()
gc.collect()

# --- Merge audio using ffmpeg ---
import subprocess
import os

import subprocess
import os

print("🎬 Embedding audio into final video...")

final_video_input = "annotated_video.mp4"  # your annotated output video
final_audio_input = os.path.join("voices", "merged_alerts.mp3")
final_output_video = "annotated_video_with_audio.mp4"

try:
    subprocess.run([
        "ffmpeg", "-y",
        "-i", final_video_input,
        "-i", final_audio_input,
        "-c:v", "copy",
        "-c:a", "aac",  # convert MP3 to AAC for MP4 compatibility
        "-shortest",    # stop when shortest stream ends
        final_output_video
    ], check=True)
    print(f"✅ Final video with voice alerts saved as: {final_output_video}")
except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg merge failed: {e}")

