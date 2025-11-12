# File: monitor_classroom_web_v12_optimized.py test github
# OPTIMIZED VERSION - Auto-monitoring, Better Performance, Fixed Detection
# By Tim Iot - Gustia Fernando, Rijalul Fahmi

import sys
import subprocess
import importlib.util

def check_and_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[INFO] Installing '{package_name}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
            return True
        except:
            return False
    return True

print("\n" + "="*70)
print("VERSION 12.0 OPTIMIZED - AUTO MONITORING + FIXED DETECTION!")
print("="*70 + "\n")

required_packages = [
    ('flask', 'flask'),
    ('flask-socketio', 'flask_socketio'),
    ('opencv-python', 'cv2'),
    ('face-recognition', 'face_recognition'),
    ('numpy', 'numpy'),
    ('requests', 'requests'),
    ('imutils', 'imutils'),
    ('mediapipe', 'mediapipe'),
    ('Pillow', 'PIL'),
    ('ultralytics', 'ultralytics'),
]

all_installed = True
for package, import_name in required_packages:
    if not check_and_install(package, import_name):
        all_installed = False

if not all_installed:
    sys.exit(1)

print("="*70 + "\n")

import cv2
import face_recognition
import numpy as np
import time
import datetime
import os
import pickle
import shutil
from flask import Flask, Response, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
import json
import requests as req
import threading
from imutils.video import VideoStream
import imutils
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
from PIL import Image
import io
import base64
import math
import queue

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics YOLO not available - body detection disabled")

# ==================== CONFIGURATION ====================
YOLO_FACE_MODEL_PATH = "yolo_models/face_detection_yunet_2023mar.onnx"
YOLO_BODY_MODEL_PATH = "yolo_models/yolov8n.pt"
DATASET_PATH = "dataset"
TRAINING_PATH = "training_data"
WHATSAPP_API_URL = "https://mpwa.genesis.my.id/send-message"
WHATSAPP_API_KEY = "0nAJJT4NdojWIecDaK58jiAo1AzoFo"
WHATSAPP_SENDER = "6285709817554"
WHATSAPP_ADMIN_NUMBER = "6285117080314"

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.42

# OPTIMIZED SETTINGS untuk RPI
PROCESS_EVERY_N_FRAMES = 2  # Dikurangi dari 1 ke 2 untuk mengurangi beban
FACE_RECOGNITION_SKIP = 3  # Dikurangi frekuensi recognisi
MONITORING_INTERVAL = 3  # Dikurangi dari 5 ke 3 untuk lebih responsif
HAND_DETECTION_SKIP = 2  # Tetap 2 untuk keseimbangan
EMOTION_DETECTION_SKIP = 4  # Baru: Skip emotion detection untuk efisiensi
RELOAD_FACES_INTERVAL = 300
STUDENT_MISSING_THRESHOLD = 8  # Dikurangi dari 10 untuk lebih cepat deteksi
STUDENT_RETURN_THRESHOLD = 3  # Baru: Threshold untuk confirm student kembali

TRAINING_CONFIDENCE_THRESHOLD = 0.40
FACE_VERIFY_FRAMES = 2

JPEG_QUALITY_LOCAL = 60  # Dikurangi dari 65 untuk performa
JPEG_QUALITY_TUNNEL = 25  # Dikurangi dari 30
WHATSAPP_COOLDOWN_SECONDS = 300
MAX_QUEUE_SIZE = 1  # Dikurangi dari 2
MAX_WORKERS = 1  # Dikurangi dari 2 untuk menghemat resource

# Hand detection thresholds - IMPROVED
HAND_DETECTION_CONFIDENCE = 0.6  # Lebih tinggi untuk akurasi
HAND_TEMPORAL_FRAMES = 3  # Butuh 3 frame konsisten
HAND_ABOVE_FACE_THRESHOLD = 0.15  # 15% dari tinggi wajah
HAND_HORIZONTAL_THRESHOLD = 1.5  # 1.5x lebar wajah

# Emotion detection thresholds - IMPROVED
EMOTION_HISTORY_SIZE = 3  # Dikurangi dari 5 untuk lebih responsif
EMOTION_CHANGE_THRESHOLD = 2  # Butuh 2 deteksi konsisten untuk ganti emosi

# Coverage metrics
COVERAGE_WINDOW_SIZE = 100  # Window untuk hitung metrik

# Multi-angle capture settings
CAPTURE_STAGES = {
    'frontal': {
        'name': 'Wajah Depan',
        'icon': '👤',
        'frames': 15,
        'duration': 15,
        'instruction': 'Hadap langsung ke kamera, jangan bergerak'
    },
    'left': {
        'name': 'Wajah Kiri',
        'icon': '👈',
        'frames': 15,
        'duration': 15,
        'instruction': 'Putar kepala ke KIRI (profile kiri), tahan'
    },
    'right': {
        'name': 'Wajah Kanan',
        'icon': '👉',
        'frames': 15,
        'duration': 15,
        'instruction': 'Putar kepala ke KANAN (profile kanan), tahan'
    },
    'vertical': {
        'name': 'Atas/Bawah',
        'icon': '👆👇',
        'frames': 15,
        'duration': 15,
        'instruction': 'Nunduk dan lihat ke atas secara bergantian'
    }
}

STAGE_ORDER = ['frontal', 'left', 'right', 'vertical']
TOTAL_CAPTURE_FRAMES = 60
MIN_FRAMES_PER_STAGE = 15

# ==================== FLASK & SOCKETIO ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                     ping_timeout=120, ping_interval=25, max_http_buffer_size=10000000)

# ==================== GLOBAL VARIABLES ====================
trained_models = {}
trained_models_lock = threading.Lock()

todays_attendance = {}
students_raising_hand = {}  # Changed to dict for better tracking
students_drowsy = {}
students_missing = {}  # Changed to dict for better tracking
students_returned = {}  # New: Track students yang kembali
system_active = True  # AUTO START - langsung aktif

display_frame = None
display_lock = threading.Lock()
encoded_frame_queue = deque(maxlen=MAX_QUEUE_SIZE)
encode_lock = threading.Lock()
vs = None
face_detector = None
body_detector = None
current_fps = 0
fps_counter = 0
fps_start_time = time.time()

cpu_temp = 0.0
fan_speed = "Auto"
whatsapp_last_sent = {}
emotion_history = {}
face_verification = {}
frame_cache = {}
last_monitoring_check = 0
last_faces_reload_time = 0

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands_detector = None
face_mesh_detector = None
hand_detection_history = {}

# Capture mode variables
capture_mode = False
capture_session_id = None
capture_buffer = {}
capture_current_stage = None
capture_stage_started = False
capture_stage_start_time = 0
capture_countdown_active = False
capture_lock = threading.Lock()

body_trackers = {}
next_tracker_id = 0

# ==================== COVERAGE METRICS ====================
coverage_metrics = {
    'total_frames': 0,
    'frames_with_faces': 0,
    'total_faces_detected': 0,
    'total_recognitions': 0,
    'successful_recognitions': 0,
    'hand_detections': 0,
    'emotion_detections': 0,
    'avg_confidence': deque(maxlen=COVERAGE_WINDOW_SIZE),
    'fps_history': deque(maxlen=COVERAGE_WINDOW_SIZE),
    'detection_times': deque(maxlen=COVERAGE_WINDOW_SIZE),
}
coverage_lock = threading.Lock()

# ==================== UTILITY FUNCTIONS ====================
def log(category, message):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] [{category}] {message}")
    try:
        socketio.emit('system_log', {
            'time': timestamp,
            'category': category,
            'message': message
        })
    except:
        pass

def get_cpu_temperature():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000.0
        return round(temp, 1)
    except:
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'],
                                  capture_output=True, text=True, timeout=2)
            temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
            return round(float(temp_str), 1)
        except:
            return 0.0

def get_fan_status():
    try:
        with open('/sys/class/thermal/cooling_device0/cur_state', 'r') as f:
            state = int(f.read().strip())
        levels = ["Off", "Very Low", "Low", "Medium", "High"]
        return levels[min(state, 4)]
    except:
        return "Auto"

def update_coverage_metrics(faces_detected, recognitions, successful_recog, hand_detected, emotion_detected, confidence, detection_time):
    """Update coverage metrics"""
    with coverage_lock:
        coverage_metrics['total_frames'] += 1
        if faces_detected > 0:
            coverage_metrics['frames_with_faces'] += 1
        coverage_metrics['total_faces_detected'] += faces_detected
        coverage_metrics['total_recognitions'] += recognitions
        coverage_metrics['successful_recognitions'] += successful_recog
        if hand_detected:
            coverage_metrics['hand_detections'] += 1
        if emotion_detected:
            coverage_metrics['emotion_detections'] += 1
        if confidence > 0:
            coverage_metrics['avg_confidence'].append(confidence)
        coverage_metrics['fps_history'].append(current_fps)
        coverage_metrics['detection_times'].append(detection_time)

def get_coverage_stats():
    """Calculate coverage statistics"""
    with coverage_lock:
        total = coverage_metrics['total_frames']
        if total == 0:
            return {}
        
        return {
            'total_frames': total,
            'face_detection_rate': round((coverage_metrics['frames_with_faces'] / total) * 100, 2),
            'avg_faces_per_frame': round(coverage_metrics['total_faces_detected'] / total, 2),
            'recognition_rate': round((coverage_metrics['successful_recognitions'] / max(1, coverage_metrics['total_recognitions'])) * 100, 2),
            'hand_detection_rate': round((coverage_metrics['hand_detections'] / total) * 100, 2),
            'emotion_detection_rate': round((coverage_metrics['emotion_detections'] / total) * 100, 2),
            'avg_confidence': round(np.mean(coverage_metrics['avg_confidence']) if len(coverage_metrics['avg_confidence']) > 0 else 0, 2),
            'avg_fps': round(np.mean(coverage_metrics['fps_history']) if len(coverage_metrics['fps_history']) > 0 else 0, 2),
            'avg_detection_time': round(np.mean(coverage_metrics['detection_times']) * 1000 if len(coverage_metrics['detection_times']) > 0 else 0, 2),  # ms
        }

def play_beep(frequency=1000, duration=0.2):
    """Play audio beep for feedback"""
    try:
        socketio.emit('play_beep', {
            'frequency': frequency,
            'duration': duration
        })
    except:
        pass

def send_whatsapp_with_json(message_text, json_data):
    try:
        full_message = f"{message_text}\n\n{'='*35}\n📊 RAW JSON:\n{'='*35}\n"
        full_message += json.dumps(json_data, indent=2, ensure_ascii=False)
        
        params = {
            'api_key': WHATSAPP_API_KEY,
            'sender': WHATSAPP_SENDER,
            'number': WHATSAPP_ADMIN_NUMBER,
            'message': full_message,
            'footer': 'Classroom v12.0'
        }
        response = req.get(WHATSAPP_API_URL, params=params, timeout=10)
        log("WHATSAPP", f"Sent notification: {message_text}")
        return response.status_code == 200
    except Exception as e:
        log("WHATSAPP", f"Error: {e}")
        return False

def can_send_whatsapp(student_id, event_type):
    global whatsapp_last_sent
    key = f"{student_id}_{event_type}"
    current_time = time.time()
    
    if key in whatsapp_last_sent:
        if current_time - whatsapp_last_sent[key] < WHATSAPP_COOLDOWN_SECONDS:
            return False
    
    whatsapp_last_sent[key] = current_time
    return True

def adaptive_brightness_enhancement_v2(frame):
    """Optimized brightness enhancement"""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg_brightness = np.mean(l)
        
        # Simplified CLAHE dengan parameter yang lebih efisien
        if avg_brightness < 100:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return enhanced, avg_brightness
    except Exception as e:
        return frame, 128

def upscale_small_face(face_roi):
    """Upscale small face ROI"""
    try:
        h, w = face_roi.shape[:2]
        if h < 80 or w < 80:
            scale_factor = max(120.0 / h, 120.0 / w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            face_roi = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return face_roi
    except:
        return face_roi

# ==================== HEAD POSE & VALIDATION ====================
def calculate_head_pose(face_landmarks):
    """Calculate head pose (yaw, pitch, roll) from face landmarks"""
    try:
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_x = nose_tip.x
        yaw = (nose_x - eye_center_x) * 180
        
        nose_y = nose_tip.y
        eye_center_y = (left_eye.y + right_eye.y) / 2
        pitch = (nose_y - eye_center_y) * 180
        
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        roll = math.degrees(math.atan2(dy, dx))
        
        return yaw, pitch, roll
    except Exception as e:
        return 0, 0, 0

def validate_capture_angle(frame, stage):
    """Validate if current frame matches the required angle for the stage"""
    global face_mesh_detector
    
    if face_mesh_detector is None:
        return True, 1.0, "No validation (face mesh disabled)"
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, "No face detected"
        
        face_landmarks = results.multi_face_landmarks[0]
        yaw, pitch, roll = calculate_head_pose(face_landmarks)

        if stage == 'frontal':
            if abs(yaw) < 20 and abs(pitch) < 25:
                confidence = 1.0 - (abs(yaw) + abs(pitch)) / 100
                return True, confidence, f"✓ Frontal OK (yaw:{yaw:.1f}° pitch:{pitch:.1f}°)"
            else:
                return False, 0.5, f"✗ Hadap ke kamera! (yaw:{yaw:.1f}° pitch:{pitch:.1f}°)"
        
        elif stage == 'left':
            if yaw > 25 and yaw < 90:
                confidence = min(1.0, yaw / 60)
                return True, confidence, f"✓ Left profile OK (yaw:{yaw:.1f}°)"
            else:
                return False, 0.3, f"✗ Putar kepala ke KIRI lebih jauh! (yaw:{yaw:.1f}°)"
        
        elif stage == 'right':
            if yaw < -25 and yaw > -90:
                confidence = min(1.0, abs(yaw) / 60)
                return True, confidence, f"✓ Right profile OK (yaw:{yaw:.1f}°)"
            else:
                return False, 0.3, f"✗ Putar kepala ke KANAN lebih jauh! (yaw:{yaw:.1f}°)"
        
        elif stage == 'vertical':
            if abs(pitch) > 15:
                confidence = min(1.0, abs(pitch) / 40)
                return True, confidence, f"✓ Vertical OK (pitch:{pitch:.1f}°)"
            else:
                return False, 0.3, f"✗ Nunduk atau lihat ke atas! (pitch:{pitch:.1f}°)"
        
        return True, 0.5, "Unknown stage"
        
    except Exception as e:
        return True, 0.5, "Validation error - accepting frame"

# ==================== BODY DETECTION ====================
def detect_bodies(frame):
    """Detect human bodies using YOLO"""
    global body_detector
    
    if not YOLO_AVAILABLE or body_detector is None:
        return []
    
    try:
        results = body_detector(frame, verbose=False, conf=0.4)
        bodies = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    x = int(x1)
                    y = int(y1)
                    bodies.append((x, y, w, h, conf))
        
        return bodies
    except Exception as e:
        return []

def match_body_to_student(body_box, known_students):
    """Try to match a detected body to a known student"""
    global frame_cache
    
    bx, by, bw, bh = body_box[:4]
    body_center_x = bx + bw // 2
    body_center_y = by + bh // 2
    best_match_id = None
    best_distance = float('inf')
    
    for student_id, cache_data in frame_cache.items():
        if student_id in known_students:
            fx, fy, fw, fh = cache_data['box']
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            distance = math.sqrt((body_center_x - face_center_x)**2 + (body_center_y - face_center_y)**2)
            
            if by <= fy and (by + bh) >= (fy + fh) and distance < bw * 0.5:
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = student_id
    
    if best_match_id:
        confidence = max(0.5, 1.0 - (best_distance / 200))
        return best_match_id, confidence
    
    return None, 0

# ==================== EMOTION DETECTION (IMPROVED) ====================
def detect_emotion_improved(face_roi):
    """Improved emotion detection with better sensitivity"""
    try:
        face_small = cv2.resize(face_roi, (80, 80))
        gray_face = cv2.cvtColor(face_small, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
        
        h, w = gray_face.shape
        eye_region = gray_face[0:int(h*0.4), :]
        mouth_region = gray_face[int(h*0.7):h, :]
        cheek_region = gray_face[int(h*0.4):int(h*0.7), :]
        
        eye_brightness = np.mean(eye_region)
        mouth_brightness = np.mean(mouth_region)
        cheek_brightness = np.mean(cheek_region)
        eye_variance = np.var(eye_region)
        mouth_variance = np.var(mouth_region)
        overall_variance = np.var(gray_face)
        eye_contrast = eye_region.max() - eye_region.min()
        
        mouth_edges = cv2.Canny(mouth_region, 50, 150)
        mouth_edge_density = np.sum(mouth_edges > 0) / mouth_edges.size
        
        # IMPROVED THRESHOLDS untuk lebih sensitif
        if eye_brightness < 70 and eye_contrast < 50:  # Lebih sensitif untuk drowsy
            return 'drowsy'
        
        if mouth_brightness > (cheek_brightness + 10) and mouth_edge_density > 0.10:
            if mouth_variance > 250:
                return 'happy'
            else:
                return 'smile'
        
        if eye_brightness < 90 and overall_variance < 250:
            return 'sad'
        
        if overall_variance > 500 and eye_brightness > 90:
            return 'surprise'
        
        if eye_variance > 350 and mouth_brightness < cheek_brightness:
            return 'angry'
        
        return 'neutral'
    except Exception as e:
        return 'neutral'

def get_smoothed_emotion(student_id, current_emotion):
    """Improved emotion smoothing with faster response"""
    global emotion_history
    
    if student_id not in emotion_history:
        emotion_history[student_id] = []
    
    emotion_history[student_id].append(current_emotion)
    if len(emotion_history[student_id]) > EMOTION_HISTORY_SIZE:
        emotion_history[student_id].pop(0)
    
    # Butuh minimal 2 deteksi konsisten untuk ganti emosi
    if len(emotion_history[student_id]) >= EMOTION_CHANGE_THRESHOLD:
        recent_emotions = emotion_history[student_id][-EMOTION_CHANGE_THRESHOLD:]
        if len(set(recent_emotions)) == 1:  # Semua sama
            return recent_emotions[0]
    
    emotion_counts = Counter(emotion_history[student_id])
    return emotion_counts.most_common(1)[0][0]

# ==================== HAND DETECTION (FIXED & IMPROVED) ====================
def count_raised_fingers(hand_landmarks):
    """Count raised fingers"""
    try:
        fingers_up = 0
        
        # Thumb
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
            fingers_up += 1
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
                fingers_up += 1
        
        return fingers_up
    except Exception as e:
        return 0

def detect_raised_hands_mediapipe(frame, detected_faces):
    """IMPROVED hand detection with better accuracy"""
    global hands_detector
    
    if hands_detector is None:
        return []
    
    try:
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)
        
        valid_hands = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                finger_count = count_raised_fingers(hand_landmarks)
                
                # IMPROVED: Minimal 3 jari untuk dianggap angkat tangan
                if finger_count < 3:
                    continue
                
                wrist = hand_landmarks.landmark[0]
                middle_base = hand_landmarks.landmark[9]
                hand_center_x = int((wrist.x + middle_base.x) / 2 * w)
                hand_center_y = int((wrist.y + middle_base.y) / 2 * h)
                
                min_distance = float('inf')
                closest_face = None
                
                for face_box in detected_faces:
                    fx, fy, fw, fh, fconf = face_box
                    face_center_x = fx + fw // 2
                    face_center_y = fy + fh // 2
                    
                    # IMPROVED: Tangan harus di atas wajah
                    if hand_center_y < (fy - fh * HAND_ABOVE_FACE_THRESHOLD):
                        horizontal_dist = abs(hand_center_x - face_center_x)
                        
                        # IMPROVED: Jarak horizontal tidak terlalu jauh
                        if horizontal_dist < fw * HAND_HORIZONTAL_THRESHOLD:
                            total_distance = np.sqrt(
                                (hand_center_x - face_center_x)**2 +
                                (hand_center_y - face_center_y)**2
                            )
                            if total_distance < min_distance:
                                min_distance = total_distance
                                closest_face = (face_center_x, face_center_y, fx, fy, fw, fh)
                
                if closest_face:
                    valid_hands.append({
                        'position': (hand_center_x, hand_center_y),
                        'finger_count': finger_count,
                        'closest_face': closest_face,
                        'distance': min_distance
                    })
        
        return valid_hands
    except Exception as e:
        log("RAISE_HAND", f"MediaPipe error: {e}")
        return []

def verify_hand_temporal(student_id, finger_count):
    """IMPROVED temporal verification for hand raising"""
    global hand_detection_history
    current_time = time.time()
    
    if student_id not in hand_detection_history:
        hand_detection_history[student_id] = []
    
    hand_detection_history[student_id].append({
        'time': current_time,
        'fingers': finger_count
    })
    
    # Hanya simpan deteksi 2 detik terakhir
    hand_detection_history[student_id] = [
        d for d in hand_detection_history[student_id]
        if current_time - d['time'] < 2.0
    ]
    
    # IMPROVED: Butuh HAND_TEMPORAL_FRAMES deteksi konsisten
    if len(hand_detection_history[student_id]) >= HAND_TEMPORAL_FRAMES:
        recent_fingers = [d['fingers'] for d in hand_detection_history[student_id][-HAND_TEMPORAL_FRAMES:]]
        if len(recent_fingers) >= HAND_TEMPORAL_FRAMES and all(f >= 3 for f in recent_fingers):
            return True
    return False

# ==================== TRACKING / PRESENCE MANAGEMENT ====================
def mark_student_seen(student_id, box):
    """Update last seen timestamp and handle attendance/return logic."""
    now = time.time()
    info = todays_attendance.get(student_id)
    if info is None:
        # New detection for the day
        todays_attendance[student_id] = {
            'first_seen': now,
            'last_seen': now,
            'present': True,
            'missing_since': None,
            'return_since': None,
            'raise_hand_count': 0,
            'emotion': 'neutral'
        }
        log("ATTENDANCE", f"{student_id} baru terdeteksi, catat ke presensi.")
    else:
        # Update last seen
        info['last_seen'] = now
        if info.get('missing_since') is not None:
            # Student kembali setelah missing
            since = now - info['missing_since']
            if since >= STUDENT_RETURN_THRESHOLD:
                info['missing_since'] = None
                info['return_since'] = now
                students_returned[student_id] = {
                    'time': now,
                    'box': box
                }
                log("RETURN", f"Siswa {student_id} kembali setelah {int(since)}s.")
                # optional notify
                if can_send_whatsapp(student_id, "returned"):
                    send_whatsapp_with_json(f"Siswa {student_id} kembali terdeteksi", {'id': student_id, 'time': datetime.datetime.now().isoformat()})
        info['present'] = True

def mark_student_missing(student_id):
    """Mark student as missing and optionally notify."""
    info = todays_attendance.get(student_id)
    now = time.time()
    if info and info.get('missing_since') is None:
        info['missing_since'] = now
        info['present'] = False
        students_missing[student_id] = {'time': now}
        log("MISSING", f"Siswa {student_id} hilang dari jangkauan kamera.")
        if can_send_whatsapp(student_id, "missing"):
            send_whatsapp_with_json(f"Siswa {student_id} hilang dari kamera", {'id': student_id, 'time': datetime.datetime.now().isoformat()})

# ==================== FRAME PROCESSING PIPELINE ====================
# We keep single worker to avoid overload on RPi
processing_lock = threading.Lock()
last_process_time = 0
last_motion_time = time.time()
INACTIVITY_TIMEOUT = 20  # seconds of no motion -> low-power

# Face detector fallback: try cv2.FaceDetectorYN, otherwise fallback to face_recognition
face_detector_obj = None
def init_face_detector():
    global face_detector_obj, face_detector
    try:
        if os.path.exists(YOLO_FACE_MODEL_PATH):
            face_detector_obj = cv2.FaceDetectorYN.create(YOLO_FACE_MODEL_PATH, "", (FRAME_WIDTH, FRAME_HEIGHT))
            log("INIT", "FaceDetectorYN loaded.")
        else:
            face_detector_obj = None
            log("INIT", "FaceDetectorYN model not found, will fallback to face_recognition.")
    except Exception as e:
        face_detector_obj = None
        log("INIT", f"FaceDetectorYN init error: {e}")

def detect_faces_in_frame(frame):
    """Return list of face boxes [(x,y,w,h,conf), ...]"""
    h, w, _ = frame.shape
    if face_detector_obj is not None:
        try:
            face_detector_obj.setInputSize((w, h))
            _, faces = face_detector_obj.detect(frame)
            faces = faces if faces is not None else []
            boxes = []
            for face in faces:
                x, y, fw, fh = map(int, face[:4])
                conf = float(face[-1])
                boxes.append((x, y, fw, fh, conf))
            return boxes
        except Exception as e:
            log("DETECT_FACE", f"FaceDetectorYN error: {e}")
    # fallback
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model='hog')  # 'cnn' is heavier
        boxes = []
        for top, right, bottom, left in locs:
            x, y, fw, fh = left, top, right-left, bottom-top
            boxes.append((x, y, fw, fh, 1.0))
        return boxes
    except Exception as e:
        log("DETECT_FACE", f"Fallback face detection error: {e}")
        return []

def process_frame(frame, frame_idx):
    """Main per-frame processing: face detect, recognize (skipped), hand detect, emotion, update metrics."""
    global last_process_time, last_motion_time, current_fps

    t0 = time.time()
    start_log_id = int(t0)
    faces_detected = 0
    recognitions = 0
    successful_recog = 0
    hand_detected_any = False
    emotion_detected_any = False
    avg_conf = 0.0

    # 1) Preprocess (brightness)
    enhanced_frame, avg_brightness = adaptive_brightness_enhancement_v2(frame)

    # 2) Motion detection cheap check to allow low-power behavior
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    global last_frame_for_motion
    if 'last_frame_for_motion' not in globals():
        globals()['last_frame_for_motion'] = gray
    frame_delta = cv2.absdiff(globals()['last_frame_for_motion'], gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_found = any(cv2.contourArea(c) > 500 for c in cnts)
    if motion_found:
        last_motion_time = time.time()
    globals()['last_frame_for_motion'] = gray

    # 3) Face detection (maybe skip some frames for performance)
    faces = []
    if frame_idx % PROCESS_EVERY_N_FRAMES == 0 or motion_found:
        faces = detect_faces_in_frame(enhanced_frame)
    faces_detected = len(faces)

    # 4) Hand detection using MediaPipe if available (run every N frames)
    hands = []
    if (frame_idx % HAND_DETECTION_SKIP) == 0:
        hands = detect_raised_hands_mediapipe(enhanced_frame, [(x,y,w,h,conf) for (x,y,w,h,conf) in faces])
        if hands:
            hand_detected_any = True

    # 5) For each face: attempt recognition (skip to reduce load), emotion
    for (x, y, fw, fh, conf) in faces:
        avg_conf += conf
        # skip too small or low conf
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # crop face ROI and upscale if small
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(enhanced_frame.shape[1], x + fw), min(enhanced_frame.shape[0], y + fh)
        face_roi = enhanced_frame[y1:y2, x1:x2]
        face_roi = upscale_small_face(face_roi)

        # Face recognition (skip frames to save CPU)
        student_id = None
        if (frame_idx % FACE_RECOGNITION_SKIP) == 0:
            try:
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_face)
                if encs:
                    # attempt match
                    best_idx = None
                    if known_face_encodings:
                        dists = face_recognition.face_distance(known_face_encodings, encs[0])
                        best_idx = int(np.argmin(dists))
                        # NOTE: use a proper tolerance instead of FACE_RECOGNITION_SKIP var - set sensible threshold
                        if dists[best_idx] <= 0.55:
                            student_id = known_face_data[best_idx]['id']
                            recognitions += 1
                            successful_recog += 1
                        else:
                            student_id = None
                else:
                    student_id = None
            except Exception as e:
                log("RECOG", f"Face recognition error: {e}")
                student_id = None

        # Emotion detection improved (skip heavy work)
        emo = None
        if (frame_idx % EMOTION_DETECTION_SKIP) == 0:
            emo_raw = detect_emotion_improved(face_roi)
            emo = get_smoothed_emotion(student_id or f"unk_{x}_{y}", emo_raw)
            emotion_detected_any = True
            # update per-student memory
            if student_id:
                if student_id not in todays_attendance:
                    mark_student_seen(student_id, (x1,y1,fw,fh))
                todays_attendance[student_id]['emotion'] = emo

        # Associate hands to faces and verify temporal
        if hands:
            for hand in hands:
                # closest_face in hand dict: (face_center_x, face_center_y, fx, fy, fw, fh)
                closest = hand.get('closest_face')
                if closest:
                    _, _, fx, fy, fww, fhh = closest
                    # ensure current face matches
                    if abs(fx - x1) < max(30, fww*0.2) and abs(fy - y1) < max(30, fhh*0.2):
                        # candidate raise-hand for this face
                        finger_count = hand.get('finger_count', 0)
                        # assign temporary student id if none
                        sid = student_id or f"unknown_{fx}_{fy}"
                        # push into temporal history
                        if verify_hand_temporal(sid, finger_count):
                            # confirmed raise-hand
                            hand_detected_any = True
                            students_raising_hand[sid] = {
                                'time': time.time(),
                                'box': (x1, y1, fw, fh),
                                'fingers': finger_count
                            }
                            # increment raise count in attendance
                            if sid in todays_attendance:
                                todays_attendance[sid]['raise_hand_count'] = todays_attendance[sid].get('raise_hand_count', 0) + 1
                            log("RAISE_HAND", f"Confirmed raise by {sid} (fingers:{finger_count})")
                            # notify once with cooldown
                            if can_send_whatsapp(sid, "raise"):
                                send_whatsapp_with_json(f"Siswa {sid} mengangkat tangan", {'id': sid, 'time': datetime.datetime.now().isoformat()})
                            break  # one hand associated per face

    # 6) Update coverage metrics
    detection_time = time.time() - t0
    avg_conf = (avg_conf / max(1, faces_detected)) if faces_detected else 0
    update_coverage_metrics(faces_detected, recognitions, successful_recog, hand_detected_any, emotion_detected_any, avg_conf, detection_time)

    # 7) Check missing students (based on last_seen)
    now = time.time()
    for sid, info in list(todays_attendance.items()):
        if info.get('present', True):
            # if last_seen older than threshold -> mark missing
            if now - info['last_seen'] > STUDENT_MISSING_THRESHOLD:
                mark_student_missing(sid)

    # log perf occasionally
    if int(time.time()) % 15 == 0:
        cov = get_coverage_stats()
        log("METRICS", f"Faces:{faces_detected} Hands:{hand_detected_any} Emo:{emotion_detected_any} FPS:{current_fps:.1f} Cov:{cov.get('face_detection_rate','N/A')}%")

    return {
        'faces': faces_detected,
        'recognitions': recognitions,
        'successful_recog': successful_recog,
        'hand': hand_detected_any,
        'emotion': emotion_detected_any,
        'detection_time': detection_time
    }

# ==================== VIDEO CAPTURE THREAD ====================
capture_thread_running = False
def video_capture_loop():
    global vs, display_frame, frame_cache, current_fps, fps_counter, fps_start_time, capture_thread_running, last_motion_time

    vs = VideoStream(src=CAM_INDEX, usePiCamera=False).start()
    time.sleep(1.0)
    frame_idx = 0
    capture_thread_running = True
    log("CAPTURE", "Video capture started.")
    try:
        while capture_thread_running:
            frame = vs.read()
            if frame is None:
                log("CAPTURE", "No frame read, retrying...")
                time.sleep(0.5)
                continue

            # resize for consistency
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            frame_idx += 1

            # update FPS
            global fps_counter, fps_start_time, current_fps
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # process frame in same thread but respecting load: skip when executor busy
            try:
                # submit to executor to avoid blocking capture (single worker so safe)
                if executor._work_queue.qsize() < MAX_QUEUE_SIZE:
                    executor.submit(process_frame, frame.copy(), frame_idx)
                else:
                    # queue full -> drop heavy processing but still update display
                    pass
            except Exception as e:
                log("EXEC", f"Submit error: {e}")

            # prepare display frame overlay (lightweight)
            disp = frame.copy()
            # overlay attendance count
            attend_count = len([1 for v in todays_attendance.values() if v.get('present')])
            cv2.putText(disp, f"Attendance: {attend_count}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # draw raise-hand boxes
            for sid, rinfo in list(students_raising_hand.items()):
                bx, by, bw, bh = rinfo['box']
                try:
                    cv2.rectangle(disp, (bx,by), (bx+bw, by+bh), (0,255,255), 2)
                    cv2.putText(disp, f"Raise:{sid}", (bx, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                except Exception:
                    pass

            # update global display_frame (for streaming)
            with display_lock:
                display_frame = disp.copy()

            # if inactivity for long, sleep small to reduce CPU
            if time.time() - last_motion_time > INACTIVITY_TIMEOUT:
                time.sleep(0.05)  # low-power idle
    except Exception as e:
        log("CAPTURE", f"Capture loop error: {e}")
    finally:
        try:
            vs.stop()
        except:
            pass
        capture_thread_running = False
        log("CAPTURE", "Video capture stopped.")

# ==================== FLASK ROUTES / STREAMING ====================
INDEX_HTML = """
<!doctype html>
<html>
<head><title>Classroom Monitor v12</title></head>
<body>
  <h2>Classroom Monitor v12</h2>
  <img id="stream" src="/video_feed" width="640" />
  <pre id="log"></pre>
<script>
  const evtSource = new EventSource('/events');
  evtSource.onmessage = function(e) {
    document.getElementById('log').textContent = e.data + "\\n" + document.getElementById('log').textContent;
  };
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

def gen_frame():
    """Generator for MJPEG stream"""
    while True:
        with display_lock:
            if display_frame is None:
                # send blank
                blank = 255 * np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                ret, jpeg = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY_LOCAL])
                frame_bytes = jpeg.tobytes()
            else:
                ret, jpeg = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY_LOCAL])
                frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    return jsonify(get_coverage_stats())

# Simple Server-Sent Events for logs (lightweight)
@app.route('/events')
def events():
    def stream():
        while True:
            time.sleep(1)
            yield f"data: {json.dumps({'time':datetime.datetime.now().isoformat(), 'attendance': len(todays_attendance)})}\n\n"
    return Response(stream(), mimetype='text/event-stream')

# ==================== STARTUP & CLEANUP ====================
def start_system():
    # init detectors
    init_face_detector()
    global hands_detector, face_mesh_detector, body_detector
    try:
        hands_detector = mp_hands.Hands(min_detection_confidence=HAND_DETECTION_CONFIDENCE, min_tracking_confidence=0.5, max_num_hands=4)
    except Exception as e:
        log("INIT", f"MediaPipe Hands init failed: {e}")
        hands_detector = None

    try:
        face_mesh_detector = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    except Exception as e:
        log("INIT", f"FaceMesh init failed: {e}")
        face_mesh_detector = None

    if YOLO_AVAILABLE:
        try:
            global body_detector
            body_detector = YOLO(YOLO_BODY_MODEL_PATH)
            log("INIT", "YOLO body detector initialized.")
        except Exception as e:
            body_detector = None
            log("INIT", f"YOLO body init error: {e}")

    # start capture thread
    t = threading.Thread(target=video_capture_loop, daemon=True)
    t.start()
    log("SYSTEM", "System started.")

def stop_system():
    global capture_thread_running
    capture_thread_running = False
    try:
        executor.shutdown(wait=False)
    except:
        pass
    log("SYSTEM", "System stopping...")

# Graceful exit handler
import atexit
atexit.register(stop_system)

# ==================== ENTRYPOINT ====================
if __name__ == '__main__':
    try:
        # Preload known faces from disk (if you want)
        # Implement a simple loader here to populate known_face_encodings and known_face_data
        known_face_encodings = []
        known_face_data = []
        if os.path.exists(DATASET_PATH):
            for filename in os.listdir(DATASET_PATH):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        base = os.path.splitext(filename)[0]
                        if '_' in base:
                            name, student_id = base.split('_', 1)
                        else:
                            name, student_id = base, base
                        image_path = os.path.join(DATASET_PATH, filename)
                        image = face_recognition.load_image_file(image_path)
                        encs = face_recognition.face_encodings(image)
                        if encs:
                            known_face_encodings.append(encs[0])
                            known_face_data.append({'id': student_id, 'name': name})
                            log("LOAD", f"Loaded face: {name} (ID:{student_id})")
                    except Exception as e:
                        log("LOAD", f"Failed to load {filename}: {e}")
        start_system()
        # run Flask SocketIO server
        socketio.run(app, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        log("SYSTEM", "Interrupted by user.")
    except Exception as e:
        log("SYSTEM", f"Fatal error: {e}")
    finally:
        stop_system()
