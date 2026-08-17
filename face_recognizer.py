"""
face_recognizer.py
==================
Combined script for face recognition, neck tracking, and eye reaction.
Highly optimized multi-threaded architecture to ensure zero camera lag.

Features:
- Recognizes known people from `known_faces/` folder using `face_recognition`
- Greets recognized people by name via TTS
- Pans the neck servo to keep the largest face centered
- Switches eye state based on face presence
- Includes a --register mode to easily capture reference photos
"""

import os
import sys
import time
import argparse
import threading
from collections import defaultdict
from urllib.parse import urlencode

import cv2
import requests
import serial
try:
    import face_recognition
except ImportError:
    print("[ERROR] Could not import face_recognition. Please install it:")
    print("        pip install face_recognition")
    sys.exit(1)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
RECOGNITION_TOLERANCE = 0.45
GREETING_COOLDOWN = 10      # seconds before greeting the same person again
GREET_UNKNOWN = True        
UNKNOWN_COOLDOWN = 10       # seconds before greeting another unknown

APP_BASE_URL = "http://127.0.0.1:5000"
EYES_API_URL = f"{APP_BASE_URL}/api/eyes"
TTS_API_URL = f"{APP_BASE_URL}/api/tts"
VOICE_ID = "edge:en-IN-NeerjaExpressiveNeural"

NECK_PORT = "COM6"
NECK_BAUD = 1000000
NECK_MOTOR_ID = 2

NECK_CENTER = 3578          
NECK_LEFT = 3158            
NECK_RIGHT = 3890           
NECK_INVERT = True          
NECK_SPEED = 1000

CAMERA_INDEX = 0
CAMERA_ROTATE = cv2.ROTATE_90_CLOCKWISE  

CONSECUTIVE_FRAMES_NEEDED = 4


# ---------------------------------------------------------------------
# DEDICATED CAMERA STREAM THREAD (Ultra-smooth FPS)
# ---------------------------------------------------------------------
class CameraStream:
    """Reads camera frames in a dedicated thread to prevent OS buffer lag."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Could not open camera {src}")
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
    
    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self
    
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                break
            # Read next frame as fast as hardware allows
            self.grabbed, self.frame = self.stream.read()
            
    def read(self):
        return self.grabbed, self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()


# ---------------------------------------------------------------------
# NECK SERVO PROTOCOL (HTTP to jaw_server.py)
# ---------------------------------------------------------------------
JAW_SERVER_URL = "http://127.0.0.1:5050"
http_session = requests.Session()

def _set_position_async(mid, position):
    pos = max(0, min(4095, int(position)))
    try:
        http_session.post(f"{JAW_SERVER_URL}/api/position/{mid}", json={"position": pos}, timeout=0.1)
    except Exception:
        pass

def set_position(ser, mid, position):
    """ser is ignored, kept for compatibility."""
    # Run in background to avoid blocking camera UI
    threading.Thread(target=_set_position_async, args=(mid, position), daemon=True).start()

def release_torque(ser, mid):
    try:
        http_session.post(f"{JAW_SERVER_URL}/api/torque/{mid}/0", timeout=1.0)
    except Exception:
        pass

def setup_motor(ser, mid, speed):
    """Configuration is mostly handled by jaw_server on startup, but we could add speed configs here."""
    print(f"[Neck] Will route commands to {JAW_SERVER_URL} for motor ID {mid}")

def open_serial():
    """Returns a dummy object so the rest of the code thinks it connected."""
    return True


# ---------------------------------------------------------------------
# API HELPERS (Eyes & TTS)
# ---------------------------------------------------------------------
def _set_eyes_state_async(state: str):
    try:
        http_session.post(EYES_API_URL, json={"command": "mode", "params": {"state": state}}, timeout=1.0)
        print(f"[Eyes] -> {state}")
    except Exception as e:
        print(f"[Eyes] Failed to set state '{state}': {e}")

def set_eyes_state(state: str) -> None:
    threading.Thread(target=_set_eyes_state_async, args=(state,), daemon=True).start()

import struct
import math

_audio_lock = threading.Lock()
_is_audio_playing = False

def _play_greeting_async(name: str):
    global _is_audio_playing
    with _audio_lock:
        if _is_audio_playing:
            return
        _is_audio_playing = True

    try:
        base_dir = r"hardocoded_wav\output\en"
        wav_path = os.path.join(base_dir, f"{name}_greeting_en.wav")
        
        if not os.path.exists(wav_path):
            wav_path = os.path.join(base_dir, "greeting_en.wav")
            if not os.path.exists(wav_path):
                print(f"[Audio] Error: Could not find greeting file at {wav_path}")
                return
                
        print(f"[Audio] Playing & Syncing Jaw: {wav_path}")
        try:
            from jaw_synced_player import play_wav_synced
            play_wav_synced(wav_path)
        except Exception as e:
            print(f"[Audio] Playback failed: {e}")
    finally:
        with _audio_lock:
            _is_audio_playing = False

def play_greeting(name: str):
    threading.Thread(target=_play_greeting_async, args=(name,), daemon=True).start()



# ---------------------------------------------------------------------
# FACE RECOGNITION SETUP
# ---------------------------------------------------------------------
def load_known_faces():
    print("[Face] Loading known faces database...")
    known_encodings = []
    known_names = []
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir): continue
            
        count = 0
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, filename)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    count += 1
        print(f"[Face] Loaded {count} reference images for '{person_name}'.")
    return known_encodings, known_names


# ---------------------------------------------------------------------
# REGISTRATION MODE
# ---------------------------------------------------------------------
def run_register(person_name):
    print(f"\n--- Registration Mode for '{person_name}' ---")
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("[ERROR] Could not open camera.")
        
    print("Press 'c' to capture an image. Press 'q' to quit.")
    
    existing = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    count = len(existing) if existing else 0
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue
            if CAMERA_ROTATE is not None:
                frame = cv2.rotate(frame, CAMERA_ROTATE)
                
            display = frame.copy()
            cv2.putText(display, f"Capturing: {person_name} ({count} saved)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registration", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('c'):
                save_path = os.path.join(person_dir, f"{person_name}_{count:03d}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"[Register] Saved {save_path}")
                count += 1
                cv2.rectangle(display, (0,0), (display.shape[1], display.shape[0]), (255,255,255), -1)
                cv2.imshow("Registration", display)
                cv2.waitKey(100)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------
# MAIN TRACKING + RECOGNITION MODE
# ---------------------------------------------------------------------
def run_combined(no_neck=False, no_eyes=False):
    import zmq
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUB)
    zmq_socket.bind("tcp://127.0.0.1:5555")
    
    known_encodings, known_names = load_known_faces()
    
    ser = None
    if not no_neck:
        ser = open_serial()
        if ser:
            setup_motor(ser, NECK_MOTOR_ID, NECK_SPEED)
        else:
            no_neck = True

    try:
        cam_stream = CameraStream(CAMERA_INDEX).start()
    except RuntimeError as e:
        sys.exit(str(e))

    # --- SHARED THREAD STATE ---
    shared_state = {
        "frame": None,          # Latest raw frame for worker
        "locations": [],        # Bounding boxes from worker
        "names": [],            # Names from worker
        "running": True,        
    }
    state_lock = threading.Lock()
    PROCESS_SCALE = 0.4   # increased from 0.25 — better accuracy with minimal speed cost

    # --- RECOGNITION WORKER THREAD ---
    def recognition_worker():
        last_greeted_times = defaultdict(float)
        last_unknown_time = 0
        current_eye_state = "idle"
        consecutive_face = 0
        consecutive_no_face = 0

        while shared_state["running"]:
            start_t = time.time()
            
            with state_lock:
                f = shared_state["frame"]
                frame_copy = f.copy() if f is not None else None
            
            if frame_copy is None:
                time.sleep(0.01)
                continue

            small_frame = cv2.resize(frame_copy, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = []
            if known_encodings:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                if not known_encodings:
                    face_names.append("Unknown")
                    continue
                # Use face distance for best-match accuracy (lower = more similar)
                import numpy as _np
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = int(_np.argmin(distances))
                if distances[best_idx] <= RECOGNITION_TOLERANCE:
                    face_names.append(known_names[best_idx])
                else:
                    face_names.append("Unknown")

            if not known_encodings and face_locations:
                face_names = ["Unknown"] * len(face_locations)

            # Update shared UI data
            with state_lock:
                shared_state["locations"] = face_locations
                shared_state["names"] = face_names

            # Eyes Logic
            if len(face_locations) > 0:
                consecutive_face += 1
                consecutive_no_face = 0
            else:
                consecutive_no_face += 1
                consecutive_face = 0

            if not no_eyes:
                if consecutive_face >= CONSECUTIVE_FRAMES_NEEDED and current_eye_state != "listening":
                    current_eye_state = "listening"
                    set_eyes_state("listening")
                elif consecutive_no_face >= CONSECUTIVE_FRAMES_NEEDED and current_eye_state != "idle":
                    current_eye_state = "idle"
                    set_eyes_state("idle")

            # Greeting Logic
            # Track which names are currently visible
            current_visible = set(face_names)
            now = time.time()
            for name in face_names:
                if name == "Unknown":
                    if GREET_UNKNOWN and (now - last_unknown_time > UNKNOWN_COOLDOWN):
                        play_greeting("Unknown")
                        last_unknown_time = now
                else:
                    # Greet if: never greeted, OR was away and came back (cooldown expired)
                    if (now - last_greeted_times[name]) > GREETING_COOLDOWN:
                        play_greeting(name)
                        last_greeted_times[name] = now

            # Throttling: ~15 FPS for recognition (was 10 FPS)
            elapsed = time.time() - start_t
            if elapsed < 0.066:
                time.sleep(0.066 - elapsed)

    worker_thread = threading.Thread(target=recognition_worker, daemon=True)
    worker_thread.start()

    # --- MAIN UI & SERVO THREAD ---
    print("\n[System] Live face tracking started. Press 'q' to quit.")
    
    lo = min(NECK_LEFT, NECK_RIGHT)
    hi = max(NECK_LEFT, NECK_RIGHT)
    half_range = (hi - lo) / 2.0
    current_pos = float(NECK_CENTER)
    last_sent_pos = current_pos
    
    SMOOTHING = 0.15
    DEADZONE = 0.06
    MIN_STEP_TO_SEND = 3

    try:
        while True:
            ret, frame = cam_stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            if CAMERA_ROTATE is not None:
                frame = cv2.rotate(frame, CAMERA_ROTATE)

            with state_lock:
                shared_state["frame"] = frame
                face_locations = shared_state["locations"]
                face_names = shared_state["names"]

            # Neck Logic
            target_pos = current_pos
            if face_locations:
                largest_face_idx = max(range(len(face_locations)), 
                                      key=lambda i: (face_locations[i][2] - face_locations[i][0]) * 
                                                    (face_locations[i][1] - face_locations[i][3]))
                
                top, right, bottom, left = face_locations[largest_face_idx]
                face_center_x = (left + right) / 2.0
                frame_center_x = (frame.shape[1] * PROCESS_SCALE) / 2.0
                offset = (face_center_x - frame_center_x) / frame_center_x
                
                if NECK_INVERT: offset = -offset
                
                if abs(offset) >= DEADZONE:
                    target_pos = NECK_CENTER + offset * half_range
                    target_pos = max(lo, min(hi, target_pos))
                    
            current_pos += (target_pos - current_pos) * SMOOTHING
            
            if not no_neck and ser:
                if abs(current_pos - last_sent_pos) >= MIN_STEP_TO_SEND:
                    set_position(ser, NECK_MOTOR_ID, current_pos)
                    last_sent_pos = current_pos

            # Draw Logic
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top = int(top / PROCESS_SCALE)
                right = int(right / PROCESS_SCALE)
                bottom = int(bottom / PROCESS_SCALE)
                left = int(left / PROCESS_SCALE)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Neck: {int(current_pos)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Broadcast frame to GUI via ZMQ
            try:
                ret_encode, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret_encode:
                    zmq_socket.send(buffer)
            except Exception as e:
                print(f"[ZMQ] Stream error: {e}")

            cv2.imshow("Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        shared_state["running"] = False
        cam_stream.stop()
        worker_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        if ser:
            release_torque(ser, NECK_MOTOR_ID)
        if not no_eyes:
            set_eyes_state("idle")
        print("[System] Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", type=str, help="Name to register")
    parser.add_argument("--no-neck", action="store_true", help="Disable neck servo")
    parser.add_argument("--no-eyes", action="store_true", help="Disable eyes API")
    args = parser.parse_args()

    if args.register:
        run_register(args.register)
    else:
        run_combined(args.no_neck, args.no_eyes)
