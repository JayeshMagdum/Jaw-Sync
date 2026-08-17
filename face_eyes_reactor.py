"""
face_eyes_reactor.py
=====================
Runs the Teachable Machine face/no-face model against a live webcam feed
and drives the robot's eyes (via the existing Flask /api/eyes endpoint in
app.py) accordingly:

    Face detected (confident, for a few consecutive frames) -> "listening" mode
    No face for a while                                     -> "idle" mode

This does NOT touch serial/hardware directly — it just calls the same
/api/eyes HTTP API your index.html dashboard already uses, so app.py
(root) must be running first.

Install (one-time):
    pip install tensorflow opencv-python requests pillow numpy

Run:
    python face_eyes_reactor.py
"""

import time
import requests
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Use tf_keras (the Keras 2 compatibility package) so that legacy Teachable
# Machine .h5 exports — which use DepthwiseConv2D(groups=1) and other Keras 2
# constructs — load without errors under TensorFlow 2.21+.
# Install once:  pip install tf-keras
# ---------------------------------------------------------------------------
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"          # tell TF to route keras → tf_keras
import tf_keras
from tf_keras.models import load_model

# ---------------------------------------------------------------------
# CONFIG — adjust paths/URL to match your setup
# ---------------------------------------------------------------------
MODEL_PATH = "eyes_model/keras_model.h5"
LABELS_PATH = "eyes_model/labels.txt"
EYES_API_URL = "http://localhost:5000/api/eyes"   # app.py must be running

CONFIDENCE_THRESHOLD = 0.80      # ignore predictions below this
CONSECUTIVE_FRAMES_NEEDED = 4    # require N frames in a row before switching
                                  # state — avoids flickering on a single bad frame
CAMERA_INDEX = 0                 # 0 = default webcam; change if you have multiple

# ---------------------------------------------------------------------
# Load model + labels (Teachable Machine's exported label format is
# "0 Face" / "1 No_Face" — one per line, index-prefixed)
# ---------------------------------------------------------------------
print("[Model] Loading Teachable Machine model...")
model = load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip()
              for line in f.readlines()]
print(f"[Model] Labels: {labels}")

# Teachable Machine's Keras export expects 224x224 RGB, normalized to [-1, 1]
INPUT_SIZE = (224, 224)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def predict(frame_bgr: np.ndarray):
    x = preprocess(frame_bgr)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return labels[idx], float(preds[idx])


def set_eyes_state(state: str) -> None:
    """state: 'listening' (face seen) or 'idle' (no face)."""
    try:
        resp = requests.post(
            EYES_API_URL,
            json={"command": "mode", "params": {"state": state}},
            timeout=1.5,
        )
        resp.raise_for_status()
        print(f"[Eyes] -> {state}")
    except Exception as e:
        print(f"[Eyes] Failed to set state '{state}': {e}")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    current_state = "idle"
    consecutive_face = 0
    consecutive_no_face = 0

    print("[Main] Starting detection loop. Press 'q' in the preview window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Main] Camera read failed, retrying...")
                time.sleep(0.2)
                continue

            label, confidence = predict(frame)
            is_face = (label == "Face" and confidence >= CONFIDENCE_THRESHOLD)

            if is_face:
                consecutive_face += 1
                consecutive_no_face = 0
            else:
                consecutive_no_face += 1
                consecutive_face = 0

            if consecutive_face >= CONSECUTIVE_FRAMES_NEEDED and current_state != "listening":
                current_state = "listening"
                set_eyes_state("listening")
            elif consecutive_no_face >= CONSECUTIVE_FRAMES_NEEDED and current_state != "idle":
                current_state = "idle"
                set_eyes_state("idle")

            # Live preview window (optional — comment out cv2.imshow lines
            # if running headless on the robot with no monitor attached)
            overlay = f"{label} ({confidence:.2f}) | state={current_state}"
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] Stopped.")


if __name__ == "__main__":
    main()
