"""
neck_tracker.py
================
Face-position tracking for the neck PAN servo (left/right), on the same
SCServo/ST3215 bus (COM5) as the jaw, but a DIFFERENT motor ID.

IMPORTANT — run this standalone, not at the same time as app.py / any
other script that opens COM5. Bus servos share one physical wire; only
one process can hold that serial port open at a time. Once this is
confirmed working, it can be merged into the same shared connection the
jaw already uses (ask me when you're ready for that step).

Two modes:

    python neck_tracker.py --calibrate
        Interactive: press 'a' / 'd' to nudge the neck left/right in
        small steps, prints the raw position value after every move.
        Use this to find your NECK_LEFT / NECK_CENTER / NECK_RIGHT
        values below BEFORE running tracking mode. Press 'q' to quit.

    python neck_tracker.py
        Live face-position tracking: opens the webcam, detects faces
        with OpenCV's built-in Haar Cascade (no training required —
        this is separate from your Teachable Machine model, which only
        classifies "face present or not", not WHERE the face is in
        frame), and pans the neck to keep the largest detected face
        centered.

Install (one-time, if not already present):
    pip install opencv-python pyserial
"""

import sys
import time
import argparse

import cv2
import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------
# HARDWARE CONFIG — fill these in after running --calibrate
# ---------------------------------------------------------------------
NECK_PORT = "COM6"          # confirmed by scan_servo_ids.py
NECK_BAUD = 1000000
NECK_MOTOR_ID = 2           # confirmed by scan_servo_ids.py (jaw is ID 1, neck is ID 2)

NECK_CENTER = 3578          # <-- REQUIRED: position value when facing straight ahead
NECK_LEFT = 3158            # <-- REQUIRED: safe limit turning to (physical) left
NECK_RIGHT = 3890           # <-- REQUIRED: safe limit turning to (physical) right
NECK_INVERT = True          # flip to True if tracking turns the wrong direction

NECK_SPEED = 1000            # lower than the jaw's speed — neck moves should be gentler

# If your camera is physically mounted sideways, rotate the feed here so
# BOTH the preview and face detection see an upright image (Haar Cascade
# expects roughly upright faces — a sideways feed can hurt detection
# accuracy, not just look wrong). Options:
#   cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE,
#   cv2.ROTATE_180, or None for no rotation.
CAMERA_ROTATE = cv2.ROTATE_90_CLOCKWISE

# ---------------------------------------------------------------------
# Same SCServo checksum packet protocol as jaw_synced_player.py —
# duplicated here (not imported) because this opens its OWN serial
# connection for standalone testing, per the port-sharing note above.
# ---------------------------------------------------------------------
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_MODE = 33


def checksum(data):
    return (~sum(data)) & 0xFF


def write_reg(ser, mid, reg, data_bytes):
    body = [0x03, reg] + list(data_bytes)
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()


def setup_motor(ser, mid, speed):
    write_reg(ser, mid, ADDR_TORQUE_ENABLE, [0])
    time.sleep(0.01)
    write_reg(ser, mid, ADDR_MODE, [0])
    time.sleep(0.01)
    write_reg(ser, mid, ADDR_TORQUE_ENABLE, [1])
    time.sleep(0.01)
    write_reg(ser, mid, ADDR_GOAL_SPEED, [speed & 0xFF, (speed >> 8) & 0xFF])


def read_position(ser, mid):
    """Read the servo's current actual position (register 56, 2 bytes)."""
    body = [0x02, 56, 2]
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(checksum(pkt[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(pkt))
    ser.flush()
    time.sleep(0.02)
    resp = ser.read(32)
    start = resp.find(b"\xff\xff")
    if start < 0 or len(resp) < start + 6:
        return None
    frame = resp[start:]
    l = frame[3]
    if len(frame) < 4 + l:
        return None
    params = frame[5: 5 + max(0, l - 2)]
    if len(params) < 2:
        return None
    return params[0] | (params[1] << 8)


def set_position(ser, mid, position):
    pos = max(0, min(4095, int(position)))
    write_reg(ser, mid, ADDR_GOAL_POSITION, [pos & 0xFF, (pos >> 8) & 0xFF])


def release_torque(ser, mid):
    write_reg(ser, mid, ADDR_TORQUE_ENABLE, [0])


def open_serial():
    try:
        return serial.Serial(NECK_PORT, NECK_BAUD, timeout=0.05)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open {NECK_PORT}: {e}")
        print("        Is app.py / another script already holding this port open?")
        sys.exit(1)


# ---------------------------------------------------------------------
# CALIBRATION MODE
# ---------------------------------------------------------------------
def run_calibrate():
    if NECK_MOTOR_ID is None:
        print("[ERROR] Set NECK_MOTOR_ID at the top of this file first "
              "(run scan_servo_ids.py to find it).")
        sys.exit(1)

    ser = open_serial()

    # Read the current position FIRST, before any setup_motor() writes —
    # writing torque/mode/speed registers right before a read can race the
    # half-duplex bus and cause the read to come back empty. Retry a few
    # times with a short settle delay for safety.
    pos = None
    for attempt in range(5):
        pos = read_position(ser, NECK_MOTOR_ID)
        if pos is not None:
            break
        time.sleep(0.1)

    if pos is None:
        print("[WARN] Could not read current servo position after 5 attempts.")
        print("       Check: correct motor ID, servo powered, DATA wire connected,")
        print("       and that no other program (app.py, etc.) has COM6 open.")
        ser.close()
        return

    setup_motor(ser, NECK_MOTOR_ID, NECK_SPEED)
    time.sleep(0.05)  # let the setup writes settle before any movement commands

    step = 30
    print("\nCalibration mode — neck motor id:", NECK_MOTOR_ID)
    print(f"Starting from CURRENT actual position (read from servo): {pos}")
    print("  a = nudge left   d = nudge right   +/- = bigger/smaller step   q = quit")
    print(f"  l = jump to saved NECK_LEFT   ({NECK_LEFT})")
    print(f"  c = jump to saved NECK_CENTER ({NECK_CENTER})")
    print(f"  r = jump to saved NECK_RIGHT  ({NECK_RIGHT})\n")
    print(f"Position: {pos}  (not moved yet — first command nudges/jumps from here)")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "a":
                pos -= step
            elif cmd == "d":
                pos += step
            elif cmd == "l":
                if NECK_LEFT is None:
                    print("NECK_LEFT not set yet.")
                    continue
                pos = NECK_LEFT
                print(f"Jumping to saved NECK_LEFT ({pos})...")
            elif cmd == "c":
                if NECK_CENTER is None:
                    print("NECK_CENTER not set yet.")
                    continue
                pos = NECK_CENTER
                print(f"Jumping to saved NECK_CENTER ({pos})...")
            elif cmd == "r":
                if NECK_RIGHT is None:
                    print("NECK_RIGHT not set yet.")
                    continue
                pos = NECK_RIGHT
                print(f"Jumping to saved NECK_RIGHT ({pos})...")
            elif cmd == "+":
                step += 10
                print(f"step = {step}")
                continue
            elif cmd == "-":
                step = max(5, step - 10)
                print(f"step = {step}")
                continue
            elif cmd == "q":
                break
            else:
                print("a / d / l / c / r / + / - / q")
                continue
            pos = max(0, min(4095, pos))
            set_position(ser, NECK_MOTOR_ID, pos)
            print(f"Position: {pos}")
    finally:
        print(f"\nLast position: {pos} — note this down as one of your "
              f"NECK_LEFT / NECK_CENTER / NECK_RIGHT values.")
        release_torque(ser, NECK_MOTOR_ID)
        ser.close()


# ---------------------------------------------------------------------
# LIVE FACE-TRACKING MODE
# ---------------------------------------------------------------------
def run_tracking():
    missing = [name for name, val in [
        ("NECK_MOTOR_ID", NECK_MOTOR_ID), ("NECK_CENTER", NECK_CENTER),
        ("NECK_LEFT", NECK_LEFT), ("NECK_RIGHT", NECK_RIGHT),
    ] if val is None]
    if missing:
        print(f"[ERROR] Set these at the top of the file first: {', '.join(missing)}")
        print("        Run: python neck_tracker.py --calibrate")
        sys.exit(1)

    ser = open_serial()
    setup_motor(ser, NECK_MOTOR_ID, NECK_SPEED)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    lo, hi = min(NECK_LEFT, NECK_RIGHT), max(NECK_LEFT, NECK_RIGHT)
    half_range = (hi - lo) / 2.0

    current_pos = float(NECK_CENTER)
    SMOOTHING = 0.15      # 0-1, lower = smoother/slower easing toward target
    DEADZONE = 0.06        # ignore tiny offsets near center, avoids jitter
    MIN_STEP_TO_SEND = 3   # don't spam the bus for sub-pixel-equivalent moves

    print("[Neck] Tracking started. Press 'q' in the preview window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if CAMERA_ROTATE is not None:
                frame = cv2.rotate(frame, CAMERA_ROTATE)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                   minSize=(60, 60))

            target_pos = current_pos  # default: hold position if no face
            if len(faces) > 0:
                # Track the largest face (closest person) if several are visible
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_center_x = x + w / 2.0
                frame_center_x = frame.shape[1] / 2.0
                offset = (face_center_x - frame_center_x) / frame_center_x  # -1..1

                if NECK_INVERT:
                    offset = -offset

                if abs(offset) >= DEADZONE:
                    target_pos = NECK_CENTER + offset * half_range
                    target_pos = max(lo, min(hi, target_pos))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ease current position toward target (smooth, not instant)
            current_pos += (target_pos - current_pos) * SMOOTHING

            if abs(current_pos - getattr(run_tracking, "_last_sent", -9999)) >= MIN_STEP_TO_SEND:
                set_position(ser, NECK_MOTOR_ID, current_pos)
                run_tracking._last_sent = current_pos

            cv2.putText(frame, f"pos={int(current_pos)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Neck Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        release_torque(ser, NECK_MOTOR_ID)
        ser.close()
        print("[Neck] Stopped, torque released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                         help="Interactive mode to find safe left/center/right positions")
    args = parser.parse_args()

    if args.calibrate:
        run_calibrate()
    else:
        run_tracking()