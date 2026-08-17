import asyncio
import base64
import io
import json
import os
import sys
import threading
import time

from flask import Flask, Response, jsonify, request, send_from_directory
import edge_tts
import serial

# ---------------------------------------------------------------------------
# This is a completely standalone jaw + motor control app.
# It does NOT import or modify the original app.py / index.html in any way.
# Run this INSTEAD of the original app.py (both cannot hold the serial port
# at the same time) unless you point PORT at a second USB adapter.
# ---------------------------------------------------------------------------

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7B289175-if00"
BAUD = 1_000_000

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jaw_config.json")

DEFAULT_CONFIG = {
    "motor_id": 1,
    "jaw_open": 1860,     # mouth fully open position
    "jaw_close": 2730,    # mouth fully closed position
    "jaw_home": 2730,     # resting position (usually same as close, can differ)
    "speed": 1200,        # servo goal speed register value
}

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_MODE = 33

POS_MIN = 0
POS_MAX = 4095

config_lock = threading.Lock()
lock = threading.Lock()


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


cfg = load_config()

# --- Serial port connection ---
MOCK_MODE = True
mock_position = cfg["jaw_close"]


class MockSerial:
    is_open = True
    def reset_input_buffer(self): pass
    def write(self, data): pass
    def flush(self): pass
    def read(self, n): return b""
    def close(self): pass


port_to_use = cfg.get("port", "COM5")
candidates = [port_to_use]
if sys.platform.startswith("win"):
    candidates.extend([f"COM{i}" for i in range(1, 21)])
else:
    candidates.extend([
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7B289175-if00",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ])

seen = set()
ports_to_try = [p for p in candidates if not (p in seen or seen.add(p))]

motor_serial = None
for p in ports_to_try:
    try:
        motor_serial = serial.Serial(p, BAUD, timeout=0.1)
        MOCK_MODE = False
        print(f"[JawServer] Connected to motor hardware on {p} at {BAUD} baud.", flush=True)
        time.sleep(0.05)
        break
    except serial.SerialException as e:
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            print(f"[JawServer][WARNING] Could not open {p} — Port is locked by another running process! ({e})", flush=True)
        continue
    except Exception:
        continue

if MOCK_MODE:
    print("[MOCK] No motor serial device found — running in MOCK mode.", flush=True)
    motor_serial = MockSerial()


def motor_checksum(data):
    return (~sum(data)) & 0xFF


def motor_write_reg(mid, reg, data_bytes):
    body = [0x03, reg] + list(data_bytes)
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(motor_checksum(pkt[2:]))
    if not MOCK_MODE:
        motor_serial.reset_input_buffer()
        motor_serial.write(bytes(pkt))
        # Removed flush/sleep/read so writes are fire-and-forget (0ms latency instead of 110ms)
    return b""


def motor_read_reg(mid, reg, n_bytes):
    global mock_position
    if MOCK_MODE:
        if reg == ADDR_PRESENT_POSITION and n_bytes == 2:
            pos = mock_position
            return bytes([pos & 0xFF, (pos >> 8) & 0xFF])
        return bytes(n_bytes)
    body = [0x02, reg, n_bytes]
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(motor_checksum(pkt[2:]))
    motor_serial.reset_input_buffer()
    motor_serial.write(bytes(pkt))
    motor_serial.flush()
    time.sleep(0.02)  # reduced from 0.06 — 20ms is enough for 1Mbaud response
    resp = motor_serial.read(20)
    start = resp.find(b"\xff\xff")
    if start < 0 or len(resp) < start + 6:
        return b""
    frame = resp[start:]
    length = frame[3]
    expected = 4 + length
    if len(frame) < expected:
        return b""
    params = frame[5:5 + max(0, length - 2)]
    return params[:n_bytes]


def motor_write_u8(mid, reg, value):
    return motor_write_reg(mid, reg, [int(value) & 0xFF])


def motor_write_u16(mid, reg, value):
    value = max(0, min(4095, int(value)))
    return motor_write_reg(mid, reg, [value & 0xFF, (value >> 8) & 0xFF])


def motor_set_position_mode(mid):
    # Many SCServo/Feetech-protocol servos silently ignore a Mode-register
    # change while torque is already enabled. Disable torque first, switch
    # mode, then re-enable torque, so this always actually takes effect
    # even if the servo was left in wheel mode from a previous session.
    motor_write_u8(mid, ADDR_TORQUE_ENABLE, 0)
    time.sleep(0.01)
    motor_write_u8(mid, ADDR_MODE, 0)
    time.sleep(0.01)
    motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1)


def motor_write_position(mid, position):
    global mock_position
    position = max(0, min(4095, int(position)))
    if MOCK_MODE:
        mock_position = position
        return b""
    return motor_write_reg(mid, ADDR_GOAL_POSITION, [position & 0xFF, (position >> 8) & 0xFF])


# Tracks which motor_ids have been put into position mode.
# /api/jaw and /api/position/<mid> compare against this before
# deciding whether to call motor_set_position_mode() — skipping the 30-50ms
# torque-disable → mode-set → torque-enable cycle when it's already set.
_modes_applied = set()

# Minimum seconds between consecutive /api/jaw serial dispatches.
# Drops calls that arrive faster than the servo can execute to prevent
# the lock queue from backing up and delivering stale positions to hardware.
# 0.080 s = ~12.5 Hz max.
# In MOCK mode this gate is bypassed so the dashboard stays responsive.
_JAW_MIN_INTERVAL_S: float = 0.080
_last_jaw_dispatch: float = 0.0
_last_jaw_pos: int = 0


def apply_startup_settings():
    global _modes_applied
    with lock:
        motor_set_position_mode(cfg["motor_id"])
        motor_write_u16(cfg["motor_id"], ADDR_GOAL_SPEED, cfg["speed"])
        _modes_applied.add(cfg["motor_id"])


apply_startup_settings()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=".")

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
@app.route("/")
def index():
    return send_from_directory(".", "jaw_dashboard.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    with config_lock:
        c = dict(cfg)
    c["mock_mode"] = MOCK_MODE
    return jsonify(c)


@app.route("/api/config", methods=["POST"])
def set_config():
    global _mode_applied_for_id
    data = request.get_json(force=True)
    with config_lock:
        old_motor_id = cfg["motor_id"]
        for key in ("motor_id", "jaw_open", "jaw_close", "jaw_home", "speed"):
            if key in data:
                cfg[key] = int(data[key])

        # --- Calibration sanity checks ---
        open_v = cfg["jaw_open"]
        close_v = cfg["jaw_close"]
        span = abs(close_v - open_v)
        errors = []
        if span < 50:
            errors.append(f"jaw span too small ({span} ticks) — likely a calibration mistake")
        if span > 1500:
            errors.append(f"jaw span very large ({span} ticks) — risk of hitting mechanical stop")
        if not (0 <= open_v <= 4095):
            errors.append(f"jaw_open {open_v} out of range 0-4095")
        if not (0 <= close_v <= 4095):
            errors.append(f"jaw_close {close_v} out of range 0-4095")
        if not (1 <= cfg["speed"] <= 4095):
            errors.append(f"speed {cfg['speed']} out of range 1-4095")
        if errors:
            # Roll back in-memory changes so cfg stays consistent
            cfg.update(load_config())
            return jsonify({"error": errors}), 400

        save_config(cfg)
        new_cfg = dict(cfg)

    with lock:
        if new_cfg["motor_id"] not in _modes_applied:
            motor_set_position_mode(new_cfg["motor_id"])
            _modes_applied.add(new_cfg["motor_id"])
        motor_write_u16(new_cfg["motor_id"], ADDR_GOAL_SPEED, new_cfg["speed"])

    return jsonify(new_cfg)


@app.route("/api/position", methods=["GET"])
def get_position():
    with config_lock:
        mid = cfg["motor_id"]
    with lock:
        data = motor_read_reg(mid, ADDR_PRESENT_POSITION, 2)
    if len(data) != 2:
        return jsonify({"error": "no position response"}), 500
    pos = data[0] | (data[1] << 8)
    return jsonify({"id": mid, "position": pos})


@app.route("/api/position", methods=["POST"])
def set_position():
    data = request.get_json(force=True)
    with config_lock:
        mid = cfg["motor_id"]
    pos = int(data.get("position", cfg["jaw_close"]))
    pos = max(0, min(4095, pos))
    with lock:
        if mid not in _modes_applied:
            motor_set_position_mode(mid)
            _modes_applied.add(mid)
        motor_write_position(mid, pos)
    print(f"[pos] → {pos}", flush=True)
    return jsonify({"id": mid, "goal": pos})


@app.route("/api/position/<int:mid>", methods=["POST"])
def set_position_mid(mid):
    data = request.get_json(force=True)
    pos = int(data.get("position", 2048))
    pos = max(0, min(4095, pos))
    speed = int(data.get("speed", 1000))
    with lock:
        if mid not in _modes_applied:
            motor_set_position_mode(mid)
            motor_write_u16(mid, ADDR_GOAL_SPEED, speed)
            _modes_applied.add(mid)
            print(f"[Motor {mid}] Initialized: position mode, speed={speed}", flush=True)
        motor_write_position(mid, pos)
    return jsonify({"id": mid, "goal": pos})


@app.route("/api/torque/<int:on>", methods=["POST"])
def torque(on):
    with config_lock:
        mid = cfg["motor_id"]
    with lock:
        motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1 if on else 0)
    return jsonify({"id": mid, "torque": bool(on)})

@app.route("/api/torque/<int:mid>/<int:on>", methods=["POST"])
def torque_mid(mid, on):
    with lock:
        motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1 if on else 0)
    return jsonify({"id": mid, "torque": bool(on)})


@app.route("/api/home", methods=["POST"])
def go_home():
    with config_lock:
        mid = cfg["motor_id"]
        home = cfg["jaw_home"]
    with lock:
        motor_write_position(mid, home)
    print(f"[home] → {home}", flush=True)
    return jsonify({"id": mid, "goal": home})


@app.route("/api/set_home", methods=["POST"])
def set_home_to_current():
    """Reads the motor's current live position and saves it as the home value."""
    with config_lock:
        mid = cfg["motor_id"]
    with lock:
        data = motor_read_reg(mid, ADDR_PRESENT_POSITION, 2)
    if len(data) != 2:
        return jsonify({"error": "no position response"}), 500
    pos = data[0] | (data[1] << 8)
    with config_lock:
        cfg["jaw_home"] = pos
        save_config(cfg)
        new_cfg = dict(cfg)
    return jsonify(new_cfg)


@app.route("/api/jaw", methods=["POST"])
def jaw():
    """amplitude: 0.0 (closed) .. 1.0 (fully open)"""
    global _modes_applied, _last_jaw_dispatch, _last_jaw_pos
    data = request.get_json(force=True)
    amp = float(data.get("amplitude", 0.0))
    amp = max(0.0, min(1.0, amp))
    with config_lock:
        mid = cfg["motor_id"]
        jaw_open = cfg["jaw_open"]
        jaw_close = cfg["jaw_close"]
    pos = int(round(jaw_close - amp * (jaw_close - jaw_open)))

    # --- Rate gate (LIVE mode only) ---
    # If the previous serial write hasn't had time to complete, drop this call
    # rather than queueing it — stale positions delivered late cause stutter.
    now = time.monotonic()
    if not MOCK_MODE and (now - _last_jaw_dispatch) < _JAW_MIN_INTERVAL_S:
        return jsonify({"amplitude": amp, "position": _last_jaw_pos, "rate_limited": True})

    with lock:
        # Only re-apply mode if the motor_id hasn't been configured yet.
        # Avoids the ~30-50ms torque-disable/re-enable cycle on every TTS word.
        if mid not in _modes_applied:
            motor_set_position_mode(mid)
            _modes_applied.add(mid)
        motor_write_position(mid, pos)

    _last_jaw_dispatch = time.monotonic()
    _last_jaw_pos = pos
    return jsonify({"amplitude": amp, "position": pos})


# ---------------------------------------------------------------------------
# TTS with per-word timing (Edge TTS WordBoundary events)
# ---------------------------------------------------------------------------

EDGE_VOICES = [
    ("en-IN-NeerjaExpressiveNeural", "Neerja Expressive — Indian English female (best)"),
    ("en-IN-NeerjaNeural",           "Neerja — Indian English female"),
    ("en-US-AriaNeural",             "Aria — US English female"),
    ("en-US-GuyNeural",              "Guy — US English male"),
    ("hi-IN-SwaraNeural",            "Swara — Hindi female"),
    ("mr-IN-AarohiNeural",           "Aarohi — Marathi female"),
]


@app.route("/api/voices")
def voices():
    return jsonify({"voices": [{"id": v, "label": l} for v, l in EDGE_VOICES],
                     "default": EDGE_VOICES[0][0]})


def synth_with_word_timings(text, voice_id):
    async def _run():
        communicate = edge_tts.Communicate(text, voice_id)
        audio_buf = io.BytesIO()
        words = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start_ms = chunk["offset"] / 10000.0
                dur_ms = chunk["duration"] / 10000.0
                words.append({
                    "text": chunk.get("text", ""),
                    "start_ms": start_ms,
                    "end_ms": start_ms + dur_ms,
                })
        return audio_buf.getvalue(), words
    return asyncio.run(_run())


@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json(force=True)
    text = str(data.get("text", "")).strip()
    voice_id = data.get("voice", EDGE_VOICES[0][0])
    if not text:
        return jsonify({"error": "missing text"}), 400
    try:
        mp3_bytes, words = synth_with_word_timings(text, voice_id)
    except Exception as e:
        return jsonify({"error": f"edge-tts failed: {e}"}), 502

    if not mp3_bytes:
        return jsonify({"error": "no audio returned"}), 502

    audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
    return jsonify({"audio_b64": audio_b64, "words": words})


if __name__ == "__main__":
    print("[JawServer] Server started successfully!", flush=True)
    print("[JawServer] Local host URL: http://127.0.0.1:5050", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
