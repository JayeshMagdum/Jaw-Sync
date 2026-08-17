import asyncio
import io
import json
import threading
import time
import urllib.error
import urllib.request
from flask import Flask, Response, jsonify, request, send_from_directory

import edge_tts
import serial

PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084518-if00"
BAUD = 1_000_000
MOTOR_IDS = [1]
EYES_PORT = "/dev/ttyUSB0"
EYES_BAUD = 115200

# Per-motor safe travel limits (mechanical-stop protection)
# Motor 2 is the InMoov jaw: 2048 is closed/center, going UP hits a hard stop.
LIMITS = {
    7: (0, 4095),
}

JAW_ID = 1
JAW_CLOSED = 2730
JAW_OPEN = 1860

import os
def load_jaw_config():
    global JAW_OPEN, JAW_CLOSED, JAW_ID
    if os.path.exists("jaw_config.json"):
        try:
            with open("jaw_config.json", "r") as f:
                c = json.load(f)
                if "jaw_open" in c: JAW_OPEN = int(c["jaw_open"])
                if "jaw_close" in c: JAW_CLOSED = int(c["jaw_close"])
                if "motor_id" in c: JAW_ID = int(c["motor_id"])
        except Exception as e:
            print("Failed to load jaw config:", e)
    # Give a tiny bit of margin (+- 200) so they can recalibrate if needed, 
    # but still prevent huge movements that break parts.
    lo = min(JAW_OPEN, JAW_CLOSED)
    hi = max(JAW_OPEN, JAW_CLOSED)
    LIMITS[JAW_ID] = (max(0, lo - 300), min(4095, hi + 300))

load_jaw_config()

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_MODE = 33

POS_MIN = 0
POS_MAX = 4095

lock = threading.Lock()

# --- Mock serial mode (no hardware connected) ---
MOCK_MODE = True
mock_positions = {mid: 2048 for mid in MOTOR_IDS}
mock_positions[JAW_ID] = JAW_CLOSED

class MockSerial:
    """Fake serial that simulates motor register reads/writes in memory."""
    is_open = True
    def reset_input_buffer(self): pass
    def write(self, data): pass
    def flush(self): pass
    def read(self, n): return b""
    def close(self): pass

try:
    motor_serial = serial.Serial(PORT, BAUD, timeout=0.1)
    MOCK_MODE = False
    time.sleep(0.05)
except Exception:
    print("[MOCK] No motor serial device found — running in MOCK mode.", flush=True)
    motor_serial = MockSerial()


def motor_checksum(data):
    return (~sum(data)) & 0xFF


def motor_write_reg(mid, reg, data_bytes):
    body = [0x03, reg] + list(data_bytes)
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(motor_checksum(pkt[2:]))
    motor_serial.reset_input_buffer()
    motor_serial.write(bytes(pkt))
    motor_serial.flush()
    time.sleep(0.01)
    return motor_serial.read(20)


def motor_read_reg(mid, reg, n_bytes):
    if MOCK_MODE:
        if reg == ADDR_PRESENT_POSITION and n_bytes == 2:
            pos = mock_positions.get(mid, 2048)
            return bytes([pos & 0xFF, (pos >> 8) & 0xFF])
        return bytes(n_bytes)
    body = [0x02, reg, n_bytes]
    length = len(body) + 1
    pkt = [0xFF, 0xFF, mid, length] + body
    pkt.append(motor_checksum(pkt[2:]))
    motor_serial.reset_input_buffer()
    motor_serial.write(bytes(pkt))
    motor_serial.flush()
    time.sleep(0.06)
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


def motor_write_raw_u16(mid, reg, value):
    value = max(0, min(65535, int(value)))
    return motor_write_reg(mid, reg, [value & 0xFF, (value >> 8) & 0xFF])


def motor_set_position_mode(mid):
    motor_write_u8(mid, ADDR_MODE, 0)
    motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1)


def motor_set_wheel_mode(mid):
    motor_write_u8(mid, ADDR_MODE, 1)
    motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1)


def motor_write_position(mid, position):
    position = max(0, min(4095, int(position)))
    if MOCK_MODE:
        mock_positions[mid] = position
        return b""
    # ST-series position command expects pos_L, pos_H, time_L, time_H.
    return motor_write_reg(mid, ADDR_GOAL_POSITION, [position & 0xFF, (position >> 8) & 0xFF, 0, 0])


# Enable torque on all configured motors and set a moderate default speed.
for mid in MOTOR_IDS:
    with lock:
        motor_set_position_mode(mid)
        motor_write_u16(mid, ADDR_GOAL_SPEED, 1500)

with lock:
    motor_write_u16(JAW_ID, ADDR_GOAL_SPEED, 1200)

DEFAULT_VOICE = "edge:en-IN-NeerjaExpressiveNeural"

EDGE_VOICES = [
    ("en-IN-NeerjaExpressiveNeural", "Neerja Expressive — Indian English female (best)"),
    ("en-IN-NeerjaNeural",           "Neerja — Indian English female"),
    ("hi-IN-SwaraNeural",            "Swara — Hindi female"),
    ("mr-IN-AarohiNeural",           "Aarohi — Marathi female"),
]


def discover_voices():
    out = []
    for vid, label in EDGE_VOICES:
        out.append({"id": "edge:" + vid, "label": label})
    return out


VOICES = discover_voices()
voice_lock = threading.Lock()
eyes_lock = threading.RLock()
eyes_serial = None
eyes_enabled = False

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

AI_MODES = {
    "receptionist": (
        "You are Neerja, a warm Indian English female robot receptionist. "
        "Greet visitors, answer briefly, ask one useful follow-up question, "
        "and keep replies natural for spoken TTS. Use 2 to 5 short sentences."
    ),
    "interviewer": (
        "You are Neerja, a professional mock interview coach. "
        "Ask one interview question at a time, then wait for the user's answer. "
        "If the user answered a previous question, give concise feedback and ask the next question. "
        "Keep it natural for spoken TTS."
    ),
    "presentation": (
        "You are Neerja, a confident presenter. Turn the user's topic into a short presentation segment. "
        "Use clear spoken language, 4 to 7 short sentences, and avoid markdown bullets."
    ),
}


def edge_synth_mp3(text, voice_id):
    async def _run():
        c = edge_tts.Communicate(text, voice_id)
        buf = io.BytesIO()
        async for chunk in c.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()
    return asyncio.run(_run())


def ollama_generate(prompt):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 180,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("response", "").strip()


def ensure_eyes_serial():
    global eyes_serial
    if eyes_serial and eyes_serial.is_open:
        return eyes_serial
    try:
        eyes_serial = serial.Serial(EYES_PORT, EYES_BAUD, timeout=0.05)
    except Exception:
        eyes_serial = MockSerial()
    return eyes_serial


def send_eyes(command, **kwargs):
    parts = [command]
    for key, value in kwargs.items():
        parts.append(f"{key}={value}")
    line = " ".join(parts) + "\n"
    try:
        with eyes_lock:
            s = ensure_eyes_serial()
            s.write(line.encode("utf-8"))
            s.flush()
        return True, line.strip()
    except Exception as e:
        return False, str(e)


app = Flask(__name__, static_folder=".")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/motors")
def motors():
    return jsonify({
        "ids": MOTOR_IDS,
        "min": POS_MIN,
        "max": POS_MAX,
        "limits": {str(k): v for k, v in LIMITS.items()},
        "jaw": {"id": JAW_ID, "max_open": JAW_OPEN, "full_close": JAW_CLOSED},
    })


@app.route("/api/position/<int:mid>", methods=["GET"])
def get_position(mid):
    if mid not in MOTOR_IDS:
        return jsonify({"error": "unknown motor"}), 404
    with lock:
        data = motor_read_reg(mid, ADDR_PRESENT_POSITION, 2)
    if len(data) != 2:
        return jsonify({"error": "no position response"}), 500
    pos = data[0] | (data[1] << 8)
    return jsonify({"id": mid, "position": pos})


@app.route("/api/position/<int:mid>", methods=["POST"])
def set_position(mid):
    if mid not in MOTOR_IDS:
        return jsonify({"error": "unknown motor"}), 404
    data = request.get_json(force=True)
    pos = int(data.get("position", 2048))
    lo, hi = LIMITS.get(mid, (POS_MIN, POS_MAX))
    pos = max(lo, min(hi, pos))
    with lock:
        motor_set_position_mode(mid)
        motor_write_position(mid, pos)
    return jsonify({"id": mid, "goal": pos})


@app.route("/api/torque/<int:mid>/<int:on>", methods=["POST"])
def torque(mid, on):
    if mid not in MOTOR_IDS:
        return jsonify({"error": "unknown motor"}), 404
    with lock:
        motor_write_u8(mid, ADDR_TORQUE_ENABLE, 1 if on else 0)
    return jsonify({"id": mid, "torque": bool(on)})


@app.route("/api/drive/<int:mid>", methods=["POST"])
def drive_motor(mid):
    if mid not in MOTOR_IDS:
        return jsonify({"error": "unknown motor"}), 404
    data = request.get_json(force=True)
    direction = str(data.get("direction", "stop"))
    speed_pct = max(0, min(100, int(data.get("speed", 0))))
    raw = int(speed_pct / 100 * 4095) & 0x7FFF
    if direction == "stop":
        raw = 0
    elif direction == "reverse":
        raw |= 0x8000
    elif direction != "forward":
        return jsonify({"error": "direction must be forward, reverse, or stop"}), 400
    with lock:
        motor_set_wheel_mode(mid)
        motor_write_raw_u16(mid, ADDR_GOAL_SPEED, raw)
    return jsonify({"id": mid, "direction": direction, "speed": speed_pct, "raw": raw})


_jaw_log_counter = 0
_jaw_max_amp = 0.0
_jaw_position_mode_ready = False


@app.route("/api/jaw", methods=["POST"])
def jaw():
    global _jaw_log_counter, _jaw_max_amp, _jaw_position_mode_ready
    data = request.get_json(force=True)
    amp = float(data.get("amplitude", 0.0))
    amp = max(0.0, min(1.0, amp))
    if amp > _jaw_max_amp:
        _jaw_max_amp = amp
    _jaw_log_counter += 1
    if _jaw_log_counter % 25 == 0:
        print(f"[jaw] amp={amp:.3f}  peak_in_window={_jaw_max_amp:.3f}", flush=True)
        _jaw_max_amp = 0.0
    pos = int(round(JAW_CLOSED - amp * (JAW_CLOSED - JAW_OPEN)))
    with lock:
        if not _jaw_position_mode_ready:
            motor_set_position_mode(JAW_ID)
            _jaw_position_mode_ready = True
        motor_write_position(JAW_ID, pos)
    return jsonify({"amplitude": amp, "position": pos})


@app.route("/api/jaw/config", methods=["POST"])
def jaw_config():
    global JAW_OPEN, JAW_CLOSED
    data = request.get_json(force=True)
    if "jaw_open" in data:
        JAW_OPEN = int(data["jaw_open"])
    if "jaw_close" in data:
        JAW_CLOSED = int(data["jaw_close"])
    
    try:
        c = {}
        if os.path.exists("jaw_config.json"):
            with open("jaw_config.json", "r") as f:
                c = json.load(f)
        c["jaw_open"] = JAW_OPEN
        c["jaw_close"] = JAW_CLOSED
        c["motor_id"] = JAW_ID
        with open("jaw_config.json", "w") as f:
            json.dump(c, f, indent=2)
            
        lo = min(JAW_OPEN, JAW_CLOSED)
        hi = max(JAW_OPEN, JAW_CLOSED)
        LIMITS[JAW_ID] = (max(0, lo - 300), min(4095, hi + 300))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"jaw_open": JAW_OPEN, "jaw_close": JAW_CLOSED})


@app.route("/api/voices")
def voices():
    return jsonify({"voices": VOICES, "default": DEFAULT_VOICE})


@app.route("/api/eyes", methods=["GET", "POST"])
def eyes():
    global eyes_enabled
    if request.method == "GET":
        return jsonify({
            "enabled": eyes_enabled,
            "port": EYES_PORT,
            "baud": EYES_BAUD,
            "protocol": "USB serial text: mode state=speaking/listening/idle, blink, look g13=<0..1> g14=<0..1>",
        })

    data = request.get_json(force=True)
    if "enabled" in data:
        eyes_enabled = bool(data["enabled"])
        if not eyes_enabled:
            send_eyes("mode", state="idle")
        return jsonify({"enabled": eyes_enabled})

    command = str(data.get("command", "mode")).strip()
    params = data.get("params", {})
    if not isinstance(params, dict):
        params = {}
    ok, detail = send_eyes(command, **params)
    return jsonify({"ok": ok, "detail": detail, "enabled": eyes_enabled})


@app.route("/api/tts")
def tts():
    text = request.args.get("text", "").strip()
    voice_id = request.args.get("voice", DEFAULT_VOICE)
    if not text:
        return jsonify({"error": "missing text"}), 400
    if not any(v["id"] == voice_id for v in VOICES):
        return jsonify({"error": f"unknown voice {voice_id}"}), 400
    engine, name = voice_id.split(":", 1)
    if engine == "edge":
        try:
            with voice_lock:
                mp3 = edge_synth_mp3(text, name)
        except Exception as e:
            return jsonify({"error": f"edge-tts failed: {e}"}), 502
        return Response(mp3, mimetype="audio/mpeg")
    return jsonify({"error": "unknown engine"}), 400


@app.route("/api/ai", methods=["POST"])
def ai_reply():
    data = request.get_json(force=True)
    mode = data.get("mode", "receptionist")
    user_text = str(data.get("text", "")).strip()
    history = data.get("history", [])
    if mode not in AI_MODES:
        return jsonify({"error": "unknown mode"}), 400
    if not user_text:
        return jsonify({"error": "missing text"}), 400

    recent = []
    if isinstance(history, list):
        for item in history[-6:]:
            if not isinstance(item, dict):
                continue
            who = "User" if item.get("role") == "user" else "Neerja"
            text = str(item.get("text", "")).strip()
            if text:
                recent.append(f"{who}: {text}")

    prompt = (
        AI_MODES[mode]
        + "\n\nConversation so far:\n"
        + ("\n".join(recent) if recent else "No previous conversation.")
        + f"\n\nUser: {user_text}\nNeerja:"
    )

    try:
        answer = ollama_generate(prompt)
    except (urllib.error.URLError, TimeoutError) as e:
        return jsonify({
            "error": "Ollama is not responding. Start it with: ollama serve",
            "detail": str(e),
        }), 503
    except Exception as e:
        return jsonify({"error": f"AI failed: {e}"}), 500

    return jsonify({"reply": answer, "model": OLLAMA_MODEL, "mode": mode})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
