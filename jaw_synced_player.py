"""
jaw_synced_player.py — Synchronized Audio Playback and Hardware Servo Jaw Driver

Plays WAV audio through system speakers while simultaneously mapping speech RMS
amplitude envelopes to hardware SCServo jaw position commands over Serial.

Features:
- Direct system speaker audio output (sounddevice / soundfile).
- Natural pitch time-stretching (librosa phase-vocoder rate adjustment).
- Real-time amplitude envelope computation and motor command dispatch.
- Robust MockSerial fallback when no motor is connected.
- Exposes fine-tuning parameters (speed rate, motor lag, gain scale, noise floor).
"""

import os
import sys
import time
import json
import atexit
import signal
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "jaw_config.json")

# Default Config values
DEFAULT_CONFIG = {
    "motor_id": 1,
    "jaw_open": 2288,
    "jaw_close": 3145,
    "jaw_home": 3145,
    "speed": 1200,
    "audio_speed": 1.0,
    "motor_lag_ms": 20,
    "gain_scale": 1.8,
    "noise_floor": 0.02,
    "port": "COM3",
    "baud": 1000000,
}

# Servo Register Addresses
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_MODE = 33


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            cfg.update(data)
        except Exception as e:
            print(f"[JawPlayer] Could not read jaw_config.json: {e}")
    return cfg


class MockSerial:
    is_open = True
    def reset_input_buffer(self): pass
    def write(self, data): pass
    def flush(self): pass
    def read(self, n): return b""
    def close(self): pass


class JawSyncedPlayer:
    def __init__(self, config=None):
        self.cfg = config if config else load_config()
        self.serial = None
        self.mock_mode = True
        self._http_mode = False
        self._http_base = None
        self.serial_lock = threading.Lock()
        self._playing = False
        self._init_connection()
        self._register_shutdown_hooks()

    def _register_shutdown_hooks(self):
        """Register atexit and SIGINT handlers to home the jaw on any exit."""
        atexit.register(self.emergency_stop)

        # SIGINT = Ctrl+C. Chain to any previously registered handler.
        prev_handler = signal.getsignal(signal.SIGINT)
        def _handler(sig, frame):
            self.emergency_stop()
            if callable(prev_handler):
                prev_handler(sig, frame)
            else:
                raise KeyboardInterrupt
        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            pass  # Can't set signal in non-main thread — atexit still covers it

    def emergency_stop(self):
        """Immediately stop audio and return jaw to closed/home position."""
        self._playing = False
        try:
            sd.stop()
        except Exception:
            pass
        try:
            if self._http_mode:
                self._http_jaw(0.0)
            else:
                home = self.cfg.get("jaw_home", self.cfg.get("jaw_close", 3145))
                self.set_position(home)
                # Disable torque so motor is free if script crashes
                mid = self.cfg.get("motor_id", 1)
                if not self.mock_mode:
                    time.sleep(0.02)
                    self._write_reg(mid, ADDR_TORQUE_ENABLE, [0])
                    time.sleep(0.02)
                    self._write_reg(mid, ADDR_TORQUE_ENABLE, [1])
        except Exception:
            pass

    def _check_jaw_server(self, base_url="http://127.0.0.1:5050"):
        """Check if jaw_server.py is running and reachable via HTTP."""
        if not _HAS_URLLIB:
            return False
        try:
            req = urllib.request.Request(f"{base_url}/api/config", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                if "motor_id" in data:
                    return True
        except Exception:
            pass
        return False

    def _http_jaw(self, amplitude):
        """Send jaw amplitude command via jaw_server HTTP API."""
        if not self._http_base:
            return
        try:
            payload = json.dumps({"amplitude": float(amplitude)}).encode()
            req = urllib.request.Request(
                f"{self._http_base}/api/jaw",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.5)
        except Exception:
            pass

    def _init_connection(self):
        """Try jaw_server HTTP API first; fall back to direct serial."""
        server_url = self.cfg.get("jaw_server_url", "http://127.0.0.1:5050")
        if self._check_jaw_server(server_url):
            self._http_mode = True
            self._http_base = server_url
            self.mock_mode = False
            self.serial = MockSerial()  # placeholder, not used in HTTP mode
            print(f"[JawPlayer] Connected to jaw_server at {server_url} (HTTP mode — serial managed by server).")
            return

        # No server running — try direct serial as before
        self._init_serial()

    def _init_serial(self):
        try:
            import serial
        except ImportError:
            serial = None

        port = self.cfg.get("port", "COM3")
        baud = self.cfg.get("baud", 1000000)
        
        # Candidate ports to try
        candidates = [port]
        if serial is not None:
            try:
                import serial.tools.list_ports
                for p_info in serial.tools.list_ports.comports():
                    candidates.append(p_info.device)
            except Exception:
                pass
        if sys.platform.startswith("win"):
            candidates.extend([f"COM{i}" for i in range(1, 21)])
        else:
            candidates.extend([
                "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7B289175-if00",
                "/dev/ttyUSB0",
                "/dev/ttyUSB1",
            ])
        
        # Deduplicate while preserving order
        seen = set()
        ports_to_try = [p for p in candidates if not (p in seen or seen.add(p))]

        if serial is not None:
            for p in ports_to_try:
                try:
                    s = serial.Serial(p, baud, timeout=0.05)
                    self.serial = s
                    self.mock_mode = False
                    print(f"[JawPlayer] Connected to motor hardware on {p} at {baud} baud.")
                    break
                except serial.SerialException as e:
                    if "PermissionError" in str(e) or "Access is denied" in str(e):
                        print(f"[JawPlayer][WARNING] Could not open {p} — Port is currently locked by another program! ({e})")
                    continue
                except Exception as e:
                    continue

        if self.mock_mode:
            print("[JawPlayer] No motor hardware connected/available — running in MOCK mode (speaker audio enabled).")
            self.serial = MockSerial()

        self._setup_motor()

    def _checksum(self, data):
        return (~sum(data)) & 0xFF

    def _write_reg(self, mid, reg, data_bytes):
        body = [0x03, reg] + list(data_bytes)
        length = len(body) + 1
        pkt = [0xFF, 0xFF, mid, length] + body
        pkt.append(self._checksum(pkt[2:]))
        with self.serial_lock:
            try:
                self.serial.reset_input_buffer()
                self.serial.write(bytes(pkt))
                self.serial.flush()
            except Exception:
                pass

    def _setup_motor(self):
        mid = self.cfg.get("motor_id", 1)
        speed = self.cfg.get("speed", 1200)
        # Position mode initialization
        self._write_reg(mid, ADDR_TORQUE_ENABLE, [0])
        time.sleep(0.01)
        self._write_reg(mid, ADDR_MODE, [0])
        time.sleep(0.01)
        self._write_reg(mid, ADDR_TORQUE_ENABLE, [1])
        time.sleep(0.01)
        # Set speed
        self._write_reg(mid, ADDR_GOAL_SPEED, [speed & 0xFF, (speed >> 8) & 0xFF])

    def set_position(self, position):
        mid = self.cfg.get("motor_id", 1)
        pos = max(0, min(4095, int(position)))
        self._write_reg(mid, ADDR_GOAL_POSITION, [pos & 0xFF, (pos >> 8) & 0xFF])

    def move_jaw(self, amplitude):
        """
        amplitude: 0.0 (fully closed) .. 1.0 (fully open)
        """
        amp = max(0.0, min(1.0, float(amplitude)))
        if self._http_mode:
            self._http_jaw(amp)
            return
        j_open = self.cfg.get("jaw_open", 2288)
        j_close = self.cfg.get("jaw_close", 3145)
        pos = int(round(j_close - amp * (j_close - j_open)))
        self.set_position(pos)

    def time_stretch_audio(self, audio: np.ndarray, rate: float) -> np.ndarray:
        if rate == 1.0 or rate <= 0:
            return audio
        try:
            import librosa
            mono = audio[:, 0] if audio.ndim == 2 else audio
            stretched = librosa.effects.time_stretch(mono.astype(np.float32), rate=rate)
            if audio.ndim == 2:
                stretched = np.stack([stretched, stretched], axis=1)
            return stretched
        except Exception as e:
            print(f"[JawPlayer][warn] librosa time_stretch failed ({e}) — playing at normal rate.")
            return audio

    def play_wav(self, file_path, rate=None):
        """
        Plays WAV file to system speakers while concurrently driving jaw hardware.
        - rate: speed rate multiplier (e.g. 0.75 = 25% slower, 1.0 = normal).
        """
        # Reload config so any changes saved via jaw_dashboard or jaw_config.json
        # (jaw_open, jaw_close, speed, gain_scale, etc.) take effect immediately
        # without restarting the app.
        fresh_cfg = load_config()
        self.cfg.update(fresh_cfg)

        path_str = str(file_path)
        if not os.path.exists(path_str):
            print(f"[JawPlayer] File not found: {path_str}")
            return

        try:
            data, sr = sf.read(path_str, dtype="float32")
        except Exception as e:
            print(f"[JawPlayer] Error reading WAV {path_str}: {e}")
            return

        play_rate = rate if rate is not None else self.cfg.get("audio_speed", 1.0)
        j_open = self.cfg.get("jaw_open", 2288)
        j_close = self.cfg.get("jaw_close", 2900)
        print(f"[JawPlayer] Playing {os.path.basename(path_str)} (speed={play_rate}x, bounds: open={j_open}, close={j_close})", flush=True)
        data_stretched = self.time_stretch_audio(data, play_rate)

        # Hop size calculation for ~30ms motor updates
        hop_ms = 30
        hop_samples = int(sr * hop_ms / 1000.0)
        num_hops = int(np.ceil(len(data_stretched) / hop_samples))

        gain_scale = self.cfg.get("gain_scale", 1.5)
        noise_floor = self.cfg.get("noise_floor", 0.005)

        # Calculate amplitude envelope per hop (normalized to peak RMS)
        mono_data = data_stretched[:, 0] if data_stretched.ndim == 2 else data_stretched
        raw_rms = []
        for i in range(num_hops):
            chunk = mono_data[i * hop_samples : (i + 1) * hop_samples]
            if len(chunk) == 0:
                raw_rms.append(0.0)
            else:
                raw_rms.append(float(np.sqrt(np.mean(chunk ** 2))))

        peak_rms = max(raw_rms) if raw_rms and max(raw_rms) > 0 else 1.0

        envelopes = []
        for r in raw_rms:
            if r < noise_floor:
                amp = 0.0
            else:
                amp = min(1.0, (r / peak_rms) * gain_scale)
            envelopes.append(amp)

        # Audio stream callback player
        current_hop = 0
        
        def audio_callback(outdata, frames, time_info, status):
            nonlocal current_hop
            hop_idx = int((current_hop * hop_samples))
            end_idx = hop_idx + frames
            if hop_idx < len(data_stretched):
                chunk = data_stretched[hop_idx:end_idx]
                if len(chunk) < frames:
                    outdata[:len(chunk)] = chunk if chunk.ndim == 2 else chunk[:, np.newaxis]
                    outdata[len(chunk):] = 0
                else:
                    outdata[:] = chunk if chunk.ndim == 2 else chunk[:, np.newaxis]
            else:
                outdata[:] = 0

        # Synchronous playback & motor thread
        self._playing = True
        try:
            channels = 2 if data_stretched.ndim == 2 else 1

            with sd.OutputStream(samplerate=sr, channels=channels, dtype="float32") as stream:
                for i in range(num_hops):
                    if not self._playing:
                        break  # Ctrl+C / emergency_stop called
                    chunk = data_stretched[i * hop_samples : (i + 1) * hop_samples]
                    if len(chunk) == 0:
                        break

                    # Send motor command for this hop frame
                    amp = envelopes[i]
                    self.move_jaw(amp)

                    # Write audio chunk to speakers
                    chunk_out = chunk if chunk.ndim == 2 else chunk[:, np.newaxis]
                    stream.write(chunk_out)

        except KeyboardInterrupt:
            print("\n[JawPlayer] Interrupted — homing jaw.", flush=True)
            self.emergency_stop()
            raise  # Re-raise so the outer app sees the Ctrl+C
        except Exception as e:
            print(f"[JawPlayer] Audio playback error: {e}")
        finally:
            self._playing = False
            # Return jaw to home/close position after playback
            j_home = self.cfg.get("jaw_home", self.cfg.get("jaw_close", 3145))
            self.set_position(j_home)



# Global singleton instance
_player_instance = None
_player_lock = threading.Lock()

def get_synced_player():
    global _player_instance
    with _player_lock:
        if _player_instance is None:
            _player_instance = JawSyncedPlayer()
        return _player_instance


def play_wav_synced(path, rate=None):
    player = get_synced_player()
    player.play_wav(path, rate=rate)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Jaw Synced Player")
    parser.add_argument("--file", type=str, required=True, help="Path to WAV file")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback speed rate (e.g. 0.75)")
    args = parser.parse_args()

    print(f"Testing play_wav_synced on: {args.file} (rate={args.rate})")
    play_wav_synced(args.file, rate=args.rate)
