"""
gui_app.py — M CAD Solutions Dashboard
======================================
iOS-inspired unified dashboard featuring:
- Live camera face recognition panel (embedded OpenCV → Tkinter)
- Voice FAQ assistant (Record / Type / Auto-answer / Jaw sync)
- Real-time transcript + matched answer display
- Settings: Whisper model switcher, mic calibration, language select
- Status pill with colour-coded states
- Keep Listening toggle for hands-free continuous conversation

Run:
    python gui_app.py

Requires:
    pip install customtkinter sounddevice soundfile face_recognition opencv-python pillow
    (plus the same deps as before: faster-whisper, noisereduce, etc.)
"""
from __future__ import annotations

import os
import sys
import threading
import time
import queue
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk
from tkinter import messagebox

try:
    import customtkinter as ctk
except ImportError:
    print("[Startup] This UI needs customtkinter:\n    pip install customtkinter")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import numpy as np
import requests
from mcad_keyword_matcher import MCADKeywordMatcher

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CSV_PATH   = BASE_DIR / "mcad_solution_faq.csv"
AUDIO_DIR  = BASE_DIR / "faqmcad_wav"
RECORDED_WAV_PATH = BASE_DIR / "recorded_query.wav"
FACE_DIR   = BASE_DIR.parent / "known_faces"
GREETING_WAV_DIR = BASE_DIR / "output" / "en"

LOW_CONF_WARN = 0.20

LANG_PROMPTS = {
    "mr": ("CATIA SolidWorks UG NX BIW फिक्स्चर डिझाईन प्लेसमेंट फी "
           "बॅच साईझ सर्टिफिकेट कोर्सेस वेळापत्रक एनरोल डेमो M CAD Solutions"),
    "hi": ("CATIA SolidWorks UG NX BIW फिक्सचर डिज़ाइन प्लेसमेंट फीस "
           "बैच साइज़ सर्टिफिकेट कोर्स समय एनरोल डेमो M CAD Solutions"),
    "en": "CATIA V5 SolidWorks UG NX BIW placement batch certificate fees M CAD Solutions",
}

# ── Design Tokens (iOS-inspired dark) ─────────────────────────────────────
FONT_FAMILY       = "SF Pro Display" if sys.platform == "darwin" else "Segoe UI"
BG_ROOT           = "#0F0F13"
SIDEBAR_BG        = "#16161C"
CARD_BG           = "#1C1C24"
CARD_BORDER       = "#2A2A36"
TEXT_PRIMARY      = "#F5F5F7"
TEXT_SECONDARY    = "#8E8E93"
ACCENT            = "#007AFF"
ACCENT_HOVER      = "#0A84FF"
ACCENT_LIGHT      = "#1A3A5C"
SUCCESS           = "#30D158"
SUCCESS_BG        = "#0D2A18"
WARNING_COLOR     = "#FF9F0A"
WARNING_BG        = "#2D1F00"
DANGER            = "#FF453A"
DANGER_BG         = "#2D0E0C"
DANGER_HOVER      = "#FF6961"
SECONDARY_BG      = "#2C2C3A"
SECONDARY_HOVER   = "#3A3A4A"
PILL_RADIUS       = 14

STATUS_STYLES = {
    "blue":   {"bg": ACCENT_LIGHT,  "fg": ACCENT},
    "red":    {"bg": DANGER_BG,     "fg": DANGER},
    "orange": {"bg": WARNING_BG,    "fg": WARNING_COLOR},
    "green":  {"bg": SUCCESS_BG,    "fg": SUCCESS},
    "gray":   {"bg": SECONDARY_BG,  "fg": TEXT_SECONDARY},
}

RECOGNITION_TOLERANCE = 0.45
GREETING_COOLDOWN     = 30
CAMERA_INDEX          = 0
CAMERA_ROTATE         = None

# Selected audio devices (None = system default)
_selected_input_device  = None
_selected_output_device = None

# ── Neck & Eyes tracking is now handled by face_recognizer.py ─────────────
# We just receive the ZMQ video stream


def save_wav(path: Path, audio_flat: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
        sf.write(str(path), audio_flat, sample_rate)
    except ImportError:
        import wave
        int16 = (np.clip(audio_flat, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sample_rate); wf.writeframes(int16.tobytes())


# ══════════════════════════════════════════════════════════════════════════
# Recorder
# ══════════════════════════════════════════════════════════════════════════
class Recorder:
    DEFAULT_AMBIENT_RMS  = 0.01
    SILENCE_MULTIPLIER   = 3.0
    MIN_SILENCE_SEC      = 1.1
    MIN_SPEECH_SEC       = 0.25
    MAX_RECORD_SEC       = 20.0
    AMBIENT_EMA_ALPHA    = 0.1
    CHUNK_SEC            = 0.1

    def __init__(self, sample_rate: int, on_auto_stop=None):
        self.sample_rate = sample_rate
        self.on_auto_stop = on_auto_stop
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._active = False
        self._ambient_rms = self.DEFAULT_AMBIENT_RMS
        self._speech_sec = self._trailing_silence_sec = 0.0
        self._speech_started = self._auto_stop_fired = False
        self._elapsed_sec = 0.0

    def _callback(self, indata, frames, time_info, status):
        if not self._active: return
        chunk = indata[:, 0].copy()
        self._frames.append(chunk)
        if self._auto_stop_fired or len(chunk) == 0: return
        chunk_dur = len(chunk) / self.sample_rate
        self._elapsed_sec += chunk_dur
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        threshold = self._ambient_rms * self.SILENCE_MULTIPLIER
        if rms >= threshold:
            self._speech_started = True
            self._speech_sec += chunk_dur
            self._trailing_silence_sec = 0.0
        else:
            self._ambient_rms = ((1 - self.AMBIENT_EMA_ALPHA) * self._ambient_rms
                                  + self.AMBIENT_EMA_ALPHA * rms)
            if self._speech_started:
                self._trailing_silence_sec += chunk_dur
        should_stop = (
            (self._speech_started and self._speech_sec >= self.MIN_SPEECH_SEC
             and self._trailing_silence_sec >= self.MIN_SILENCE_SEC)
            or self._elapsed_sec >= self.MAX_RECORD_SEC)
        if should_stop:
            self._auto_stop_fired = True
            if self.on_auto_stop: self.on_auto_stop()

    def start(self):
        import sounddevice as sd
        self._frames = []
        self._active = True
        self._ambient_rms = self.DEFAULT_AMBIENT_RMS
        self._speech_sec = self._trailing_silence_sec = 0.0
        self._speech_started = False
        self._elapsed_sec = 0.0
        self._auto_stop_fired = False
        bs = max(1, int(self.sample_rate * self.CHUNK_SEC))
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                       dtype="float32", blocksize=bs,
                                       device=_selected_input_device,
                                       callback=self._callback)
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._active = False
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        if not self._frames: return np.array([], dtype=np.float32)
        return np.concatenate(self._frames)


# ══════════════════════════════════════════════════════════════════════════
# AudioPlayer
# ══════════════════════════════════════════════════════════════════════════
class AudioPlayer:
    def __init__(self):
        self._stop_event = threading.Event()
        self._playing = False
        self._lock = threading.Lock()

    def is_playing(self) -> bool:
        with self._lock: return self._playing

    def stop(self) -> None:
        self._stop_event.set()
        try:
            import sounddevice as sd; sd.stop()
        except Exception: pass
        try:
            jaw_sync_dir = str(BASE_DIR.parent)
            if jaw_sync_dir not in sys.path: sys.path.insert(0, jaw_sync_dir)
            import jaw_synced_player
            if hasattr(jaw_synced_player, "stop_playback"):
                jaw_synced_player.stop_playback()
        except Exception: pass

    def play(self, path: Path, rate: float = 1.0, on_done=None) -> None:
        if not path.exists():
            print(f"[Audio] MISSING: {path}")
            if on_done: on_done()
            return
        with self._lock: self._playing = True
        self._stop_event.clear()
        try:
            jaw_sync_dir = str(BASE_DIR.parent)
            if jaw_sync_dir not in sys.path: sys.path.insert(0, jaw_sync_dir)
            from jaw_synced_player import play_wav_synced
            play_wav_synced(path, rate=rate)
        except Exception as e:
            print(f"[Audio] Synced fallback ({e}). Standard play...")
            try:
                import soundfile as sf, sounddevice as sd
                data, sr = sf.read(str(path), dtype="float32")
                sd.play(data, sr, device=_selected_output_device)
                stream = sd.get_stream()
                while stream is not None and stream.active:
                    if self._stop_event.is_set(): sd.stop(); break
                    time.sleep(0.05)
            except Exception as err:
                print(f"[Audio] Error: {err}")
        finally:
            with self._lock: self._playing = False
            if on_done: on_done()


# ══════════════════════════════════════════════════════════════════════════
# Face recognizer state (Moved to face_recognizer.py)
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# Main GUI
# ══════════════════════════════════════════════════════════════════════════
class MCADGui:
    CAMERA_W = 540
    CAMERA_H = 380

    def __init__(self, root: "ctk.CTk", matcher: MCADKeywordMatcher):
        self.root = root
        self.matcher = matcher
        self.recorder: Recorder | None = None
        self.player = AudioPlayer()
        self.last_match = None
        self.msg_queue: queue.Queue = queue.Queue()
        self._suppress_autoresume = False

        self._cam = None
        self._cam_running = False
        self._cam_thread: threading.Thread | None = None
        self._cam_photo = None

        root.title("M CAD Solutions — Dashboard")
        root.geometry("1340x820")
        root.minsize(1100, 700)
        root.configure(fg_color=BG_ROOT)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()
        self._poll_queue()
        self._start_camera()

    # ── Style helpers ──────────────────────────────────────────────────────
    def _card(self, parent, **pack_kw):
        f = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=16,
                         border_color=CARD_BORDER, border_width=1)
        pkw = {"fill": "x", "padx": 0, "pady": (0, 12)}
        pkw.update(pack_kw)
        f.pack(**pkw)
        return f

    def _section_label(self, parent, text: str):
        ctk.CTkLabel(parent, text=text, font=(FONT_FAMILY, 11, "bold"),
                     text_color=TEXT_SECONDARY, anchor="w"
                     ).pack(fill="x", padx=16, pady=(12, 4))

    def _primary_btn(self, parent, text, cmd, **kw):
        d = dict(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=TEXT_PRIMARY,
                 corner_radius=PILL_RADIUS, height=38, font=(FONT_FAMILY, 13, "bold"))
        d.update(kw); return ctk.CTkButton(parent, text=text, command=cmd, **d)

    def _secondary_btn(self, parent, text, cmd, **kw):
        d = dict(fg_color=SECONDARY_BG, hover_color=SECONDARY_HOVER,
                 text_color=TEXT_PRIMARY, corner_radius=PILL_RADIUS, height=34,
                 font=(FONT_FAMILY, 12))
        d.update(kw); return ctk.CTkButton(parent, text=text, command=cmd, **d)

    def _danger_btn(self, parent, text, cmd, **kw):
        d = dict(fg_color=DANGER, hover_color=DANGER_HOVER, text_color=TEXT_PRIMARY,
                 corner_radius=PILL_RADIUS, height=34, font=(FONT_FAMILY, 12, "bold"))
        d.update(kw); return ctk.CTkButton(parent, text=text, command=cmd, **d)

    def _text_btn(self, parent, text, cmd, **kw):
        d = dict(fg_color="transparent", hover_color=SECONDARY_BG,
                 text_color=ACCENT, corner_radius=12, height=32,
                 font=(FONT_FAMILY, 12, "bold"), border_width=0)
        d.update(kw); return ctk.CTkButton(parent, text=text, command=cmd, **d)

    # ── Audio Devices card ─────────────────────────────────────────────────
    def _build_audio_devices_card(self, parent):
        import sounddevice as sd
        try:
            devices = sd.query_devices()
        except Exception:
            devices = []

        input_names  = ["System Default"]
        output_names = ["System Default"]
        self._input_device_map  = {0: None}   # display-index → sd device index
        self._output_device_map = {0: None}

        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                idx = len(input_names)
                self._input_device_map[idx] = i
                input_names.append(f"🎙 {d['name']}")
            if d["max_output_channels"] > 0:
                idx = len(output_names)
                self._output_device_map[idx] = i
                output_names.append(f"🔊 {d['name']}")

        card = self._card(parent)
        self._section_label(card, "AUDIO DEVICES")

        # Input mic row
        in_row = ctk.CTkFrame(card, fg_color="transparent")
        in_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(in_row, text="Mic", font=(FONT_FAMILY, 12),
                     text_color=TEXT_SECONDARY, width=44, anchor="w").pack(side="left")
        self._input_var = tk.StringVar(value=input_names[0])
        ctk.CTkOptionMenu(
            in_row, values=input_names, variable=self._input_var,
            command=self._on_input_change,
            fg_color=SECONDARY_BG, button_color=SECONDARY_BG,
            button_hover_color=SECONDARY_HOVER, text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_BG, font=(FONT_FAMILY, 12),
            dynamic_resizing=False, width=260
        ).pack(side="left", padx=(6, 0), fill="x", expand=True)

        # Output speaker row
        out_row = ctk.CTkFrame(card, fg_color="transparent")
        out_row.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(out_row, text="Speaker", font=(FONT_FAMILY, 12),
                     text_color=TEXT_SECONDARY, width=44, anchor="w").pack(side="left")
        self._output_var = tk.StringVar(value=output_names[0])
        ctk.CTkOptionMenu(
            out_row, values=output_names, variable=self._output_var,
            command=self._on_output_change,
            fg_color=SECONDARY_BG, button_color=SECONDARY_BG,
            button_hover_color=SECONDARY_HOVER, text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_BG, font=(FONT_FAMILY, 12),
            dynamic_resizing=False, width=260
        ).pack(side="left", padx=(6, 0), fill="x", expand=True)

    def _on_input_change(self, selected: str):
        global _selected_input_device
        for disp_idx, sd_idx in self._input_device_map.items():
            label = "System Default" if sd_idx is None else None
            if label is None:
                try:
                    import sounddevice as sd
                    label = f"🎙 {sd.query_devices(sd_idx)['name']}"
                except Exception:
                    label = str(sd_idx)
            if label == selected:
                _selected_input_device = sd_idx
                self._set_status(f"Mic → {selected.replace('🎙 ', '')}", "gray")
                return

    def _on_output_change(self, selected: str):
        global _selected_output_device
        for disp_idx, sd_idx in self._output_device_map.items():
            label = "System Default" if sd_idx is None else None
            if label is None:
                try:
                    import sounddevice as sd
                    label = f"🔊 {sd.query_devices(sd_idx)['name']}"
                except Exception:
                    label = str(sd_idx)
            if label == selected:
                _selected_output_device = sd_idx
                self._set_status(
                    f"Speaker → {selected.replace('🔊 ', '')}", "gray")
                return

    # ── Build UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top header bar

        header = ctk.CTkFrame(self.root, fg_color=SIDEBAR_BG, height=58,
                              corner_radius=0, border_color=CARD_BORDER, border_width=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="  M CAD Solutions",
                     font=(FONT_FAMILY, 17, "bold"),
                     text_color=TEXT_PRIMARY).pack(side="left", padx=24, pady=14)
        ctk.CTkLabel(header, text="Dashboard", font=(FONT_FAMILY, 13),
                     text_color=TEXT_SECONDARY).pack(side="left")

        self.status_badge = ctk.CTkLabel(
            header, text="● Idle",
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=STATUS_STYLES["blue"]["bg"],
            text_color=STATUS_STYLES["blue"]["fg"],
            corner_radius=PILL_RADIUS, height=28, padx=14)
        self.status_badge.pack(side="right", padx=24)

        # Two-column body
        body = ctk.CTkFrame(self.root, fg_color=BG_ROOT)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5, minsize=380)
        body.grid_columnconfigure(1, weight=6, minsize=480)
        body.grid_rowconfigure(0, weight=1)

        # ─── LEFT: voice controls ──────────────────────────────────────
        left_scroll = ctk.CTkScrollableFrame(body, fg_color=BG_ROOT,
                                              scrollbar_button_color=SECONDARY_BG)
        left_scroll.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        # Actions card
        actions = self._card(left_scroll)
        self._section_label(actions, "ACTIONS")
        grid = ctk.CTkFrame(actions, fg_color="transparent")
        grid.pack(fill="x", padx=12, pady=(0, 12))
        grid.grid_columnconfigure((0, 1), weight=1)

        self.record_btn = self._primary_btn(grid, "🎙  Record", self.on_record)
        self.record_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=4)

        self.stop_btn = self._secondary_btn(grid, "⏹  Stop Now", self.on_stop_recording)
        self.stop_btn.configure(state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=4)

        self.rerecord_btn = self._secondary_btn(grid, "🔁  Re-record", self.on_rerecord)
        self.rerecord_btn.configure(state="disabled")
        self.rerecord_btn.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=4)

        self.stop_play_btn = self._danger_btn(grid, "🔇  Stop Playback", self.on_stop_playback)
        self.stop_play_btn.configure(state="disabled")
        self.stop_play_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=4)

        # Voice card
        voice = self._card(left_scroll)
        self._section_label(voice, "VOICE")
        voice_row = ctk.CTkFrame(voice, fg_color="transparent")
        voice_row.pack(fill="x", padx=12, pady=(0, 4))

        self.listen_btn = self._text_btn(voice_row, "👂 Play My Voice", self.on_listen)
        self.listen_btn.configure(state="disabled")
        self.listen_btn.pack(side="left")

        self.replay_btn = self._text_btn(voice_row, "🔊 Replay Answer", self.on_play_answer)
        self.replay_btn.configure(state="disabled")
        self.replay_btn.pack(side="left", padx=(8, 0))

        switch_row = ctk.CTkFrame(voice, fg_color="transparent")
        switch_row.pack(fill="x", padx=12, pady=(4, 12))
        self.auto_listen_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(switch_row, text="Keep listening after each answer",
                      variable=self.auto_listen_var, onvalue=True, offvalue=False,
                      progress_color=ACCENT, font=(FONT_FAMILY, 12),
                      text_color=TEXT_PRIMARY).pack(anchor="w")

        # Type a question
        ask = self._card(left_scroll)
        self._section_label(ask, "OR TYPE A QUESTION")
        ask_row = ctk.CTkFrame(ask, fg_color="transparent")
        ask_row.pack(fill="x", padx=12, pady=(0, 12))

        self.lang_var = tk.StringVar(value="mr")
        ctk.CTkOptionMenu(ask_row, values=["hi", "mr", "en"], variable=self.lang_var,
                          width=60, fg_color=SECONDARY_BG, button_color=SECONDARY_BG,
                          button_hover_color=SECONDARY_HOVER, text_color=TEXT_PRIMARY,
                          dropdown_fg_color=CARD_BG,
                          font=(FONT_FAMILY, 12)).pack(side="left")

        self.type_entry = ctk.CTkEntry(
            ask_row, placeholder_text="Type your question…",
            fg_color=SECONDARY_BG, border_color=CARD_BORDER,
            corner_radius=12, font=(FONT_FAMILY, 13), text_color=TEXT_PRIMARY)
        self.type_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.type_entry.bind("<Return>", lambda e: self.on_type_submit())
        self._primary_btn(ask_row, "Ask", self.on_type_submit,
                          height=32, width=60).pack(side="left")

        # Settings card
        settings = self._card(left_scroll)
        self._section_label(settings, "SETTINGS")
        s_row = ctk.CTkFrame(settings, fg_color="transparent")
        s_row.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(s_row, text="Whisper model", font=(FONT_FAMILY, 12),
                     text_color=TEXT_SECONDARY).pack(side="left")
        self.model_var = tk.StringVar(value=self._current_model_name())
        ctk.CTkOptionMenu(s_row, values=["medium", "large-v3-turbo", "large-v3", "small"],
                          variable=self.model_var, command=self.on_model_change,
                          width=160, fg_color=SECONDARY_BG, button_color=SECONDARY_BG,
                          button_hover_color=SECONDARY_HOVER, text_color=TEXT_PRIMARY,
                          dropdown_fg_color=CARD_BG,
                          font=(FONT_FAMILY, 12)).pack(side="left", padx=(8, 12))
        self._text_btn(s_row, "Calibrate mic (3s)", self.on_calibrate).pack(side="left")

        s_row2 = ctk.CTkFrame(settings, fg_color="transparent")
        s_row2.pack(fill="x", padx=12, pady=(4, 12))
        self._secondary_btn(s_row2, "🔄 Reload Face DB", self._reload_faces,
                             height=30, font=(FONT_FAMILY, 11)).pack(side="left")
        self.cam_toggle_btn = self._secondary_btn(
            s_row2, "📷 Stop Camera", self._toggle_camera,
            height=30, font=(FONT_FAMILY, 11))
        self.cam_toggle_btn.pack(side="left", padx=(8, 0))

        # Audio Devices card
        self._build_audio_devices_card(left_scroll)

        # Transcript card
        transcript_card = self._card(left_scroll)
        self._section_label(transcript_card, "TRANSCRIPT")
        self.transcript_box = ctk.CTkTextbox(
            transcript_card, height=80, wrap="word", corner_radius=10,
            fg_color=BG_ROOT, font=(FONT_FAMILY, 12), text_color=TEXT_PRIMARY)
        self.transcript_box.pack(fill="x", padx=12, pady=(0, 12))
        self.transcript_box.configure(state="disabled")

        # Answer card
        answer_card = self._card(left_scroll, pady=(0, 4))
        self._section_label(answer_card, "MATCHED ANSWER")
        self.answer_box = ctk.CTkTextbox(
            answer_card, height=150, wrap="word", corner_radius=10,
            fg_color=BG_ROOT, font=(FONT_FAMILY, 13), text_color=TEXT_PRIMARY)
        self.answer_box.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.answer_box.configure(state="disabled")

        self.conf_var = tk.StringVar(value="")
        ctk.CTkLabel(answer_card, textvariable=self.conf_var,
                     font=(FONT_FAMILY, 11), text_color=TEXT_SECONDARY,
                     anchor="w").pack(fill="x", padx=12, pady=(0, 12))

        # ─── RIGHT: Camera ─────────────────────────────────────────────
        right_col = ctk.CTkFrame(body, fg_color=BG_ROOT)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

        cam_card = ctk.CTkFrame(right_col, fg_color=CARD_BG, corner_radius=20,
                                border_color=CARD_BORDER, border_width=1)
        cam_card.pack(fill="both", expand=True)

        cam_header = ctk.CTkFrame(cam_card, fg_color="transparent")
        cam_header.pack(fill="x", padx=16, pady=(14, 0))
        ctk.CTkLabel(cam_header, text="📷  Live Camera",
                     font=(FONT_FAMILY, 14, "bold"),
                     text_color=TEXT_PRIMARY, anchor="w").pack(side="left")

        self.face_label_var = tk.StringVar(value="No face detected")
        self.face_pill = ctk.CTkLabel(
            cam_header, textvariable=self.face_label_var,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=STATUS_STYLES["gray"]["bg"],
            text_color=STATUS_STYLES["gray"]["fg"],
            corner_radius=PILL_RADIUS, height=24, padx=10)
        self.face_pill.pack(side="right")

        self.cam_canvas = tk.Canvas(cam_card, bg="#0A0A0E", highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True, padx=14, pady=12)
        self.cam_canvas.create_text(
            self.CAMERA_W // 2, self.CAMERA_H // 2,
            text="Camera feed will appear here",
            fill=TEXT_SECONDARY, font=(FONT_FAMILY, 14))

        log_card = ctk.CTkFrame(cam_card, fg_color=BG_ROOT, corner_radius=10)
        log_card.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(log_card, text="Detection Log", font=(FONT_FAMILY, 11, "bold"),
                     text_color=TEXT_SECONDARY, anchor="w"
                     ).pack(fill="x", padx=10, pady=(6, 2))
        self.cam_log = ctk.CTkTextbox(log_card, height=90, font=(FONT_FAMILY, 11),
                                       fg_color=BG_ROOT, text_color=TEXT_PRIMARY,
                                       corner_radius=0, wrap="word")
        self.cam_log.pack(fill="x", padx=10, pady=(0, 8))
        self.cam_log.configure(state="disabled")

    # ── Camera ─────────────────────────────────────────────────────────────
    def _start_camera(self):
        try:
            import cv2
        except ImportError:
            self._log_cam("opencv-python not installed — camera disabled")
            return
        if not _HAS_PIL:
            self._log_cam("Pillow not installed — camera disabled (pip install pillow)")
            return
        self._cam_running = True
        self._cam_thread = threading.Thread(target=self._cam_worker, daemon=True)
        self._cam_thread.start()

    def _stop_camera(self):
        self._cam_running = False
        if self._cam_thread:
            self._cam_thread.join(timeout=2.0)
        try:
            if self._cam and self._cam.isOpened(): self._cam.release()
        except Exception: pass
        self._cam = None
        self.cam_canvas.delete("all")
        w = self.cam_canvas.winfo_width() or self.CAMERA_W
        h = self.cam_canvas.winfo_height() or self.CAMERA_H
        self.cam_canvas.create_text(w // 2, h // 2,
                                     text="Camera Off",
                                     fill=TEXT_SECONDARY, font=(FONT_FAMILY, 14))
        self.cam_toggle_btn.configure(text="📷 Start Camera")

    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self.cam_toggle_btn.configure(text="📷 Stop Camera")
            self._start_camera()

    def _reload_faces(self):
        self._log_cam("Face DB Reload is handled by face_recognizer.py now.")

    def _cam_worker(self):
        import cv2
        try:
            import zmq
        except ImportError:
            self.msg_queue.put(("cam_log", "ZMQ not installed. Please pip install pyzmq"))
            return

        context = zmq.Context()
        zmq_socket = context.socket(zmq.SUB)
        zmq_socket.connect("tcp://127.0.0.1:5555")
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Discard buffered frames immediately — only keep newest
        zmq_socket.setsockopt(zmq.RCVHWM, 1)
        zmq_socket.setsockopt(zmq.CONFLATE, 1)  # keep only latest message

        self.msg_queue.put(("cam_log", "Listening for ZMQ video stream..."))

        while self._cam_running:
            try:
                buffer = zmq_socket.recv(flags=zmq.NOBLOCK)
                frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), 1)
                if frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                # Replace any pending frame — no point rendering stale frames
                try:
                    self.msg_queue.get_nowait() if not self.msg_queue.empty() and \
                        not self.msg_queue.queue[0][0] != "cam_frame" else None
                except Exception:
                    pass
                self.msg_queue.put(("cam_frame", pil_img))
            except zmq.Again:
                time.sleep(0.005)
            except Exception as e:
                self.msg_queue.put(("cam_log", f"ZMQ Error: {e}"))
                time.sleep(1.0)

    def _update_cam_frame(self, pil_img):
        try:
            w = self.cam_canvas.winfo_width()
            h = self.cam_canvas.winfo_height()
            if w < 10 or h < 10: w, h = self.CAMERA_W, self.CAMERA_H
            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            self.cam_canvas.delete("all")
            self.cam_canvas.create_image(0, 0, anchor="nw", image=photo)
            self._cam_photo = photo
        except Exception: pass

    def _log_cam(self, text: str):
        self.cam_log.configure(state="normal")
        self.cam_log.insert("end", f"{text}\n")
        self.cam_log.see("end")
        self.cam_log.configure(state="disabled")

    # ── Record / Stop / Re-record ──────────────────────────────────────────
    def on_record(self):
        if self.player.is_playing():
            self._suppress_autoresume = True
            self.player.stop()
        import stt_engine
        self._set_status("Listening… answer comes after you stop talking", "red")
        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.rerecord_btn.configure(state="disabled")
        self.stop_play_btn.configure(state="disabled")
        self.listen_btn.configure(state="disabled")
        self.replay_btn.configure(state="disabled")
        self._set_text(self.transcript_box, "")
        self._set_text(self.answer_box, "")
        self.conf_var.set("")
        self.recorder = Recorder(
            stt_engine.SAMPLE_RATE,
            on_auto_stop=lambda: self.msg_queue.put(("auto_stop",)))
        self.recorder.start()

    def on_stop_recording(self):
        if self.recorder is None: return
        audio = self.recorder.stop()
        self.recorder = None
        self.stop_btn.configure(state="disabled")
        self.record_btn.configure(state="normal")
        if audio.size == 0:
            self._set_status("No audio captured — try again", "orange"); return
        import stt_engine
        save_wav(RECORDED_WAV_PATH, audio, stt_engine.SAMPLE_RATE)
        self.listen_btn.configure(state="normal")
        self.rerecord_btn.configure(state="normal")
        self._set_status("Transcribing...", "blue")
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def on_rerecord(self):
        self.last_match = None
        self._set_text(self.transcript_box, "")
        self._set_text(self.answer_box, "")
        self.conf_var.set("")
        self.rerecord_btn.configure(state="disabled")
        self.listen_btn.configure(state="disabled")
        self.replay_btn.configure(state="disabled")
        self.on_record()

    def on_listen(self):
        if RECORDED_WAV_PATH.exists():
            threading.Thread(target=self.player.play,
                             args=(RECORDED_WAV_PATH,), daemon=True).start()

    def on_play_answer(self):
        if self.last_match: self._start_answer_playback(self.last_match)

    def on_stop_playback(self):
        self._suppress_autoresume = True
        self.player.stop()

    def _start_answer_playback(self, match_result):
        path = self.matcher.audio_path(match_result.audio_file, match_result.lang)
        self.stop_play_btn.configure(state="normal")
        threading.Thread(
            target=self.player.play,
            kwargs={"path": path,
                    "on_done": lambda: self.msg_queue.put(("playback_done",))},
            daemon=True).start()

    # ── STT + matching ─────────────────────────────────────────────────────
    def _process_audio(self, audio_flat: np.ndarray):
        import stt_engine
        try:
            denoised = stt_engine._denoise(audio_flat, sr=stt_engine.SAMPLE_RATE, skip_if_clean=True)
            lang, prob = self._detect_lang(stt_engine, denoised)
            model = stt_engine._ensure_medium_loaded()
            prompt = LANG_PROMPTS.get(lang, "")
            # Use single beam and a single low temperature for speed
            beam = 1
            segs, info = model.transcribe(
                denoised,
                beam_size=beam,
                best_of=beam,
                language=lang,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False,
                initial_prompt=prompt,
                temperature=0,
                no_speech_threshold=0.45,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-0.6,
                word_timestamps=False)
            raw_text = " ".join(s.text.strip() for s in segs
                                if getattr(s, "avg_logprob", 0) >= -0.6).strip()
            text = stt_engine.apply_domain_corrections(raw_text)
            if lang != "mr" and text and stt_engine._is_marathi_by_transcript(text):
                lang = "mr"
            hallucinated = bool(text and stt_engine._has_repetition_loop(text))
            if hallucinated:
                text = ""
            match_result = self.matcher.match(text, lang) if text else None
            self.msg_queue.put(("result", raw_text, text, lang, prob, match_result, hallucinated))
        except Exception as e:
            self.msg_queue.put(("error", str(e)))

    def _detect_lang(self, stt_engine, denoised):
        try:
            _top, _prob, all_probs = stt_engine.whisper_small.detect_language(denoised)
            lang_prob_map = {c: p for c, p in all_probs}
            best_code, best_prob = None, 0.0
            for code, prob in all_probs:
                if code in stt_engine.SUPPORTED_LANGS and prob > best_prob:
                    best_code, best_prob = code, prob
            if best_code is not None:
                hi_p = lang_prob_map.get("hi", 0.0)
                mr_p = lang_prob_map.get("mr", 0.0)
                if best_code == "en" and (best_prob < 0.70 or (hi_p + mr_p) >= 0.20):
                    best_code = "mr" if mr_p >= hi_p else "hi"
                elif best_code == "hi" and mr_p > 0 and mr_p >= hi_p * 0.80:
                    best_code = "mr"
                if lang_prob_map.get(best_code, best_prob) < 0.20:
                    best_code = "mr"
            lang = best_code or "mr"
            return lang, lang_prob_map.get(lang, 1.0)
        except Exception:
            return "mr", 1.0

    # ── Typed question ─────────────────────────────────────────────────────
    def on_type_submit(self):
        raw = self.type_entry.get().strip()
        if not raw: return
        if self.player.is_playing():
            self._suppress_autoresume = True
            self.player.stop()
        lang = self.lang_var.get()
        self.type_entry.delete(0, "end")
        self._set_status("Matching...", "blue")
        match_result = self.matcher.match(raw, lang)
        self.msg_queue.put(("result", raw, raw, lang, 1.0, match_result, False))

    # ── Settings ───────────────────────────────────────────────────────────
    def _current_model_name(self) -> str:
        try:
            import stt_engine
            return stt_engine.HI_MR_MODEL_NAME
        except Exception:
            return "medium"

    def on_model_change(self, _value=None):
        import stt_engine
        stt_engine.set_himr_model_name(self.model_var.get())
        self._set_status(f"Model → '{self.model_var.get()}' (next transcribe)", "gray")

    def on_calibrate(self):
        self._set_status("Calibrating mic (3s, stay quiet)...", "orange")
        def run():
            import stt_engine
            stt_engine.calibrate_noise_floor(3.0)
            self.msg_queue.put(("calibrated",))
        threading.Thread(target=run, daemon=True).start()

    # ── Small helpers ──────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str = "blue"):
        s = STATUS_STYLES.get(color, STATUS_STYLES["blue"])
        self.status_badge.configure(text=f"● {text}",
                                     fg_color=s["bg"], text_color=s["fg"])

    def _set_text(self, box: "ctk.CTkTextbox", text: str):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    # ── Queue polling ──────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            # Always process non-cam items immediately; for cam_frame keep only latest
            for _ in range(5):
                item = self.msg_queue.get_nowait()
                if item[0] == "cam_frame":
                    # Drain all pending cam_frames, keep only the newest
                    latest = item
                    try:
                        while True:
                            nxt = self.msg_queue.get_nowait()
                            if nxt[0] == "cam_frame":
                                latest = nxt
                            else:
                                self._handle_queue_item(nxt)
                    except queue.Empty:
                        pass
                    self._handle_queue_item(latest)
                    break
                else:
                    self._handle_queue_item(item)
        except queue.Empty:
            pass
        self.root.after(16, self._poll_queue)  # ~60Hz polling

    def _handle_queue_item(self, item):
        kind = item[0]

        if kind == "cam_frame":
            self._update_cam_frame(item[1])

        elif kind == "cam_log":
            self._log_cam(item[1])

        elif kind == "face_detected":
            names = item[1]
            if names:
                label = "  ·  ".join(names)
                self.face_label_var.set(f"👤 {label}")
                self.face_pill.configure(
                    fg_color=STATUS_STYLES["green"]["bg"],
                    text_color=STATUS_STYLES["green"]["fg"])
                self._log_cam(f"[{time.strftime('%H:%M:%S')}] Detected: {label}")
            else:
                self.face_label_var.set("No face detected")
                self.face_pill.configure(
                    fg_color=STATUS_STYLES["gray"]["bg"],
                    text_color=STATUS_STYLES["gray"]["fg"])

        elif kind == "result":
            _, raw_text, text, lang, prob, match_result, hallucinated = item
            debug = f"Raw: {raw_text}\nCorrected: {text}\nLang: {lang} (p={prob:.2f})"
            if hallucinated: debug += "\n[Discarded — repetition-loop hallucination]"
            self._set_text(self.transcript_box, debug)
            self.last_match = match_result
            if match_result is None:
                self._set_text(self.answer_box, "(no match)")
                self.conf_var.set("")
                self.replay_btn.configure(state="disabled")
                self._set_status("No match — try again or type it", "orange")
            else:
                self._set_text(self.answer_box,
                                f"[{match_result.topic}]  {match_result.answer}")
                tag = "CONFIDENT" if match_result.confidence >= LOW_CONF_WARN else "LOW CONF"
                self.conf_var.set(
                    f"Confidence: {match_result.confidence:.2f} ({tag})  "
                    f"Audio: {match_result.audio_file} [{match_result.lang}]")
                self.replay_btn.configure(state="normal")
                self._set_status("Playing answer…", "green")
                self._start_answer_playback(match_result)

        elif kind == "auto_stop":
            self.on_stop_recording()

        elif kind == "playback_done":
            self.stop_play_btn.configure(state="disabled")
            if self._suppress_autoresume:
                self._suppress_autoresume = False
            elif self.auto_listen_var.get() and self.recorder is None:
                self.on_record()
            else:
                self._set_status("Idle — Record your next question", "blue")

        elif kind == "error":
            self._set_status("Error", "red")
            messagebox.showerror("Processing error", item[1])

        elif kind == "calibrated":
            self._set_status("Mic calibrated. Idle", "blue")

    # ── Clean exit ─────────────────────────────────────────────────────────
    def on_close(self):
        self._stop_camera()
        try:
            if self.recorder: self.recorder.stop()
        except Exception: pass
        try: self.player.stop()
        except Exception: pass
        try:
            import stt_engine
            stt_engine.close_mic()
        except Exception: pass
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════
def main():
    if not CSV_PATH.exists():
        print(f"[Startup] FAQ CSV not found at {CSV_PATH}")
        sys.exit(1)

    matcher = MCADKeywordMatcher(str(CSV_PATH), audio_dir=str(AUDIO_DIR))
    if not AUDIO_DIR.exists():
        print(f"[Startup] NOTE: audio folder {AUDIO_DIR} does not exist yet")

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = MCADGui(root, matcher)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
