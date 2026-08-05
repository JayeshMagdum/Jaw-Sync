"""
pc4_robot_client.py  (v4 - synchronous, no polling)
====================================================

Run this on PC4 (the robot PC) only.

Real architecture (confirmed from PC1/PC2/PC3 server code):
  PC4 --(1. POST wav)--> PC1 /transcribe
       PC1 internally calls PC2 /ask (STT -> RAG -> LLM) and returns
       the answer text in the SAME response (ollama_response.answer).
       PC1/PC2 do NOT forward anything to PC3 automatically.

  PC4 --(2. POST answer text)--> PC3 /generate-speech
       PC3 returns wav bytes directly in the response (no session_id,
       no polling - fully synchronous).

  PC4 --(3. play wav + jaw sync)

So PC4 makes exactly two sequential calls per turn: PC1 then PC3.
"""

import io
import os
import time
import wave
import logging
import requests
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PC1_SUBMIT_URL = "https://monogram-outright-joystick.ngrok-free.dev/transcribe"   # PC4 -> PC1 (wav in, answer text out)
PC3_TTS_URL = "https://patronize-gallows-deranged.ngrok-free.dev/generate-speech"  # PC4 -> PC3 (text in, wav out)

# PC3's SpeechRequest defaults to "hi" if lang is omitted. Nothing upstream
# (PC1/PC2) currently tells PC4 what language the answer is in, so this is
# a fixed default for now. Change if you add lang detection later.
DEFAULT_TTS_LANG = "hi"

REQUEST_TIMEOUT_PC1 = 60   # STT + RAG + LLM can take a while
REQUEST_TIMEOUT_PC3 = 60   # TTS synthesis

SAMPLE_RATE = 16000
CHANNELS = 1
# Adaptive VAD: silence is detected when RMS drops to within SILENCE_FLOOR_MULTIPLIER
# of the ambient noise floor (measured before you start speaking).
# 3.0 = speech threshold is 3x the ambient floor RMS.
# Raise this if the robot environment is very noisy and silence detection cuts off speech early.
# Lower it if silence is not being detected (recording always runs to the full 15s limit).
# NOTE: With high ambient noise (~25k RMS), a multiplier of 1.5 sets the threshold at ~37k,
#       which typical speech may not exceed. 3.0 gives a better separation from noise.
SILENCE_FLOOR_MULTIPLIER = 3.0
NOISE_CALIBRATION_SEC = 0.5   # How long to listen silently before recording to measure ambient noise
SILENCE_DURATION = 1.25       # Seconds of post-speech silence before stopping
MAX_RECORD_SECONDS = 15
# Minimum RMS of filtered audio before sending to PC1. If filters reduce audio below this
# level, the recording is likely over-subtracted and will return blank transcription.
MIN_FILTERED_RMS = 500.0

# NOTE ON LATENCY: If you're on the same LAN as PC1/PC3, switch the URLs above
# to direct static IPs (e.g. http://192.168.1.101:8000/transcribe) instead of
# ngrok tunnels. ngrok routes through the internet even on a local network and
# adds significant latency to every single request.

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_audio")
os.makedirs(WORK_DIR, exist_ok=True)

JAW_SYNC_ENABLED = True
JAW_MIN_PULSE = 500
JAW_MAX_PULSE = 800

# Noise Filter Settings
ENABLE_NOISE_FILTER = True
LOW_CUTOFF_HZ = 80.0       # High-pass filter threshold to remove DC offset, motor hum & sub-bass rumble
HIGH_CUTOFF_HZ = 7500.0    # Low-pass filter threshold to remove high-frequency static/hiss
ENABLE_SPECTRAL_SUBTRACTION = True # Reduces ambient background noise
PREEMPHASIS_ALPHA = 0.85   # Boosts speech frequencies; 0.85 is gentler than 0.97 to avoid clipping speech

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pc4_robot_client")

# Persistent HTTP session: reuses TCP/TLS connections across turns,
# avoiding a full handshake overhead on every POST (especially important over ngrok HTTPS).
_session = requests.Session()


class BlankAudioError(Exception):
    """Raised when PC1 detected no speech in the recording."""
    pass


# ---------------------------------------------------------------------------
# NOISE FILTERING PIPELINE
# ---------------------------------------------------------------------------

def apply_bandpass_filter(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, low_cutoff: float = LOW_CUTOFF_HZ, high_cutoff: float = HIGH_CUTOFF_HZ) -> np.ndarray:
    """Applies bandpass filtering (Butterworth if scipy available, otherwise smooth FFT filtering)."""
    try:
        from scipy.signal import butter, sosfilt
        nyquist = 0.5 * sample_rate
        low = max(0.001, low_cutoff / nyquist)
        high = min(0.999, high_cutoff / nyquist)
        sos = butter(4, [low, high], btype='bandpass', output='sos')
        filtered = sosfilt(sos, audio.astype(np.float32))
        return np.clip(filtered, -32768, 32767).astype(np.int16)
    except ImportError:
        audio_float = audio.astype(np.float32)
        n = len(audio_float)
        if n == 0:
            return audio
        fft_data = np.fft.rfft(audio_float)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        gain = np.ones_like(freqs)
        if low_cutoff > 0:
            hp_mask = freqs < low_cutoff
            gain[hp_mask] = 0.5 * (1 - np.cos(np.pi * freqs[hp_mask] / low_cutoff))
        if high_cutoff < sample_rate / 2:
            lp_mask = freqs > high_cutoff
            band = (sample_rate / 2) - high_cutoff
            if band > 0:
                rel = np.clip((freqs[lp_mask] - high_cutoff) / band, 0, 1)
                gain[lp_mask] = 0.5 * (1 + np.cos(np.pi * rel))
        filtered_fft = fft_data * gain
        filtered_audio = np.fft.irfft(filtered_fft, n=n)
        return np.clip(filtered_audio, -32768, 32767).astype(np.int16)


def apply_spectral_subtraction(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, noise_est_sec: float = 0.25) -> np.ndarray:
    """Removes stationary background noise using spectral subtraction."""
    try:
        import noisereduce as nr
        audio_float = audio.astype(np.float32)
        # prop_decrease=0.3: light reduction; keeps most of the speech intact
        # 0.8 wiped speech entirely, 0.5 still reduced too much (~60-73%)
        filtered = nr.reduce_noise(y=audio_float, sr=sample_rate, prop_decrease=0.3)
        return np.clip(filtered, -32768, 32767).astype(np.int16)
    except ImportError:
        audio_float = audio.astype(np.float32)
        n_samples = len(audio_float)
        n_noise = min(int(sample_rate * noise_est_sec), n_samples)
        if n_noise < 256:
            return audio
        noise_frame = audio_float[:n_noise]
        noise_fft = np.fft.rfft(noise_frame)
        noise_mag = np.abs(noise_fft) / len(noise_frame)

        frame_len = 512
        hop_len = 256
        if n_samples < frame_len:
            return audio
        window = np.hanning(frame_len)
        output = np.zeros(n_samples, dtype=np.float32)
        window_sum = np.zeros(n_samples, dtype=np.float32)

        for start in range(0, n_samples - frame_len + 1, hop_len):
            end = start + frame_len
            segment = audio_float[start:end] * window
            spec = np.fft.rfft(segment)
            mag = np.abs(spec)
            phase = np.angle(spec)

            interp_noise = np.interp(
                np.linspace(0, 1, len(mag)),
                np.linspace(0, 1, len(noise_mag)),
                noise_mag
            ) * len(segment)

            clean_mag = np.maximum(mag - 1.2 * interp_noise, 0.1 * mag)
            clean_spec = clean_mag * np.exp(1j * phase)
            clean_segment = np.fft.irfft(clean_spec, n=frame_len)

            output[start:end] += clean_segment * window
            window_sum[start:end] += window ** 2

        mask = window_sum > 1e-6
        output[mask] /= window_sum[mask]
        return np.clip(output, -32768, 32767).astype(np.int16)


def apply_preemphasis(audio: np.ndarray, alpha: float = PREEMPHASIS_ALPHA) -> np.ndarray:
    """Pre-emphasis filter to boost higher speech frequencies."""
    if alpha <= 0 or len(audio) == 0:
        return audio
    audio_float = audio.astype(np.float32)
    filtered = np.append(audio_float[0], audio_float[1:] - alpha * audio_float[:-1])
    return np.clip(filtered, -32768, 32767).astype(np.int16)


def apply_noise_filters(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Master filter function to clean microphone audio for STT."""
    if not ENABLE_NOISE_FILTER or len(audio) == 0:
        return audio

    log.info("Applying noise filters to recorded audio...")
    rms_before = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    filtered = audio.copy()

    # 1. Band-pass filter (remove hum, rumble, and high-frequency static)
    if LOW_CUTOFF_HZ > 0 or HIGH_CUTOFF_HZ < sample_rate / 2:
        filtered = apply_bandpass_filter(filtered, sample_rate, LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ)

    # 2. Spectral noise subtraction
    if ENABLE_SPECTRAL_SUBTRACTION:
        filtered = apply_spectral_subtraction(filtered, sample_rate)

    # 3. Speech pre-emphasis filter
    if PREEMPHASIS_ALPHA > 0:
        filtered = apply_preemphasis(filtered, PREEMPHASIS_ALPHA)

    rms_after = float(np.sqrt(np.mean(filtered.astype(np.float32) ** 2)))
    log.info(f"Filter RMS: {rms_before:.1f} -> {rms_after:.1f}")
    if rms_after < MIN_FILTERED_RMS:
        log.warning(
            f"Filtered audio RMS ({rms_after:.1f}) is below MIN_FILTERED_RMS ({MIN_FILTERED_RMS}). "
            "Noise filters may have over-subtracted. Using unfiltered audio instead."
        )
        return audio  # Fall back to unfiltered to avoid sending silence to STT

    return filtered


# ---------------------------------------------------------------------------
# STEP 0: record from mic
# ---------------------------------------------------------------------------

def record_audio(out_path: str) -> tuple[np.ndarray, str]:
    """Records audio until silence or MAX_RECORD_SECONDS.
    Uses adaptive VAD: calibrates ambient noise floor first, then stops recording
    when RMS drops back to near-floor level after speech.
    Returns (audio_array, out_path) so callers can upload from memory without re-reading the file.
    """
    block_duration = 0.25
    block_size = int(SAMPLE_RATE * block_duration)

    # -- Phase 1: Calibrate ambient noise floor (do NOT save these frames) --
    log.info(f"Calibrating ambient noise for {NOISE_CALIBRATION_SEC}s...")
    calibration_rms_vals = []
    n_calib_blocks = max(1, int(NOISE_CALIBRATION_SEC / block_duration))
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        for _ in range(n_calib_blocks):
            block, _ = stream.read(block_size)
            rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
            calibration_rms_vals.append(rms)

    noise_floor = float(np.mean(calibration_rms_vals))
    # Dynamic silence threshold: RMS must stay at or below this to count as silence
    silence_threshold = max(200.0, noise_floor * SILENCE_FLOOR_MULTIPLIER)
    log.info(f"Noise floor RMS: {noise_floor:.1f}  |  Silence threshold: {silence_threshold:.1f}")
    log.info("Recording... speak now.")

    # -- Phase 2: Record until silence or timeout --
    t0 = time.time()
    frames = []
    silence_time = 0.0
    total_time = 0.0
    speech_detected = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        while total_time < MAX_RECORD_SECONDS:
            block, _ = stream.read(block_size)
            frames.append(block.copy())
            rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
            total_time += block_duration

            if rms > silence_threshold:
                speech_detected = True
                silence_time = 0.0
            else:
                silence_time += block_duration

            # Only stop on silence AFTER we've heard at least 1s of audio
            # and some actual speech above the floor
            if total_time > 1.0 and speech_detected and silence_time >= SILENCE_DURATION:
                log.info(f"Silence detected after {total_time:.1f}s, stopping recording.")
                break

    audio = np.concatenate(frames, axis=0)

    # Apply noise filters before saving
    audio = apply_noise_filters(audio.flatten(), SAMPLE_RATE)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    # Always overwrite a fixed "last_recording.wav" so you can play it back
    # to check whether your voice was actually captured.
    last_rec_path = os.path.join(WORK_DIR, "last_recording.wav")
    import shutil
    shutil.copy2(out_path, last_rec_path)
    log.info(f"Recording done in {(time.time()-t0)*1000:.0f} ms -> {out_path}")
    log.info(f"Sample copy saved -> {last_rec_path}  (play to verify your voice was captured)")
    return audio, out_path


# ---------------------------------------------------------------------------
# STEP 1: submit audio to PC1, get answer text back (synchronous)
# ---------------------------------------------------------------------------

def _audio_array_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Serialises a numpy int16 audio array to WAV bytes in memory (no disk I/O)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def submit_to_pc1(wav_path: str, audio: np.ndarray = None) -> str:
    """
    Sends wav to PC1 /transcribe. PC1 internally calls PC2 /ask and returns
    the full response, shape:
      {
        "status": "success",
        "filename": ...,
        "processing_time": ...,
        "speech_to_text": "<transcribed question>",
        "ollama_response": {
            "status": "success",
            "answer": "<LLM answer text>",
            ...
        }
      }
    Returns the answer text (str). Raises if the chain failed anywhere.
    If `audio` numpy array is provided, uploads from memory (faster - avoids disk re-read).
    Falls back to reading wav_path from disk if audio is None.
    """
    log.info(f"[PC1] Submitting audio -> {PC1_SUBMIT_URL}")
    t0 = time.time()

    if audio is not None:
        # Upload directly from memory - avoids a file open() round-trip
        wav_bytes = _audio_array_to_wav_bytes(audio)
        files = {"file": (os.path.basename(wav_path), wav_bytes, "audio/wav")}
        resp = _session.post(PC1_SUBMIT_URL, files=files, timeout=REQUEST_TIMEOUT_PC1)
    else:
        with open(wav_path, "rb") as f:
            files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
            resp = _session.post(PC1_SUBMIT_URL, files=files, timeout=REQUEST_TIMEOUT_PC1)
    resp.raise_for_status()
    log.info(f"[PC1] Round-trip took {(time.time()-t0)*1000:.0f} ms")

    data = resp.json()
    log.info(f"[PC1] Transcribed: {data.get('speech_to_text', '')!r}")

    ollama_response = data.get("ollama_response") or {}
    if ollama_response.get("error") == "blank_audio":
        raise BlankAudioError("No speech detected in recording, skipping this turn.")
    if "error" in ollama_response:
        raise RuntimeError(f"[PC1->PC2] Ollama call failed: {ollama_response['error']}")

    answer = ollama_response.get("answer", "")
    if not answer.strip():
        raise RuntimeError(f"[PC1->PC2] Empty answer in response: {data}")

    log.info(f"[PC2] Answer: {answer!r}")
    return answer


# ---------------------------------------------------------------------------
# STEP 2: send answer text to PC3, get wav bytes back (synchronous)
# ---------------------------------------------------------------------------

def synthesize_on_pc3(text: str, out_wav_path: str, lang: str = DEFAULT_TTS_LANG) -> None:
    log.info(f"[PC3] Requesting TTS (lang={lang}) -> {PC3_TTS_URL}")
    t0 = time.time()
    resp = _session.post(
        PC3_TTS_URL,
        json={"text": text, "lang": lang},
        timeout=REQUEST_TIMEOUT_PC3,
    )
    resp.raise_for_status()
    log.info(f"[PC3] Round-trip took {(time.time()-t0)*1000:.0f} ms")

    with open(out_wav_path, "wb") as f:
        f.write(resp.content)
    log.info(f"[PC3] Audio received -> {out_wav_path} ({len(resp.content)} bytes)")


# ---------------------------------------------------------------------------
# STEP 3: play wav + jaw sync (RMS envelope based)
# ---------------------------------------------------------------------------

def _init_servo():
    if not JAW_SYNC_ENABLED:
        return None
    try:
        import serial
        return serial.Serial("COM7", 1000000, timeout=0.1)  # adjust port/baud
    except Exception as e:
        log.warning(f"Servo not initialized (jaw sync disabled): {e}")
        return None


def _set_jaw_position(ser, pulse: int):
    if ser is None:
        return
    try:
        # TODO: replace with your actual SCSerial write_pos() packet
        pass
    except Exception as e:
        log.debug(f"Servo write failed: {e}")


def play_with_jaw_sync(wav_path: str):
    log.info(f"Playing {wav_path} with jaw sync...")

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    ser = _init_servo()
    chunk_ms = 50
    chunk_size = int(sr * chunk_ms / 1000)

    sd.play(audio.astype(np.int16), sr)
    start_time = time.time()
    idx = 0
    total_samples = len(audio)

    try:
        while idx < total_samples:
            chunk = audio[idx: idx + chunk_size]
            if len(chunk) == 0:
                break
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            norm = min(rms / 3000.0, 1.0)
            pulse = int(JAW_MIN_PULSE + norm * (JAW_MAX_PULSE - JAW_MIN_PULSE))
            _set_jaw_position(ser, pulse)
            idx += chunk_size
            target_time = start_time + (idx / sr)
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        _set_jaw_position(ser, JAW_MIN_PULSE)
        sd.wait()
        if ser is not None:
            ser.close()

    log.info("Playback + jaw sync finished.")


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

LAST_RECORDING_PATH = os.path.join(WORK_DIR, "last_recording.wav")
LAST_RESPONSE_PATH  = os.path.join(WORK_DIR, "last_response.wav")


def play_audio_file(path: str):
    """Plays a WAV file for the user to hear (blocking). Uses sounddevice."""
    if not os.path.exists(path):
        log.warning(f"No audio file found at {path}")
        return
    log.info(f"Playing back: {path}")
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    sd.play(audio.astype(np.int16), sr)
    sd.wait()
    log.info("Playback finished.")


def run_pipeline(use_last: bool = False):
    """
    Runs one full turn of the pipeline.
    use_last=True  -> skips recording, reuses last_recording.wav
    use_last=False -> records fresh audio from mic
    """
    ts = int(time.time())
    response_wav = os.path.join(WORK_DIR, f"response_{ts}.wav")
    journey_start = time.time()

    if use_last:
        if not os.path.exists(LAST_RECORDING_PATH):
            log.warning("No last_recording.wav found. Falling back to fresh recording.")
            use_last = False
        else:
            log.info(f"[REUSE] Using last saved recording: {LAST_RECORDING_PATH}")

    if use_last:
        # Load from disk - no re-recording
        with wave.open(LAST_RECORDING_PATH, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(raw, dtype=np.int16)
        wav_path = LAST_RECORDING_PATH
    else:
        # Fresh recording
        user_wav = os.path.join(WORK_DIR, f"user_{ts}.wav")
        audio_array, wav_path = record_audio(user_wav)

    # PC1 (STT + RAG + LLM) -> answer text
    answer_text = submit_to_pc1(wav_path, audio=audio_array)

    # PC3 (TTS) -> wav bytes
    synthesize_on_pc3(answer_text, response_wav)

    # Keep a fixed copy so the user can replay it from the menu
    import shutil
    shutil.copy2(response_wav, LAST_RESPONSE_PATH)
    log.info(f"Response copy saved -> {LAST_RESPONSE_PATH}")

    journey_ms = (time.time() - journey_start) * 1000
    log.info(f"TOTAL journey time: {journey_ms:.0f} ms")

    # Play response with jaw sync
    play_with_jaw_sync(response_wav)


def prompt_next_turn(use_last: bool) -> tuple[str, bool]:
    """
    Shows the interactive menu between turns.
    Returns (action, new_use_last) where action is one of:
      'run'  - proceed with current mode
      'quit' - exit the loop
    """
    mode_label = "USE LAST AUDIO" if use_last else "RECORD"
    has_rec  = os.path.exists(LAST_RECORDING_PATH)
    has_resp = os.path.exists(LAST_RESPONSE_PATH)
    print(f"\n{'─'*55}")
    print(f"  Mode: [{mode_label}]")
    print(f"  [Enter] Run next turn in current mode")
    print(f"  [r]     Switch to RECORD mode")
    print(f"  [u]     Switch to USE LAST AUDIO mode")
    print(f"  [l]     Listen to last recording  {'(none yet)' if not has_rec else ''}")
    print(f"  [p]     Play last response         {'(none yet)' if not has_resp else ''}")
    print(f"  [q]     Quit")
    print(f"{'─'*55}")
    try:
        choice = input("  > ").strip().lower()
    except EOFError:
        return "run", use_last

    if choice == "q":
        return "quit", use_last
    elif choice == "r":
        log.info("Switched to RECORD mode.")
        return "run", False
    elif choice == "u":
        log.info("Switched to USE LAST AUDIO mode.")
        return "run", True
    elif choice == "l":
        play_audio_file(LAST_RECORDING_PATH)
        return prompt_next_turn(use_last)  # re-show menu after playback
    elif choice == "p":
        play_audio_file(LAST_RESPONSE_PATH)
        return prompt_next_turn(use_last)  # re-show menu after playback
    else:
        # Enter or unrecognised -> proceed with current mode
        return "run", use_last


if __name__ == "__main__":
    log.info("PC4 robot client (v4, synchronous, no polling) started. Press Ctrl+C to stop.")
    log.info(f"PC1 endpoint: {PC1_SUBMIT_URL}")
    log.info(f"PC3 endpoint: {PC3_TTS_URL}")

    use_last_audio = False  # start in RECORD mode by default

    try:
        while True:
            # Show menu FIRST — user decides what to do before any recording starts
            action, use_last_audio = prompt_next_turn(use_last_audio)
            if action == "quit":
                log.info("Stopped by user.")
                break

            try:
                run_pipeline(use_last=use_last_audio)
            except BlankAudioError as e:
                log.info(f"Skipping turn: {e}")
            except Exception as e:
                log.error(f"Turn failed, continuing to next turn: {e}")
    except KeyboardInterrupt:
        log.info("Stopped by user.")