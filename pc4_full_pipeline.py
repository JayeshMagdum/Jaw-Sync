"""
pc4_full_pipeline.py
=====================
Merged pipeline: local hardcoded FAQ fast-path, falling through to the
full 4-PC pipeline (PC1 STT+RAG/LLM -> PC3 TTS) when local confidence is low.

Flow per turn:
  1. Listen + transcribe LOCALLY using hardocoded_wav/stt_engine.py's
     already-tuned Whisper pipeline (domain corrections, hi/mr bias, etc.)
  2. Run the local FAQ matcher (mcad_keyword_matcher.py) against the transcript.
  3. If confidence >= LOCAL_CONF_THRESHOLD (same 0.20 threshold app.py already
     uses as LOW_CONF_WARN):
       -> FAST PATH: play the pre-recorded WAV directly.
          PC1/PC2/PC3 are never contacted for this turn.
  4. Else (local match too weak):
       -> ESCALATE: re-listen for a moment, POST that audio to PC1
          (which internally does its own STT + calls PC2 for RAG/LLM),
          get answer text back, POST that text to PC3 for TTS, get wav
          bytes back, play with real jaw-sync.

Every stage is timestamped and logged so you can directly compare
fast-path latency vs. full-escalation latency, and decide later whether
PC1/PC2 are worth keeping based on real numbers instead of guesses.

BEFORE RUNNING — 2 things to check on your machine:
  1. This script imports your PC4 client module by name below
     (see the `PC4_CLIENT_MODULE` line). If you saved it as something
     other than `pc4_robot_client2.py`, change that one line to match.
  2. Run this from the Jaw-Sync repo ROOT (same folder as jaw_config.json
     and jaw_synced_player.py), with hardocoded_wav/ as a subfolder —
     exactly your current D:\\JAW_SYNC\\JAW_SYNC layout.

Usage:
    (venv) PS D:\\JAW_SYNC\\JAW_SYNC> python pc4_full_pipeline.py
"""

import os
import sys
import time
import logging

import numpy as np

# ---------------------------------------------------------------------------
# PATH SETUP — makes both the root scripts and hardocoded_wav/ importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARDCODED_DIR = os.path.join(BASE_DIR, "hardocoded_wav")
sys.path.insert(0, HARDCODED_DIR)   # so `import stt_engine`, `import mcad_keyword_matcher` work
sys.path.insert(0, BASE_DIR)        # so `import jaw_synced_player` works

# >>> CHANGE THIS if your PC4 client file has a different name <<<
PC4_CLIENT_MODULE = "pc4_robot_client2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pc4_full_pipeline")

# ---------------------------------------------------------------------------
# IMPORTS — reusing your existing, already-tuned code as-is (no edits to
# any file inside hardocoded_wav/, per its own HARDCODED_WAV_RULES.txt)
# ---------------------------------------------------------------------------
log.info("Loading local STT + matcher (this also opens the mic + loads Whisper)...")
import stt_engine as stt                                    # noqa: E402
from mcad_keyword_matcher import MCADKeywordMatcher          # noqa: E402

log.info(f"Loading PC4 escalation client from '{PC4_CLIENT_MODULE}.py'...")
_pc4_client = __import__(PC4_CLIENT_MODULE)
submit_to_pc1 = _pc4_client.submit_to_pc1
synthesize_on_pc3 = _pc4_client.synthesize_on_pc3
BlankAudioError = _pc4_client.BlankAudioError
WORK_DIR = _pc4_client.WORK_DIR
DEFAULT_TTS_LANG = _pc4_client.DEFAULT_TTS_LANG

# Real, working jaw-sync playback (JawSyncedPlayer.play_wav) — used instead
# of the placeholder/stub _set_jaw_position() in the PC4 client, which does
# not actually write to the servo yet (`# TODO: replace with your actual
# SCSerial write_pos() packet` -> `pass`).
from jaw_synced_player import JawSyncedPlayer                 # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Same threshold hardocoded_wav/app.py already uses as LOW_CONF_WARN.
# Reusing it (instead of inventing a new number) means this gate has
# already been sanity-checked against your real FAQ data via test_matcher.py.
LOCAL_CONF_THRESHOLD = 0.20

CSV_PATH = os.path.join(HARDCODED_DIR, "mcad_solution_faq.csv")
AUDIO_DIR = os.path.join(HARDCODED_DIR, "faqmcad_wav")

_matcher = MCADKeywordMatcher(csv_path=CSV_PATH, audio_dir=AUDIO_DIR)
_jaw = JawSyncedPlayer()   # auto-loads jaw_config.json, auto-connects (or MOCK mode)


def _float32_to_int16(audio_flat: np.ndarray) -> np.ndarray:
    """stt_engine captures float32 in [-1, 1]; PC1 expects int16 PCM WAV."""
    return np.clip(audio_flat * 32768.0, -32768, 32767).astype(np.int16)


def run_one_turn() -> None:
    turn_start = time.time()

    # ---- 1. Listen + transcribe locally ------------------------------------
    t0 = time.time()
    text, lang = stt.listen_and_transcribe()
    local_stt_ms = (time.time() - t0) * 1000
    log.info(f"[LOCAL-STT] {local_stt_ms:.0f} ms | lang={lang} | text={text!r}")

    if not text.strip():
        log.info("No speech detected — skipping turn.")
        return

    # ---- 2. Local FAQ match --------------------------------------------------
    t0 = time.time()
    result = _matcher.match(text, lang)
    match_ms = (time.time() - t0) * 1000
    log.info(
        f"[MATCH] {match_ms:.0f} ms | confidence={result.confidence:.2f} | "
        f"topic={result.topic!r}"
    )

    if result.confidence >= LOCAL_CONF_THRESHOLD:
        # ---- 3. FAST PATH: play the hardcoded wav — PC1/PC2/PC3 untouched --
        wav_path = _matcher.audio_path(result.audio_file, result.lang)
        log.info(f"[FAST-PATH] Playing local answer: {wav_path}")
        _jaw.play_wav(str(wav_path))
        total_ms = (time.time() - turn_start) * 1000
        log.info(f"[TOTAL] fast-path turn: {total_ms:.0f} ms")
        return

    # ---- 4. ESCALATE: need fresh audio bytes to upload to PC1 ---------------
    log.info("[ESCALATE] Local confidence too low — falling back to PC1/PC2/PC3.")
    log.info("Ask your question again for the full pipeline...")

    t0 = time.time()
    audio_flat = stt._capture_utterance(stt.MAX_CHUNKS, stt.SILENCE_CHUNKS_NEEDED)
    recapture_ms = (time.time() - t0) * 1000

    if audio_flat is None:
        log.warning("No usable speech captured on escalation re-listen — aborting turn.")
        return
    log.info(f"[RE-CAPTURE] {recapture_ms:.0f} ms")

    audio_int16 = _float32_to_int16(audio_flat)
    ts = int(time.time())
    user_wav_path = os.path.join(WORK_DIR, f"escalate_user_{ts}.wav")
    response_wav_path = os.path.join(WORK_DIR, f"escalate_response_{ts}.wav")

    pc1_ms = None
    pc3_ms = None
    try:
        t0 = time.time()
        answer_text = submit_to_pc1(user_wav_path, audio=audio_int16)
        pc1_ms = (time.time() - t0) * 1000
        log.info(f"[PC1] {pc1_ms:.0f} ms | answer={answer_text!r}")

        tts_lang = lang if lang in ("hi", "mr", "en") else DEFAULT_TTS_LANG
        t0 = time.time()
        synthesize_on_pc3(answer_text, response_wav_path, lang=tts_lang)
        pc3_ms = (time.time() - t0) * 1000
        log.info(f"[PC3] {pc3_ms:.0f} ms")

        _jaw.play_wav(response_wav_path)

    except BlankAudioError as e:
        log.info(f"Skipping turn: {e}")
    except Exception as e:
        log.error(f"Escalation failed: {e}")

    total_ms = (time.time() - turn_start) * 1000
    pc1_display = f"{pc1_ms:.0f}" if pc1_ms is not None else "N/A"
    pc3_display = f"{pc3_ms:.0f}" if pc3_ms is not None else "N/A"
    log.info(
        f"[TOTAL] escalated turn: {total_ms:.0f} ms  "
        f"(local_stt={local_stt_ms:.0f} match={match_ms:.0f} "
        f"recapture={recapture_ms:.0f} pc1={pc1_display} pc3={pc3_display})"
    )


if __name__ == "__main__":
    log.info("PC4 merged pipeline started. Speak a question. Ctrl+C to stop.")
    try:
        while True:
            try:
                run_one_turn()
            except Exception as e:
                log.error(f"Turn failed, continuing to next turn: {e}")
            print("\n--- Ready for next turn ---\n")
    except KeyboardInterrupt:
        log.info("Stopped by user.")
        stt.close_mic()
