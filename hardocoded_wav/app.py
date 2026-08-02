"""
app.py — M CAD Solutions voice FAQ assistant.

Pipeline (no RAG, no LLM, no live TTS):

    mic --(stt_engine.listen_and_transcribe)--> text, lang
        --(mcad_query_normalizer + mcad_keyword_matcher)--> best FAQ row
            --(play_wav)--> pre-recorded audio_file from output_new/<lang>/

Every answer is spoken by playing the WAV file named in the CSV's
audio_file column (e.g. greeting_hi.wav, catia_course_mr.wav) from
output_new/<lang>/. There is no text-to-speech step. If a WAV is
missing, the app prints the matched answer text and the expected
filename instead of crashing, so you can keep developing before all
recordings exist.

Usage
-----
    python app.py                # normal mic loop (STT -> match -> play wav)
    python app.py --text         # type questions instead of speaking
                                  # (useful for testing the matcher without
                                  # a mic / before Whisper models are set up)
    python app.py --calibrate    # measure mic noise floor (see stt_engine.py)

Layout expected (matches Suraj's actual folders)
-------------------------------------------------
    hardocoded_wav/
      app.py
      mcad_solution_faq.csv
      mcad_keyword_matcher.py
      mcad_query_normalizer.py
      stt_engine.py
      output_new/
        hi/
          greeting_hi.wav
          phone_hi.wav
          ...
        mr/
          greeting_mr.wav
          phone_mr.wav
          ...
        en/            (add later — English is on hold for now)

Only Hindi and Marathi are wired up for matching right now (see
ACTIVE_LANGS below). English rows in the CSV are left alone — flip
ACTIVE_LANGS back to include "en" whenever you're ready for it.
"""

from __future__ import annotations

import os
import sys

# Force UTF-8 for stdout/stderr — Windows terminals default to cp1252
# which cannot encode Devanagari (Hindi/Marathi) characters and raises
# UnicodeEncodeError when Whisper returns a Marathi transcript.
# reconfigure() is Python 3.7+ and is safe on non-Windows too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# MUST be set before faster_whisper / numpy / torch / ctranslate2 get
# imported anywhere (including transitively, via stt_engine below) —
# on Windows these each bundle their own OpenMP runtime (libiomp5md.dll
# vs libomp140.x86_64.dll), and loading both in one process raises
# "OMP: Error #15: Initializing libiomp5md.dll, but found ... already
# initialized." Setting this env var is Intel's own documented (if
# "unsupported") workaround — it's safe here since this app only ever
# runs single-threaded model calls (no simultaneous OpenMP workloads
# competing for the same threads).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

from mcad_keyword_matcher import MCADKeywordMatcher, MatchResult

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "mcad_solution_faq.csv"
AUDIO_DIR = BASE_DIR / "faqmcad_wav" # audio_path() appends /<lang>/<file>
# NOTE: was "faqmcad_wav\faq2" — a Windows-only backslash path (breaks on
# Linux/Mac) pointing at an older duplicate recording set. output_new/ is
# the folder this file's own docstring documents and matches the CSV's
# audio_file column (e.g. output_new/hi/greeting_hi.wav).

# Only these languages are "live" right now — English is on hold until
# Suraj asks for it. STT/typed input in any other language still gets
# transcribed/read, but is forced to fall back rather than match, so an
# accidental English utterance doesn't quietly answer from unrecorded
# English wavs.
ACTIVE_LANGS = ("hi", "mr", "en")

# Confidence below this is treated the same as a fallback for display
# purposes (matcher already returns the fallback row itself at 0.0, this
# is just for the console log tag).
LOW_CONF_WARN = 0.20


# ── WAV playback (no TTS) ──────────────────────────────────────────────────
# Imported lazily so `python app.py --text` (matcher-only testing) works
# even on a machine without sounddevice/an audio device configured.
PLAYBACK_SPEED_RATE = 1.0

def play_wav(path: Path) -> None:
    """Play a WAV file to system speakers while driving hardware jaw motor in sync."""
    if not path.exists():
        print(f"[Audio] MISSING wav (add the recording to fix this): {path}")
        return
    try:
        # Import synced player from JAW_SYNC directory
        jaw_sync_dir = str(BASE_DIR.parent)
        if jaw_sync_dir not in sys.path:
            sys.path.insert(0, jaw_sync_dir)
        from jaw_synced_player import play_wav_synced
        play_wav_synced(path, rate=PLAYBACK_SPEED_RATE)
    except Exception as e:
        print(f"[Audio] Synced player fallback ({e}). Playing standard WAV...")
        try:
            import soundfile as sf
            import sounddevice as sd

            data, samplerate = sf.read(str(path), dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as err:
            print(f"[Audio] Could not play {path.name}: {err}")


def speak_result(result: MatchResult, matcher: MCADKeywordMatcher) -> None:
    """Log the matched answer and play its WAV."""
    tag = "OK" if result.confidence >= LOW_CONF_WARN else "LOW-CONF"
    print(f"[Answer][{tag}] ({result.confidence:.2f}) [{result.lang}] {result.answer}")
    play_wav(matcher.audio_path(result.audio_file, result.lang))


def match_active_lang_only(matcher: MCADKeywordMatcher, text: str, lang: str) -> MatchResult:
    """Only hi/mr are wired up right now (see ACTIVE_LANGS). If STT/typed
    input comes in as anything else, force the fallback rather than
    matching against unrecorded-language wavs."""
    if lang not in ACTIVE_LANGS:
        print(f"[Matcher] lang={lang!r} is not active yet (ACTIVE_LANGS={ACTIVE_LANGS}) — using fallback.")
        lang = "hi"   # fallback text/wav still needs *a* language; hi is the safer default
        return matcher.match("", lang)
    return matcher.match(text, lang)


# ── Text mode (no mic — types questions instead) ───────────────────────────
def run_text_loop(matcher: MCADKeywordMatcher) -> None:
    print("\nText mode — type a question (hi/mr only for now). Ctrl+C to quit.\n")
    print("Tip: prefix with 'hi:' or 'mr:' to force a language, e.g. 'hi: sampark number kya hai'\n")
    try:
        while True:
            raw = input("You: ").strip()
            if not raw:
                continue
            lang = "hi"
            for prefix in ("en:", "hi:", "mr:"):
                if raw.lower().startswith(prefix):
                    lang = prefix[:-1]
                    raw = raw[len(prefix):].strip()
                    break
            result = match_active_lang_only(matcher, raw, lang)
            speak_result(result, matcher)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")


# ── Voice mode (mic -> STT -> match -> play wav) ───────────────────────────
def run_voice_loop(matcher: MCADKeywordMatcher) -> None:
    import stt_engine

    # stt_engine.py's DOMAIN_CORRECTIONS and initial_prompt now default to
    # real M CAD Solutions vocabulary (CATIA/SolidWorks/UG NX/BIW/placement/
    # fees/etc.) — no need to blank them out anymore. If you ever log a new
    # mishearing from the "[STT] [lang] ..." console line, add it straight
    # into stt_engine.py's DOMAIN_CORRECTIONS dict.

    print("\nVoice mode — speak in Hindi, Marathi, or English. Ctrl+C to quit.\n")
    try:
        while True:
            text, lang = stt_engine.listen_and_transcribe()
            if not text:
                print("(nothing detected)")
                continue
            print(f"\n[STT] [{lang}] {text}")
            result = match_active_lang_only(matcher, text, lang)
            speak_result(result, matcher)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            stt_engine.close_mic()
        except Exception:
            pass


RECORDED_WAV_PATH = BASE_DIR / "recorded_query.wav"


def _save_wav(path: Path, audio_flat: np.ndarray, sample_rate: int = 16000) -> None:
    """Save float32 mono audio to 16-bit WAV file using built-in wave module."""
    try:
        import soundfile as sf
        sf.write(str(path), audio_flat, sample_rate)
    except ImportError:
        import wave
        int16_data = (np.clip(audio_flat, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV file to float32 mono audio array using built-in wave module fallback."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
            float32_data = (int16_data / 32767.0).astype(np.float32)
            return float32_data, sr


def print_debug_panel(
    raw_text: str,
    corrected_text: str,
    lang: str,
    lang_prob: float,
    model_name: str,
    match_result,
    audio_sec: float | None = None,
    hallucination: bool = False,
) -> None:
    print("\n" + "=" * 66)
    print("                DEBUG TRANSCRIPTION & MATCH REPORT")
    print("=" * 66)
    if audio_sec is not None:
        print(f"  [Audio Clip]    : {audio_sec:.2f} seconds")
    print(f"  [Whisper Model] : {model_name}")
    print(f"  [Detected Lang] : {lang} (prob: {lang_prob:.2f})")
    print(f"  [Raw Whisper]   : \"{raw_text}\"")
    if hallucination:
        print(f"  [Hallucination] : YES — repetition loop detected, transcript discarded")
    elif corrected_text != raw_text:
        print(f"  [Domain Correct]: \"{corrected_text}\"")
    else:
        print(f"  [Domain Correct]: (no domain corrections needed)")

    print("-" * 66)
    if match_result:
        tag = "CONFIDENT" if match_result.confidence >= LOW_CONF_WARN else "LOW CONFIDENCE"
        if hallucination:
            print(f"  [Would Match]   : {match_result.topic} — but transcript was discarded (hallucination)")
        else:
            print(f"  [Matched Topic] : {match_result.topic} ({tag})")
        print(f"  [Confidence]    : {match_result.confidence:.2f}")
        print(f"  [Answer Text]   : \"{match_result.answer}\"")
        print(f"  [Answer Audio]  : {match_result.audio_file} [{match_result.lang}]")
    else:
        print("  [Matcher]       : (no match — empty or hallucinated transcript)")
    print("=" * 66)


# ── Interactive Menu Mode ──────────────────────────────────────────────────
def run_menu_loop(matcher: MCADKeywordMatcher) -> None:
    import numpy as np
    import stt_engine

    last_match_result: MatchResult | None = None
    last_text: str = ""
    last_lang: str = "mr"

    print("\n" + "=" * 64)
    print("      M CAD Solutions Voice Assistant — Debug Menu")
    print("=" * 64)

    def print_menu():
        current_m = stt_engine.HI_MR_MODEL_NAME
        print("\nMenu Controls:")
        print("  [r] Record new sound from mic")
        print("  [u] Reuse recorded sound (re-transcribe & match without recording again)")
        print("  [l] Listen to recorded sound (play back your voice recording)")
        print("  [p] Listen to final output (play back matched FAQ answer WAV)")
        print("  [t] Type question manually")
        print(f"  [c] Change Whisper model (Current: '{current_m}')")
        print("  [m] Show menu options")
        print("  [q] Quit")
        print("-" * 64)

    print_menu()
    try:
        while True:
            cmd = input("\nSelect option [r/u/l/p/t/c/m/q]: ").strip().lower()
            if not cmd:
                continue

            if cmd == "r":
                print("\n[*] Recording voice query... Speak now into your microphone!", flush=True)
                audio_flat = stt_engine._capture_utterance(stt_engine.MAX_CHUNKS, stt_engine.SILENCE_CHUNKS_NEEDED)
                if audio_flat is None or len(audio_flat) == 0:
                    print("[!] No speech detected from microphone.")
                    continue

                audio_sec = len(audio_flat) / stt_engine.SAMPLE_RATE
                _save_wav(RECORDED_WAV_PATH, audio_flat, stt_engine.SAMPLE_RATE)
                print(f"[Record] Saved recording to '{RECORDED_WAV_PATH.name}' ({audio_sec:.1f}s)")

                denoised = stt_engine._denoise(audio_flat, sr=stt_engine.SAMPLE_RATE)
                det_lang, det_prob = "mr", 1.0
                try:
                    _top_lang, _top_prob, all_probs = stt_engine.whisper_small.detect_language(denoised)
                    lang_prob_map = {c: p for c, p in all_probs}
                    best_code, best_prob = None, 0.0
                    for code, prob in all_probs:
                        if code in stt_engine.SUPPORTED_LANGS and prob > best_prob:
                            best_code, best_prob = code, prob
                    if best_code is not None:
                        hi_p = lang_prob_map.get("hi", 0.0)
                        mr_p = lang_prob_map.get("mr", 0.0)
                        # English guard: only accept en with very high confidence (>=0.70)
                        if best_code == "en" and (best_prob < 0.70 or (hi_p + mr_p) >= 0.20):
                            best_code = "mr" if mr_p >= hi_p else "hi"
                        # Marathi bias: if hi vs mr within 20%, prefer mr
                        elif best_code == "hi" and mr_p > 0 and mr_p >= hi_p * 0.80:
                            best_code = "mr"
                        # Low-confidence fallback
                        if lang_prob_map.get(best_code, best_prob) < 0.20:
                            best_code = "mr"
                    det_lang = best_code or "mr"
                    det_prob = lang_prob_map.get(det_lang, 1.0)
                except Exception:
                    det_lang, det_prob = "mr", 1.0

                lang = det_lang or "mr"
                model = stt_engine._ensure_medium_loaded()
                model_label = stt_engine.HI_MR_MODEL_NAME
                # Bug#5 fix: company name at end so it doesn't anchor decode start
                _prompts_r = {
                    "mr": ("CATIA SolidWorks UG NX BIW फिक्स्चर डिझाईन प्लेसमेंट फी "
                             "बॅच साईझ सर्टिफिकेट कोर्सेस वेळापत्रक एनरोल डेमो M CAD Solutions"),
                    "hi": ("CATIA SolidWorks UG NX BIW फिक्सचर डिज़ाइन प्लेसमेंट फीस "
                             "बैच साइज़ सर्टिफिकेट कोर्स समय एनरोल डेमो M CAD Solutions"),
                    "en": "CATIA V5 SolidWorks UG NX BIW placement batch certificate fees M CAD Solutions",
                }
                prompt = _prompts_r.get(lang, "")
                beam = 5 if lang in ("hi", "mr") else 1
                _temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Bug#3 fix

                print(f"[STT] Transcribing using Whisper {model_label} (detected lang={lang})...")
                segs, info = model.transcribe(
                    denoised,
                    beam_size=beam,
                    best_of=beam,
                    language=lang,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                    temperature=_temperatures,   # Bug#3 fix
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                )
                raw_text = " ".join(s.text.strip() for s in segs if getattr(s, 'avg_logprob', 0) >= -1.0).strip()
                text = stt_engine.apply_domain_corrections(raw_text)
                # Transcript-based Marathi override: if Marathi lexical markers
                # found in transcript, override hi→mr regardless of mel-classifier
                if lang != "mr" and text and stt_engine._is_marathi_by_transcript(text):
                    print(f"[STT] Marathi marker in transcript — overriding lang: {lang} → mr")
                    lang = "mr"
                is_hallucination = False
                if text and stt_engine._has_repetition_loop(text):
                    print(f"[STT] Discarding repetition-loop hallucination: {text[:80]!r}...")
                    is_hallucination = True
                    # Still run matcher on raw text so debug shows what it would have matched
                    last_match_result = match_active_lang_only(matcher, text, lang)
                    text = ""
                elif text:
                    last_match_result = match_active_lang_only(matcher, text, lang)
                else:
                    last_match_result = None

                last_text = text
                last_lang = lang

                print_debug_panel(
                    raw_text=raw_text,
                    corrected_text=text,
                    lang=lang,
                    lang_prob=det_prob,
                    model_name=model_label,
                    match_result=last_match_result,
                    audio_sec=audio_sec,
                    hallucination=is_hallucination,
                )
                print("--> Press 'p' to listen to the final output answer audio.")

            elif cmd == "u":
                if not RECORDED_WAV_PATH.exists():
                    print("[!] No recorded sound found yet. Use 'r' to record first.")
                    continue

                data, sr = _load_wav(RECORDED_WAV_PATH)
                audio_sec = len(data) / sr

                if sr != 16000:
                    num_samples = int(len(data) * 16000 / sr)
                    data = np.interp(np.linspace(0, len(data), num_samples, endpoint=False), np.arange(len(data)), data).astype(np.float32)

                # Bug#6 fix: loaded WAV was already denoised during recording;
                # skip_if_clean=True avoids double-denoising which removes
                # real Marathi phoneme energy from an already-clean recording.
                denoised = stt_engine._denoise(data, sr=16000, skip_if_clean=True)
                det_lang, det_prob = "mr", 1.0
                try:
                    _top_lang, _top_prob, all_probs = stt_engine.whisper_small.detect_language(denoised)
                    lang_prob_map = {c: p for c, p in all_probs}
                    best_code, best_prob = None, 0.0
                    for code, prob in all_probs:
                        if code in stt_engine.SUPPORTED_LANGS and prob > best_prob:
                            best_code, best_prob = code, prob
                    if best_code is not None:
                        hi_p = lang_prob_map.get("hi", 0.0)
                        mr_p = lang_prob_map.get("mr", 0.0)
                        # English guard: only accept en with very high confidence (>=0.70)
                        if best_code == "en" and (best_prob < 0.70 or (hi_p + mr_p) >= 0.20):
                            best_code = "mr" if mr_p >= hi_p else "hi"
                        # Marathi bias: if hi vs mr within 20%, prefer mr
                        elif best_code == "hi" and mr_p > 0 and mr_p >= hi_p * 0.80:
                            best_code = "mr"
                        # Low-confidence fallback
                        if lang_prob_map.get(best_code, best_prob) < 0.20:
                            best_code = "mr"
                    det_lang = best_code or "mr"
                    det_prob = lang_prob_map.get(det_lang, 1.0)
                except Exception:
                    det_lang, det_prob = "mr", 1.0

                lang = det_lang or "mr"
                model = stt_engine._ensure_medium_loaded()
                model_label = stt_engine.HI_MR_MODEL_NAME
                _prompts_u = {
                    "mr": ("CATIA SolidWorks UG NX BIW फिक्स्चर डिझाईन प्लेसमेंट फी "
                             "बॅच साईझ सर्टिफिकेट कोर्सेस वेळापत्रक एनरोल डेमो M CAD Solutions"),
                    "hi": ("CATIA SolidWorks UG NX BIW फिक्सचर डिज़ाइन प्लेसमेंट फीस "
                             "बैच साइज़ सर्टिफिकेट कोर्स समय एनरोल डेमो M CAD Solutions"),
                    "en": "CATIA V5 SolidWorks UG NX BIW placement batch certificate fees M CAD Solutions",
                }
                prompt = _prompts_u.get(lang, "")
                beam = 5 if lang in ("hi", "mr") else 1
                _temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

                print(f"[STT] Re-transcribing recorded sound using Whisper {model_label} (detected lang={lang})...")
                segs, info = model.transcribe(
                    denoised,
                    beam_size=beam,
                    best_of=beam,
                    language=lang,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                    temperature=_temperatures,   # Bug#3 fix
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                )
                raw_text = " ".join(s.text.strip() for s in segs if getattr(s, 'avg_logprob', 0) >= -1.0).strip()
                text = stt_engine.apply_domain_corrections(raw_text)
                # Transcript-based Marathi override: if Marathi lexical markers
                # found in transcript, override hi→mr regardless of mel-classifier
                if lang != "mr" and text and stt_engine._is_marathi_by_transcript(text):
                    print(f"[STT] Marathi marker in transcript — overriding lang: {lang} → mr")
                    lang = "mr"
                is_hallucination = False
                if text and stt_engine._has_repetition_loop(text):
                    print(f"[STT] Discarding repetition-loop hallucination: {text[:80]!r}...")
                    is_hallucination = True
                    last_match_result = match_active_lang_only(matcher, text, lang)
                    text = ""
                elif text:
                    last_match_result = match_active_lang_only(matcher, text, lang)
                else:
                    last_match_result = None

                last_text = text
                last_lang = lang

                print_debug_panel(
                    raw_text=raw_text,
                    corrected_text=text,
                    lang=lang,
                    lang_prob=det_prob,
                    model_name=model_label,
                    match_result=last_match_result,
                    audio_sec=audio_sec,
                    hallucination=is_hallucination,
                )
                print("--> Press 'p' to listen to the final output answer audio.")

            elif cmd == "c":
                print(f"\nCurrent Whisper Model: '{stt_engine.HI_MR_MODEL_NAME}'")
                print("Available Models:")
                print("  [1] medium           (Default — balanced speed & accuracy)")
                print("  [2] large-v3-turbo   (High speed + high accuracy for Marathi)")
                print("  [3] large-v3         (Maximum accuracy for Marathi / Devanagari)")
                print("  [4] small            (Lightweight / fast)")
                choice = input("Select model [1-4]: ").strip()
                model_map = {
                    "1": "medium",
                    "2": "large-v3-turbo",
                    "3": "large-v3",
                    "4": "small",
                }
                new_m = model_map.get(choice)
                if new_m:
                    stt_engine.set_himr_model_name(new_m)
                    print(f"[Model Switch] Changed Whisper model to '{new_m}'. (Will load on next transcribe)")
                else:
                    print("[!] Invalid choice.")

            elif cmd == "l":
                if not RECORDED_WAV_PATH.exists():
                    print("[!] No recorded sound found yet. Use 'r' to record first.")
                    continue
                print(f"\n[Playback] Playing back recorded voice clip ('{RECORDED_WAV_PATH.name}')...")
                play_wav(RECORDED_WAV_PATH)

            elif cmd == "p":
                if last_match_result is None:
                    print("[!] No matched FAQ result available yet. Use 'r', 'u', or 't' first.")
                    continue
                audio_file = matcher.audio_path(last_match_result.audio_file, last_match_result.lang)
                print(f"\n[Playback] Playing final output answer WAV ('{audio_file.name}')...")
                print(f"Answer text: {last_match_result.answer}")
                play_wav(audio_file)

            elif cmd == "t":
                raw = input("\nType your question: ").strip()
                if not raw:
                    continue
                lang = "mr"
                for prefix in ("en:", "hi:", "mr:"):
                    if raw.lower().startswith(prefix):
                        lang = prefix[:-1]
                        raw = raw[len(prefix):].strip()
                        break
                last_text = raw
                last_lang = lang
                last_match_result = match_active_lang_only(matcher, raw, lang)
                print_debug_panel(
                    raw_text=raw,
                    corrected_text=raw,
                    lang=lang,
                    lang_prob=1.0,
                    model_name="N/A (Typed Text)",
                    match_result=last_match_result,
                )
                print("--> Press 'p' to listen to the final output answer audio.")

            elif cmd == "m":
                print_menu()

            elif cmd == "q":
                print("\nExiting. Goodbye!")
                break
            else:
                print("Unknown option. Press 'm' to show menu options.")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting. Goodbye!")
    finally:
        try:
            stt_engine.close_mic()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="M CAD Solutions FAQ voice assistant")
    parser.add_argument("--debug", action="store_true", help="Run interactive debug menu [r/u/l/p/t/c/m/q]")
    parser.add_argument("--menu", action="store_true", help="Run interactive debug menu [r/u/l/p/t/c/m/q]")
    parser.add_argument("--model", type=str, default=None, help="Whisper model size for HI/MR (medium, large-v3-turbo, large-v3, small)")
    parser.add_argument("--text", action="store_true", help="Type questions instead of using the mic")
    parser.add_argument("--calibrate", action="store_true", help="Measure mic noise floor and exit")
    parser.add_argument("--csv", default=str(CSV_PATH), help="Path to the FAQ CSV")
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR),
                         help="Folder containing per-language wav subfolders (faqmcad_wav/hi, faqmcad_wav/mr, ...)")
    parser.add_argument("--audio-speed", type=float, default=1.0,
                         help="Playback speed rate multiplier (e.g. 0.75 = 25% slower)")
    args = parser.parse_args()

    global PLAYBACK_SPEED_RATE
    PLAYBACK_SPEED_RATE = args.audio_speed

    if args.model:
        import stt_engine
        stt_engine.set_himr_model_name(args.model)

    if args.calibrate:
        import stt_engine
        stt_engine.calibrate_noise_floor(3.0)
        stt_engine.close_mic()
        return

    if not Path(args.csv).exists():
        print(f"[Startup] FAQ CSV not found at {args.csv}")
        sys.exit(1)

    matcher = MCADKeywordMatcher(args.csv, audio_dir=args.audio_dir)

    if not Path(args.audio_dir).exists():
        print(f"[Startup] NOTE: audio folder {args.audio_dir} does not exist yet — "
              f"playback will be skipped until the WAV files are added there.")

    # Print current jaw configuration (Open & Close calibration numbers)
    try:
        jaw_sync_dir = str(BASE_DIR.parent)
        if jaw_sync_dir not in sys.path:
            sys.path.insert(0, jaw_sync_dir)
        from jaw_synced_player import load_config
        cfg = load_config()
        print("\n" + "=" * 64)
        print("  JAW MOTOR HARDWARE CONFIGURATION")
        print("=" * 64)
        print(f"  Motor ID     : {cfg.get('motor_id', 1)}")
        print(f"  COM Port     : {cfg.get('port', 'COM5')}")
        print(f"  Jaw Max Open : {cfg.get('jaw_open', 2288)}")
        print(f"  Jaw Full Close: {cfg.get('jaw_close', 2900)}")
        print(f"  Jaw Home     : {cfg.get('jaw_home', 2900)}")
        print(f"  Servo Speed  : {cfg.get('speed', 1200)}")
        print(f"  Audio Speed  : {PLAYBACK_SPEED_RATE}x")
        print("=" * 64 + "\n")
    except Exception as e:
        print(f"[Startup] Could not load jaw config: {e}")

    if args.debug or args.menu:
        run_menu_loop(matcher)
    elif args.text:
        run_text_loop(matcher)
    else:
        # Default behavior: run actual continuous voice loop
        run_voice_loop(matcher)


if __name__ == "__main__":
    main()