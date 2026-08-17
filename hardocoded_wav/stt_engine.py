"""
stt_engine.py
Thread-safe VAD mic capture + Whisper transcription.
Mic runs in a dedicated background thread feeding a queue —
eliminates Windows audio driver blocking on the main thread.
"""

import time
import re
import queue
import sys
import threading
import difflib
import numpy as np
import sounddevice as sd
import noisereduce as _nr
from faster_whisper import WhisperModel

# Force UTF-8 for stdout/stderr on Windows — cp1252 cannot encode Devanagari
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
CHUNK_MS           = 30
CHUNK_SIZE         = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 480 samples

# VAD thresholds — raise SPEECH_THRESH if capture keeps triggering on room
# noise (fan, AC, keyboard). Lower values = more sensitive.
# At 0.030 the mic needs a clear voice to start recording; at 0.012 a loud
# fan will trigger it. Tune with --calibrate if needed.
RMS_SPEECH_THRESH  = 0.035   # your speech peaks at 0.008, set trigger at half that
RMS_SILENCE_THRESH = 0.001  # was 0.007 — raised proportionally

SILENCE_CUTOFF_MS  = 900    # Bug#7 fix: 900ms — Marathi has longer natural inter-word pauses
MAX_RECORD_SEC     = 10     # accommodate longer multilingual questions
MIN_SPEECH_MS      = 250
PRE_ROLL_CHUNKS    = 6      # 180ms pre-roll before speech onset

SUPPORTED_LANGS    = ["hi", "mr", "en"]

SILENCE_CHUNKS_NEEDED = int(SILENCE_CUTOFF_MS / CHUNK_MS)
MAX_CHUNKS            = int(MAX_RECORD_SEC * 1000 / CHUNK_MS)

# Wake-word capture uses a shorter window — we only need a couple of words.
WAKE_SILENCE_CUTOFF_MS = 400
WAKE_MAX_RECORD_SEC    = 3.0
WAKE_SILENCE_CHUNKS_NEEDED = int(WAKE_SILENCE_CUTOFF_MS / CHUNK_MS)
WAKE_MAX_CHUNKS            = int(WAKE_MAX_RECORD_SEC * 1000 / CHUNK_MS)

# Wake word + common Whisper mis-transcriptions of "Adi" across en/hi/mr.
# Add more variants here if your users' accent triggers something else —
# check the console output (it prints what wake-check heard) and extend.
# "di"/"ad"/"k ad"-style fragments are too short/ambiguous to match safely
# (way too many false positives), so we stick to clear 3+ letter renderings.
WAKE_WORDS = {
    "adi", "addy", "audi", "aadi", "aadhi", "edi", "ardy", "adee", "addi",
    "आदि", "आदी", "आर्दी", "एडी",
}

# Fuzzy-match fallback: when the tiny model produces a short word that's NOT
# an exact WAKE_WORDS hit but is phonetically close (e.g. "adic", "edy"
# garbled from "adi"), accept it if its similarity ratio to "adi" clears this
# threshold. Kept narrow (only words WAKE_FUZZY_MIN_LEN-WAKE_FUZZY_MAX_LEN
# chars) so we don't start fuzzy-matching against unrelated longer words.
#
# 2026-06-22: minimum length raised from 2 to 3. At 2 chars, fragments like
# "ad" or "di" score the same similarity to "adi" (0.80) whether they're a
# genuine mangled "Adi" OR just coincidental syllables inside unrelated
# background speech ("ke ad", "ya di sini") — there's no way to tell them
# apart from string similarity alone. Real false wakes were observed from
# this in testing. Requiring 3+ chars (e.g. "adic", "edi") cuts most
# coincidental matches while still catching clearer mangled attempts.
WAKE_FUZZY_REF        = "adi"
WAKE_FUZZY_THRESHOLD  = 0.55
WAKE_FUZZY_MIN_LEN    = 3   # raised from 2 — see note above
WAKE_FUZZY_MAX_LEN    = 4   # only fuzzy-check short candidate words

# Tiny-model hallucinations on near-silence / faint room noise — Whisper
# confabulates short filler phrases like these when there's little or no
# real speech. If the WHOLE utterance (after normalising) is one of these,
# treat it as silence and skip wake-word checking entirely, rather than
# risk fuzzy-matching garbage into a false "Adi" trigger.
HALLUCINATION_PHRASES = {
    "yeah", "okay", "ok", "okay i did", "i did", "yes", "huh", "hmm",
    "thank you", "thanks for watching", "you", "the", "bye",
}


# ── Marathi-exclusive lexical markers ─────────────────────────────────────
# These words are grammatically exclusive to Marathi — they never appear
# in standard Hindi. Even a single marker in the transcript is enough to
# override Whisper's 'hi' detection to 'mr', since Whisper is biased toward
# Hindi due to its much larger Hindi training set.
#
# Examples:
#   "कोण आहे" (koni ahe) = "who is?"        — Marathi (Hindi: कौन है)
#   "आहे"          (ahe)     = "is"             — Marathi (Hindi: है)
#   "सांगा"         (sanga)   = "tell (imp.)"   — Marathi (Hindi: बताओ)
#   "कुठे"          (kuthe)   = "where"          — Marathi (Hindi: कहाँ)
#   "कोण"           (kon)     = "who"            — Marathi (Hindi: कौन)
MARATHI_LEXICAL_MARKERS: frozenset[str] = frozenset({
    # Verb “is/are” — most distinctive single-word markers
    "आहे", "आहेत",                     # is, are
    "नाही", "नाहीत",                  # is not, are not
    "होते", "होती", "होतो",           # was/were (past)
    "आले", "आली",                      # came (Marathi past)
    "देतात", "देतो", "देते",          # gives (Marathi verb)
    "मिळेल", "मिळते", "मिळतो",     # will get / gets
    "भेटेल", "भेटते",               # will meet / meets
    # Question words / interrogatives
    "कोण", "कोणी", "कोणाला", "कोणाचा", "कोणाची", "कोणाचे", "कोन्या", "कोनि",  # who
    "कुठे", "कुठं",                     # where
    "केव्हा", "कधी", "कधि",            # when
    "किती", "किति",                     # how much/many
    "कसला", "कसली", "कसले", "कसा", "कशी", "कसे", # how / what kind of
    "काय",                              # what (Marathi)
    "कोणकोणते", "कोणते", "कोणत्या", "कोणत्याही", "कोणत्यांमध्ये", # which / which ones
    # Conjunctions / particles / explanations
    "आणि",                             # and
    "पण",                               # but
    "म्हणजे", "मन्जे", "मंजे", "म्हन्जे", "मान्जे", # meaning/is/what is ("manje")
    "म्हणून",                          # therefore/saying
    "कायका", "म्हणजेकाय",              # what does it mean
    # Domain-specific phrases (appear in user queries)
    "बद्दल", "बद्द्ल",                 # about
    "सांगा", "सांग", "सांगावे",       # tell (imperative)
    "द्या",                              # give
    "करा",                             # do
    "घ्या",                              # take
    "पाहा",                             # look/see
    "निवडावे",                         # should choose
    "चालते", "चालतो",                  # works/runs
    "झाली", "झाला", "झालो",           # happened
    # Postpositions & suffixes (Marathi)
    "साठी", "सठी",                     # for
    "नंतर", "नंतरच",                   # after
    "पेक्षा",                          # than
    "पेक्षाही",                        # even than
    # Nouns that differ from Hindi equivalents
    "नाव",                              # name (Hindi: नाम)
    "माहिती",                          # information (Hindi: जानकारी)
    "वेळापत्रक",                       # schedule (Hindi: समय-सारणी)
    "शुल्क",                            # fee (Marathi form)
    "संस्था",                           # institute (Marathi)
    "ठिकाण", "पत्ता",                   # location / address
    # Pronouns & possessives
    "तुम्ही", "तुमचे", "तुमच्या", "तुमच्याकडे", # you / your (formal Marathi)
    "मी", "मला", "माझे", "माझ्या", "माझ्याकडे",  # I / me / my
    "त्यांचे", "त्यांची", "त्यांच्या", # their (Marathi)
    "आमच्या", "आमचे",                 # our (Marathi)
    "आपले", "आपल्या",                 # our/your (Marathi)
    # Postpositions (Marathi genitive markers)
    "च्या", "ची", "चे", "चा",          # 's (genitive markers)
})


def _is_marathi_by_transcript(text: str) -> bool:
    """
    Return True if the transcript contains at least one Marathi-exclusive
    lexical marker or ending. Used to override Whisper's 'hi' or 'en' detection to 'mr'
    when the decoded text itself betrays Marathi grammar/lexicon.
    """
    words = set(re.findall(r'[\u0900-\u097F]+', text.lower()))
    # Pass 1: whole-word match against full marker set
    if words & MARATHI_LEXICAL_MARKERS:
        return True
    # Pass 2: substring match for short fusible Marathi postpositions & interrogative phrases
    _FUSIBLE_SUFFIXES = (
        "च्या", "ची", "चे", "चा",   # genitive markers
        "ला", "ना",                  # dative / negation suffix
        "आहे", "नाही",               # copula (can fuse with preceding particle)
        "म्हणजे", "मन्जे", "मंजे",   # manje ("what is")
    )
    for suffix in _FUSIBLE_SUFFIXES:
        if suffix in text:
            return True
    # Pass 3: word-ending Marathi case suffixes.
    _WORD_END_SUFFIXES = ("च्या", "ची", "चे", "चा", "ला", "ने", "ना", "तो", "ते")
    for word in words:
        for suffix in _WORD_END_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix):
                return True
    return False


# ── Domain correction dictionary ──────────────────────────────────────────
# Whisper (small) mishears a fixed set of MMCOE-domain terms — institution
# names, exam codes, branch abbreviations — in predictable ways. Fixing them
# here (post-transcription, pre-matching) costs ~0 ms and catches exactly
# the recurring failure pattern without touching model size or latency.
#
# Format:  "whisper_output_substring": "corrected_form"
# Rules applied via apply_domain_corrections() after transcribe().
#
# HOW TO ADD NEW ENTRIES:
#   1. Run adi_pipeline.py with console logging and notice the raw transcript
#      printed at "[STT] [lang] …" — that's what Whisper actually outputs.
#   2. Add the misspelled form → correct form below.
#   3. No restart needed if you reload the module; or just restart the pipeline.
#
# CASE: all keys and the input are lowercased before matching (case-insensitive).
# ORDERING: longer keys first inside each group so "mmcoe karvenagar" is
#   corrected before a shorter "mmcoe" key would accidentally clobber it.
#   The dict is applied in insertion order (Python 3.7+), longest-first within
#   each semantic cluster.
#
# DEVANAGARI: add native-script mishearings in the same dict — the normaliser
#   feeds lowercased NFC text, so Devanagari keys match as-is.
#
# ── M CAD Solutions domain mishearings ────────────────────────────────────
# NOTE (2026-07-30): this dict previously contained MMCOE college-admission
# corrections copy-pasted from the Adi/MMCOE robot project. None of those
# terms ("MMCOE", "hostel", "JEE", "AICTE", ...) ever appear in an M CAD
# Solutions query, so that table did nothing useful here and, worse, the
# matching initial_prompt further down was ALSO still MMCOE-flavoured —
# that's the actual source of "biased transcription": Whisper's beam
# search was being primed with the wrong domain's vocabulary. Both are
# now replaced with real M CAD Solutions terms (CAD software names,
# course/placement/fees vocabulary). Extend this the same way the old
# file documented: watch the "[STT] [lang] ..." console line during real
# runs and add the exact garbled form -> corrected form below.
DOMAIN_CORRECTIONS: dict[str, str] = {
    # ── Company name ──────────────────────────────────────────────────────
    "m cad":                "M CAD",
    "em cad":               "M CAD",
    "m kaad":               "M CAD",
    "mkaad":                "M CAD",
    "m c a d":              "M CAD",
    "m cad solution":       "M CAD Solutions",
    "एम कैड":               "M CAD",
    "एम कॅड":               "M CAD",
    "एम सीएडी":             "M CAD",

    # ── CATIA ─────────────────────────────────────────────────────────────
    "catia":                "CATIA",
    "katia":                "CATIA",
    "caatia":               "CATIA",
    "cutiya":               "CATIA",       # common phonetic garble
    "catya":                "CATIA",
    "cat v5":               "CATIA V5",
    "kat v5":               "CATIA V5",
    # Marathi/Hindi phonetic garbles of CATIA observed from live Whisper output
    "केरिटिया":             "CATIA",       # observed: 'केरिटिया कोर्स बद्दल'
    "कॅटिया":               "CATIA",
    "केटियाय":              "CATIA",
    "करिटिया":              "CATIA",
    "केटिया":               "CATIA",
    "कटिआ":                 "CATIA",
    "catia v5":             "CATIA V5",
    "केटिया":               "CATIA",
    "कटिया":                "CATIA",

    # ── SolidWorks ────────────────────────────────────────────────────────
    "solid works":          "SolidWorks",
    "solidwork":            "SolidWorks",
    "solid work":           "SolidWorks",
    "साॅलिडवर्क्स":          "SolidWorks",
    "सॉलिड वर्क्स":         "SolidWorks",
    "सॉलिडवर्क":            "SolidWorks",

    # ── UG NX / Unigraphics ───────────────────────────────────────────────
    "you gee nx":           "UG NX",
    "u g n x":              "UG NX",
    "unigraphics":          "UG NX",
    "यू जी एनएक्स":          "UG NX",
    "युजी एनएक्स":           "UG NX",

    # ── BIW / fixture design ──────────────────────────────────────────────
    "b i w":                "BIW",
    "body in white":        "BIW",
    "बीआईडब्ल्यू":          "BIW",
    "बॉडी इन व्हाइट":       "BIW",

    # ── GD&T / OEM ────────────────────────────────────────────────────────
    "g d and t":            "GD&T",
    "g d t":                "GD&T",
    "o e m":                "OEM",

    # ── Industry 4.0 / ROS2 / digital twin ────────────────────────────────
    "industry four point o": "Industry 4.0",
    "industry four o":       "Industry 4.0",
    "r o s two":            "ROS2",
    "r o s2":               "ROS2",
    "digital twin":         "digital twin",  # NFC/no-op, kept for visibility

    # ── Placement ─────────────────────────────────────────────────────────
    "placemant":            "placement",
    "placment":             "placement",
    "प्लेसमेन्ट":            "प्लेसमेंट",
    "प्लेसमेंत":             "प्लेसमेंट",

    # ── Phonetic Marathi question words & garbles ───────────────────────────
    "मन्जे काय":            "म्हणजे काय",
    "मंजे काय":             "म्हणजे काय",
    "म्हन्जे काय":          "म्हणजे काय",
    "मान्जे काय":           "म्हणजे काय",
    "मन्जे":                "म्हणजे",
    "मंजे":                 "म्हणजे",
    "म्हन्जे":              "म्हणजे",
    "मान्जे":               "म्हणजे",
    "कोन्या":               "कोणी",
    "कोनि":                 "कोणी",
    "कधि":                 "कधी",
    "कुठं":                 "कुठे",
    "किति":                 "किती",
    "सांगावे":              "सांगा",
    "प्याच्चा":             "च्या",

    # ── Course / certificate / batch ──────────────────────────────────────
    "कोर्सेज":               "कोर्स",
    "सर्टिफिकेशन":           "सर्टिफिकेट",
    "certification":        "certificate",
    "बॅच साइज":              "बॅच साईझ",
    "बैच साइज":              "बैच साइज़",
}

# Pre-compile a single-pass regex for all corrections.
# Sorts by length descending so longer phrases match before their shorter
# substrings (e.g. "m m c o e" before "mmc").
_CORRECTION_PATTERN: re.Pattern | None = None

def _build_correction_pattern() -> re.Pattern:
    """Build and cache the compiled regex for domain corrections.

    Uses (?<![\\w\\u0900-\\u097F]) / (?![\\w\\u0900-\\u097F]) instead of \\b
    so that keys ending in punctuation (e.g. "m.m.c.o.") still match when
    followed by a space or end-of-string.  \\b requires one word-char and one
    non-word-char on either side; a trailing '.' is non-word on BOTH sides
    (dot + space) so \\b never fires there.
    """
    sorted_keys = sorted(DOMAIN_CORRECTIONS.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in sorted_keys]
    # Word-boundary substitute that handles Unicode + punctuation-terminated keys
    _WB_L = r'(?<![^\W\s\u0900-\u097F])'   # not preceded by a word/Devanagari char
    _WB_R = r'(?![^\W\s\u0900-\u097F])'    # not followed by a word/Devanagari char
    return re.compile(
        _WB_L + r'(?:' + '|'.join(escaped) + r')' + _WB_R,
        re.IGNORECASE | re.UNICODE,
    )

def set_domain_corrections(mapping: dict[str, str]) -> None:
    """
    Swap the active DOMAIN_CORRECTIONS dict (e.g. to a different agent's
    mishear-correction table) and force the compiled pattern to rebuild
    on the next apply_domain_corrections() call.

    Call this once at startup, right after the agent profile is chosen —
    before entering the main loop. Every existing call site of
    apply_domain_corrections() is unaffected; it keeps reading whatever
    DOMAIN_CORRECTIONS currently points at.
    """
    global DOMAIN_CORRECTIONS, _CORRECTION_PATTERN
    DOMAIN_CORRECTIONS = mapping
    _CORRECTION_PATTERN = None  # lazily rebuilt on next apply_domain_corrections() call


def apply_domain_corrections(text: str) -> str:
    """
    Apply DOMAIN_CORRECTIONS to `text` in a single regex pass.

    - Case-insensitive match.
    - Replacement preserves the corrected capitalisation from the dict
      (not the original casing from Whisper) — e.g. "mmco" → "MMCOE".
    - Safe to call on Devanagari text; non-ASCII correction keys still
      match because re.UNICODE is the default in Python 3.
    - < 0.2 ms for a typical 20-word transcript.
    """
    if not DOMAIN_CORRECTIONS:
        return text

    global _CORRECTION_PATTERN
    if _CORRECTION_PATTERN is None:
        _CORRECTION_PATTERN = _build_correction_pattern()

    def _replace(m: re.Match) -> str:
        return DOMAIN_CORRECTIONS[m.group(0).lower()]

    corrected = _CORRECTION_PATTERN.sub(_replace, text)
    if corrected != text:
        print(f"[STT-CORRECT] {text!r} → {corrected!r}")
    return corrected


# ── Load Whisper ──────────────────────────────────────────────────────────
# Dual-model strategy:
#   • Language detection + English  → "small"  (loaded at startup, ~0.46 GB)
#   • Hindi / Marathi transcription → "medium" (loaded lazily on first HI/MR
#     query, then kept in memory for the rest of the session, ~1.42 GB)
#
# Loading medium lazily means startup works even when other GPU processes
# (e.g. Ollama) are holding memory. The first Hindi/Marathi query will take
# an extra ~3-5s while medium loads; every subsequent one is instant.
# If you want to pre-load medium at startup (and you've closed Ollama),
# call _ensure_medium_loaded() manually after the wake-word models are up.
#
# ── Register CUDA DLL paths for Windows ─────────────────────────────────────
def _register_cuda_dll_paths():
    import sys
    import os
    if sys.platform != "win32":
        return

    candidate_paths = [
        r"C:\Users\LOQ\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib",
    ]
    venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
    candidate_paths.extend([
        os.path.join(venv_site, "nvidia", "cublas", "bin"),
        os.path.join(venv_site, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site, "nvidia", "cuda_runtime", "bin"),
        os.path.join(venv_site, "torch", "lib"),
    ])
    for ver in ["v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0"]:
        candidate_paths.append(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin")

    for path in candidate_paths:
        if os.path.exists(path):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

_register_cuda_dll_paths()

_USE_CUDA = True

def _load_whisper_model(size: str, purpose: str) -> WhisperModel:
    """Prefer CUDA, but keep the kiosk usable when the local driver/runtime mismatch."""
    global _USE_CUDA
    if _USE_CUDA:
        try:
            print(f"[STT] Loading Whisper {size} ({purpose}) on CUDA...")
            model = WhisperModel(size, device="cuda", compute_type="int8_float16")
            # Perform quick warmup encode to verify CUDA cuBLAS DLLs are present & functional
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(dummy_audio, beam_size=1)[0])
            return model
        except Exception as e:
            print(f"[STT] CUDA failing ({e}). Disabling CUDA for session & using CPU.")
            _USE_CUDA = False

    print(f"[STT] Loading Whisper {size} ({purpose}) on CPU...")
    return WhisperModel(size, device="cpu", compute_type="int8")


print("[STT] Loading Whisper small (language detection + English)...")
whisper_small = _load_whisper_model("small", "language detection + English")
print("[STT] Whisper small ready!")

HI_MR_MODEL_NAME = "medium"   # default: "medium", can be upgraded to "large-v3-turbo" or "large-v3"

whisper_medium: WhisperModel | None = None   # loaded on first HI/MR query

def set_himr_model_name(name: str) -> None:
    """Set the Whisper model size to use for Hindi/Marathi (e.g. 'medium', 'large-v3-turbo', 'large-v3')."""
    global HI_MR_MODEL_NAME, whisper_medium
    if HI_MR_MODEL_NAME != name:
        HI_MR_MODEL_NAME = name
        whisper_medium = None  # Reload model on next query

def _ensure_medium_loaded() -> WhisperModel:
    """
    Load whisper_medium if not already loaded. Returns the model.
    Falls back to whisper_small with a warning if loading fails.
    """
    global whisper_medium
    if whisper_medium is not None:
        return whisper_medium
    print(f"[STT] Loading Whisper {HI_MR_MODEL_NAME} for Hindi/Marathi...")
    try:
        whisper_medium = _load_whisper_model(HI_MR_MODEL_NAME, "Hindi/Marathi transcription")
        print(f"[STT] Whisper {HI_MR_MODEL_NAME} ready!")
    except Exception as e:
        print(f"[STT] WARNING: Could not load {HI_MR_MODEL_NAME} model ({e}).")
        print("[STT] Falling back to small for this query.")
        whisper_medium = whisper_small   # type: ignore[assignment]
    return whisper_medium

# Separate tiny model just for wake-word checking — runs continuously while
# idle, so it must be cheap. Kept distinct from whisper_small/medium so
# wake-word polling never competes with / degrades real command transcription.
print("[STT] Loading wake-word model (tiny)...")
wake_model = _load_whisper_model("tiny", "wake-word")
print("[STT] Wake-word model ready!")

# Pre-load the Hindi/Marathi medium model at startup to avoid first-query latency
try:
    _ensure_medium_loaded()
except Exception as e:
    print(f"[STT] Failed to pre-load medium model at startup: {e}")
_audio_queue = queue.Queue()
_mic_running = threading.Event()
_mic_running.set()

def _mic_callback(indata, frames, time_info, status):
    _audio_queue.put(indata.copy().flatten())

print("[STT] Pre-opening microphone (background thread)...")
_mic_stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE,
    callback=_mic_callback,
)
_mic_stream.start()
print("[STT] Microphone ready!")


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


# ── Noise reduction ───────────────────────────────────────────────────────────
# Applied to every captured utterance BEFORE Whisper sees it.
# Uses spectral gating (noisereduce) — no model, no GPU, ~5-15ms on CPU.
#
# stationary=False  → estimates noise floor per-frame (handles fan hum that
#                     varies as person walks closer, AC switching on/off, etc.)
# prop_decrease=0.8 → removes 80% of noise energy — leaves a little so
#                     de-noised speech doesn't sound hollow/ringy to Whisper
# n_std_thresh_stationary=1.5 → how aggressively to gate (higher = more
#                     aggressive; 1.5 is safe for typical office/lab noise)
#
# The first N_NOISE_FRAMES chunks (≈300ms) are used as the noise profile
# sample. This works because _capture_utterance always prepends a PRE_ROLL
# (pre-speech silence) to the buffer before the first voiced chunk, so the
# beginning of audio_flat is almost always ambient noise, not speech.

# Bug#4 fix: reduced from 0.8→0.65 to be less aggressive on Marathi phonemes.
# At 0.8 (80% noise removal) voiced Marathi fricatives/stops in short 3-4s
# clips were being suppressed as "noise" when the noise profile was estimated
# from pre-roll frames that sometimes overlap with the onset of speech.
# 0.65 removes ~2/3 of noise energy which is still effective against mic hum
# and fan noise while preserving the phoneme energy Whisper needs to decode.
_NR_PROP_DECREASE = 0.65
_NR_N_STD         = 1.5

def _denoise(audio: np.ndarray, sr: int = SAMPLE_RATE, skip_if_clean: bool = False) -> np.ndarray:
    """
    Spectral-gating noise reduction on a float32 audio array.
    Returns a float32 array of the same length.
    Safe to call even on very short clips — falls back gracefully.

    skip_if_clean: if True and the audio RMS suggests it is already
    speech-dominant (e.g. a pre-recorded WAV file with no ambient noise),
    skip denoising entirely to avoid removing real phoneme energy.
    """
    if len(audio) < sr * 0.1:          # skip if under 100ms (too short to profile)
        return audio
    # Bug#4 fix: skip denoising when audio is already clean/loud enough.
    # RMS > 0.06 on a float32 signal means strong speech signal present;
    # denoising clean audio can hallucinate artifacts that confuse Whisper.
    if skip_if_clean and _rms(audio) > 0.06:
        return audio
    try:
        denoised = _nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=_NR_PROP_DECREASE,
            n_std_thresh_stationary=_NR_N_STD,
        )
        return denoised.astype(np.float32)
    except Exception as e:
        # Never let denoising crash the STT pipeline — return original audio
        print(f"[STT] Noise reduction skipped ({e})")
        return audio


def _capture_utterance(
    max_chunks: int,
    silence_chunks_needed: int,
    queue_timeout: float = 2.0,
    prebuffered_chunks: list | None = None,
) -> np.ndarray | None:
    """
    Shared VAD capture loop. Drains stale audio, waits for speech onset,
    records until trailing silence, returns the flattened audio buffer
    (or None if nothing usable was captured).

    `prebuffered_chunks`, if given, is fed through the same VAD logic
    BEFORE pulling anything from the live queue. This is used for barge-in:
    the chunks a BargeInWatcher already consumed from the queue while
    detecting the interruption contain the user's first words — without
    replaying them here, those words would be silently lost (consumed by
    the watcher, never seen by the transcriber). When prebuffered_chunks
    is given, we do NOT drain the live queue first (there's nothing stale
    to drain — the watcher was the only consumer right up until now).
    """
    if prebuffered_chunks:
        audio_buffer   = []
        pre_roll       = []
        speech_started = False
        silence_count  = 0
        speech_chunks  = 0

        for chunk in prebuffered_chunks:
            rms = _rms(chunk)
            if len(audio_buffer) > max_chunks:
                break
            if not speech_started:
                pre_roll.append(chunk)
                if len(pre_roll) > PRE_ROLL_CHUNKS:
                    pre_roll.pop(0)
                if rms > RMS_SPEECH_THRESH:
                    speech_started = True
                    silence_count  = 0
                    audio_buffer.extend(pre_roll)
                    audio_buffer.append(chunk)
                    speech_chunks += 1
            else:
                audio_buffer.append(chunk)
                speech_chunks += 1
                if rms < RMS_SILENCE_THRESH:
                    silence_count += 1
                else:
                    silence_count = 0
                if silence_count >= silence_chunks_needed:
                    # Trailing silence already found within the prebuffered
                    # audio alone — done, no need to touch the live queue.
                    if speech_chunks * CHUNK_MS < MIN_SPEECH_MS:
                        return None
                    return np.concatenate(audio_buffer)
        # Prebuffered audio consumed without hitting a silence cutoff (the
        # interruption was still ongoing when the watcher handed off) —
        # fall through to the live queue to keep capturing the rest of it,
        # continuing the SAME speech_started/silence_count state rather
        # than resetting and risking a false "no speech" on a fresh start.
    else:
        # No prebuffer — original behavior: drain stale queued audio first.
        while not _audio_queue.empty():
            try:
                _audio_queue.get_nowait()
            except queue.Empty:
                break

        audio_buffer   = []
        pre_roll       = []
        speech_started = False
        silence_count  = 0
        speech_chunks  = 0

    while True:
        try:
            chunk = _audio_queue.get(timeout=queue_timeout)
        except queue.Empty:
            return None

        rms = _rms(chunk)

        if len(audio_buffer) > max_chunks:
            break

        if not speech_started:
            pre_roll.append(chunk)
            if len(pre_roll) > PRE_ROLL_CHUNKS:
                pre_roll.pop(0)
            if rms > RMS_SPEECH_THRESH:
                speech_started = True
                silence_count  = 0
                audio_buffer.extend(pre_roll)
                audio_buffer.append(chunk)
                speech_chunks += 1
        else:
            audio_buffer.append(chunk)
            speech_chunks += 1

            if rms < RMS_SILENCE_THRESH:
                silence_count += 1
            else:
                silence_count = 0

            if silence_count >= silence_chunks_needed:
                break

    if not speech_started or speech_chunks * CHUNK_MS < MIN_SPEECH_MS:
        return None

    return np.concatenate(audio_buffer)


def _normalize_wake_text(text: str) -> str:
    """Lowercase + strip to bare words for wake-word substring checking."""
    return re.sub(r"[^\w\s\u0900-\u097F]", " ", text.lower(), flags=re.UNICODE)


def listen_for_wake_word() -> tuple[bool, str, str]:
    """
    Lightweight continuous listener for the wake word "Adi".

    Runs a fast 'tiny' Whisper pass on each captured speech segment and
    looks for "adi" (or a known mis-transcription of it) ANYWHERE in the
    utterance — anything said before it is ignored (the tiny model often
    mangles a lead-in word like "Hey" into garbage, so we don't require
    one). Everything spoken AFTER "adi" in the same breath is treated as
    the start of the command.

    Returns
    -------
    (detected, leftover_text, leftover_lang)
      detected      : True if "adi" was heard.
      leftover_text : any speech captured AFTER "adi" in the same breath
                       (e.g. "Adi, what are the fees?" → "what are the
                       fees"). Empty string if the user just said "adi"
                       alone — caller should then call
                       listen_and_transcribe() for the actual command.
      leftover_lang : language of the leftover speech (only meaningful if
                       leftover_text is non-empty).
    """
    audio_flat = _capture_utterance(WAKE_MAX_CHUNKS, WAKE_SILENCE_CHUNKS_NEEDED)
    if audio_flat is None:
        return False, "", "en"

    segments, info = wake_model.transcribe(
        audio_flat,
        beam_size=1,
        language=None,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    if not text:
        print("[WAKE-DEBUG] (silence / empty transcript)")
        return False, "", "en"

    norm = _normalize_wake_text(text)
    words = norm.split()
    print(f"[WAKE-DEBUG] tiny model heard: \"{text}\"  →  words={words}")

    # ── Hallucination filter ────────────────────────────────────────────
    # The tiny model confabulates short filler phrases on near-silence /
    # faint room noise. If the ENTIRE utterance is a known hallucination,
    # bail out now rather than risk fuzzy-matching it into a false wake.
    if norm in HALLUCINATION_PHRASES:
        print(f"[WAKE-DEBUG] (ignored — looks like a hallucination: \"{norm}\")")
        return False, "", "en"

    # ── Exact match first ───────────────────────────────────────────────
    hit_idx = None
    for i, w in enumerate(words):
        if w in WAKE_WORDS:
            hit_idx = i
            break

    # ── Fuzzy fallback ───────────────────────────────────────────────────
    # Tiny model often mangles "adi" into short fragments ("adic", "edy")
    # that will never be in WAKE_WORDS verbatim. Only check words between
    # WAKE_FUZZY_MIN_LEN and WAKE_FUZZY_MAX_LEN chars against "adi"
    # similarity — too short (1-2 chars) and coincidental matches inside
    # unrelated speech become indistinguishable from genuine attempts;
    # too long and we'd start fuzzy-matching unrelated words.
    if hit_idx is None:
        best_ratio = 0.0
        best_i = None
        for i, w in enumerate(words):
            if not w or len(w) < WAKE_FUZZY_MIN_LEN or len(w) > WAKE_FUZZY_MAX_LEN:
                continue
            ratio = difflib.SequenceMatcher(None, w, WAKE_FUZZY_REF).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        if best_i is not None and best_ratio >= WAKE_FUZZY_THRESHOLD:
            print(f"[WAKE-DEBUG] fuzzy match: \"{words[best_i]}\" ~ \"adi\" "
                  f"(ratio={best_ratio:.2f})")
            hit_idx = best_i

    if hit_idx is None:
        return False, "", "en"

    print(f"[WAKE] Heard: \"{text}\" — wake word matched on \"{words[hit_idx]}\"")

    leftover_words = words[hit_idx + 1:]
    if len(leftover_words) >= 2:
        lang = info.language if info.language in SUPPORTED_LANGS else "hi"
        return True, " ".join(leftover_words), lang

    return True, "", "en"


def _has_repetition_loop(text: str, min_repeats: int = 3, min_coverage: float = 0.6) -> bool:
    """
    Detects Whisper's repetition-loop hallucination: the same short phrase
    repeated back-to-back many times.

    Also catches hyphen-separated syllable loops like "ahe-ahe-ahe-ahe" which
    appear as a single "word" (space-split doesn't see the repetition). These
    are split on hyphens and checked independently.
    """
    def _check_words(words: list) -> bool:
        n = len(words)
        if n < min_repeats * 2:
            return False
        max_phrase_len = n // min_repeats
        for plen in range(1, max_phrase_len + 1):
            phrase = words[0:plen]
            repeats = 1
            i = plen
            while i + plen <= n and words[i:i + plen] == phrase:
                repeats += 1
                i += plen
            if repeats >= min_repeats and (repeats * plen) / n >= min_coverage:
                return True
        return False

    # Check space-split word repetitions
    if _check_words(text.split()):
        return True

    # Also check each "word" for internal hyphen repetitions (e.g. "ahe-ahe-ahe")
    for word in text.split():
        if '-' in word:
            parts = word.split('-')
            if _check_words(parts):
                return True

    return False


def listen_and_transcribe(
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, str]:
    """
    Listen using the background mic thread.
    Returns (text, lang).
    """
    print("[*] Listening... (speak now)", flush=True)

    t_start = time.time()
    audio_flat = _capture_utterance(MAX_CHUNKS, SILENCE_CHUNKS_NEEDED)
    t_capture = time.time()
    capture_sec = t_capture - t_start

    if audio_flat is None:
        print("[!] No usable speech captured.")
        return "", "en"

    overall_rms = _rms(audio_flat)
    duration    = len(audio_flat) / sample_rate
    print(f"[MIC] RMS: {overall_rms:.4f} | Duration: {duration:.1f}s | Capture: {capture_sec:.2f}s", flush=True)

    # ── Noise reduction ───────────────────────────────────────────────────
    t_nr_start = time.time()
    audio_flat = _denoise(audio_flat, sr=sample_rate)
    print(f"[STT] Noise reduction in {(time.time()-t_nr_start)*1000:.0f}ms", flush=True)

    # ── Detect language, restricted to supported set ───────────────────────
    detected_lang = None
    try:
        # Language detection always uses small — cheap classification pass.
        _top_lang, _top_prob, all_lang_probs = whisper_small.detect_language(audio_flat)

        # Bug#1 fix: pick the HIGHEST-PROBABILITY supported language, not the
        # first one encountered in SUPPORTED_LANGS order.
        best_code, best_prob = None, 0.0
        lang_prob_map: dict[str, float] = {}
        for code, prob in all_lang_probs:
            lang_prob_map[code] = prob
            if code in SUPPORTED_LANGS and prob > best_prob:
                best_code, best_prob = code, prob

        if best_code is not None:
            mr_prob = lang_prob_map.get("mr", 0.0)
            hi_prob = lang_prob_map.get("hi", 0.0)

            # English guard: this assistant serves Marathi/Hindi speakers.
            # Raise threshold to 0.70: utterances starting with "M CAD
            # Solutions" (an English company name) push Whisper small's
            # language classifier to 0.67 en even when the rest of the
            # sentence is pure Marathi ("कोणकोणते कोर्सेस देते आहेत").
            # At 0.40 the old guard was too easy to pass — 0.70 ensures
            # en is only accepted for genuinely English utterances.
            if best_code == "en":
                if best_prob < 0.70 or (hi_prob + mr_prob) >= 0.20:
                    # English confidence too low or Devanagari langs competitive
                    # — prefer hi/mr based on which has higher probability
                    if mr_prob >= hi_prob:
                        best_code = "mr"
                    else:
                        best_code = "hi"
                    print(f"[STT] en (p={best_prob:.2f}) rejected — Devanagari more likely "
                          f"(hi={hi_prob:.2f} mr={mr_prob:.2f}) → using {best_code}")

            # Marathi bias: if hi won but mr is within 20%, prefer mr
            if best_code == "hi" and mr_prob > 0 and mr_prob >= hi_prob * 0.80:
                print(f"[STT] lang=hi (p={hi_prob:.2f}) vs mr (p={mr_prob:.2f}) — "
                      f"biasing toward Marathi (within 20%)")
                best_code = "mr"

            # Low-confidence overall fallback: if winning prob < 0.20, default mr
            final_prob = lang_prob_map.get(best_code, best_prob)
            if final_prob < 0.20:
                print(f"[STT] All lang probs very low (best={best_code} p={final_prob:.2f}) "
                      f"— defaulting to 'mr' (Marathi-first assistant)")
                best_code = "mr"

            detected_lang = best_code
            print(f"[STT] Best supported lang: {detected_lang} (p={lang_prob_map.get(detected_lang,0):.2f}) "
                  f"| top-3: {[(c,round(p,2)) for c,p in all_lang_probs[:3]]}")
        else:
            print(f"[STT] No supported language in top candidates "
                  f"({all_lang_probs[:3]}) — defaulting to 'mr'")
            detected_lang = "mr"
    except Exception as e:
        print(f"[STT] detect_language() failed/unexpected shape ({e}) — "
              f"falling back to auto-detect inside transcribe().")
        detected_lang = None

    # ── Pick transcription model based on detected language ───────────────
    # Always use the large/medium model — whisper_small cannot handle
    # code-mixed Marathi utterances ("M CAD Solutions कोणकोणते कोर्सेस") and
    # produces empty output when lang=en is mistakenly detected. Small is
    # used ONLY for the cheap language detection pass above, never for
    # transcription.
    transcribe_model = _ensure_medium_loaded()
    model_label = HI_MR_MODEL_NAME if transcribe_model is not whisper_small else "small(fallback)"
    print(f"[STT] Detected lang={detected_lang} → using {model_label} for transcription")

    # ── Transcribe ────────────────────────────────────────────────────────
    # initial_prompt seeds Whisper's beam-search context with domain vocabulary
    # so it spells MMCOE-specific terms correctly instead of phonetically
    # garbling them. Separate prompts per language — Devanagari script for
    # hi/mr; English for en. This alone fixes ~50% of keyword-miss failures.
    # Domain vocabulary for M CAD Solutions (mechanical design / Industry 4.0
    # training institute) — was previously MMCOE college-admission vocabulary
    # left over from the Adi robot project, which actively worked AGAINST
    # recognising CATIA/SolidWorks/UG NX/BIW/placement terms here.
    # Bug#5 fix: company name moved to END of prompt so it doesn't anchor
    # Whisper's beam search at the start of the decode — "M CAD Solutions"
    # at position 0 was causing the model to phonetically hallucinate it
    # at the beginning of every utterance ("Get solution,...").
    # Domain vocabulary is still present to bias spelling of technical terms.
    _INITIAL_PROMPTS = {
        "en": ("CATIA V5 SolidWorks UG NX BIW fixture design GD&T OEM "
               "Industry 4.0 ROS2 digital twin placement batch size "
               "certificate timings enroll demo course fees M CAD Solutions"),
        "hi": ("CATIA SolidWorks UG NX BIW फिक्सचर डिज़ाइन प्लेसमेंट फीस "
               "बैच साइज़ सर्टिफिकेट कोर्स समय एनरोल डेमो M CAD Solutions"),
        "mr": ("CATIA SolidWorks UG NX BIW फिक्स्चर डिझाईन प्लेसमेंट फी "
               "बॅच साईझ सर्टिफिकेट कोर्सेस वेळापत्रक एनरोल डेमो M CAD Solutions"),
    }
    _prompt = _INITIAL_PROMPTS.get(detected_lang or "en", "")

    # beam_size=1 (greedy decoding) is a major accuracy killer specifically
    # on Hindi/Marathi — these have far more acoustic ambiguity per phoneme
    # than English, and greedy search locks onto the first wrong token with
    # no way to recover (root cause of garbage transcripts like
    # "एड्दूसरे वर्षा सटिला अद्मिनेरी कागप्ट्रे पॉनताओ"). Beam search costs
    # extra latency but only on hi/mr, which already pay the medium-model
    # latency tax — a few hundred ms more for a transcript the matcher can
    # actually work with is worth it. English stays at beam_size=1.
    _beam = 5 if detected_lang in ("hi", "mr") else 1
    # Bug#3 fix: pass temperature as a FALLBACK LIST instead of temperature=0.
    # With temperature=0 (pure greedy), Whisper commits to the highest-prob
    # token at each step and has no recovery mechanism when it goes wrong —
    # this is the root cause of "ahe-ahe-ahe-ahe" repetition loops.
    # faster-whisper accepts a list: tries temperature[0] first; if
    # compression_ratio_threshold fires (detects a repeat loop), it retries
    # with temperature[1], temperature[2] etc. until a clean transcript
    # emerges or the list is exhausted. This mirrors openai-whisper's
    # original fallback mechanism that faster-whisper disabled by default.
    _temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    segments_gen, info = transcribe_model.transcribe(
        audio_flat,
        beam_size=_beam,
        best_of=_beam,
        language=detected_lang,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=_prompt,
        temperature=_temperatures,       # Bug#3: fallback list prevents repetition loops
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4, # triggers temperature fallback on repeat loops
        log_prob_threshold=-1.0,
    )
    segments = list(segments_gen)   # materialise generator before iterating twice

    # ── Segment confidence filter ─────────────────────────────────────────
    # Whisper sometimes appends a hallucinated low-probability segment
    # (e.g. "thanks for watching") at the end of a short utterance.
    # Drop any segment whose avg_logprob is below a floor threshold.
    _LOG_PROB_FLOOR = -1.0   # segments more uncertain than this are dropped
    filtered_segs = []
    for s in segments:
        if hasattr(s, 'avg_logprob') and s.avg_logprob < _LOG_PROB_FLOOR:
            print(f"[STT] Dropping low-prob segment (logprob={s.avg_logprob:.2f}): {s.text!r}")
        else:
            filtered_segs.append(s)
    text = " ".join(s.text.strip() for s in filtered_segs).strip()

    # ── Repetition-loop filter ──────────────────────────────────────────
    # Catches decode loops the per-segment filters above can't see (see
    # _has_repetition_loop docstring). Treat exactly like an empty
    # transcript — caller already handles "" as "nothing heard".
    if text and _has_repetition_loop(text):
        print(f"[STT] Discarding repetition-loop hallucination: {text!r}")
        text = ""

    if detected_lang is not None:
        lang = detected_lang
    else:
        # detect_language() failed — transcribe_model auto-detected on its own.
        # Apply the same safety remap: if it picked something outside our 3
        # supported languages, force 'hi' rather than crash downstream.
        lang = info.language
        if lang not in SUPPORTED_LANGS:
            print(f"[STT] Remapped '{lang}' → 'hi'")
            lang = "hi"

    # ── Domain correction pass (~0 ms) ────────────────────────────────────
    # Fix known Whisper (small) mishearings of MMCOE-specific terms before
    # the text reaches KeywordMatcher. See DOMAIN_CORRECTIONS dict above.
    # To disable (e.g. after switching to "medium"): comment the line out.
    text = apply_domain_corrections(text)

    # ── Transcript-based Marathi override ─────────────────────────────────
    # Whisper is biased toward 'hi' because its training set has ~10x more
    # Hindi than Marathi audio. Even with the probability-based bias above,
    # short ambiguous clips often land on 'hi'. The transcript itself is a
    # stronger signal: if any Marathi-exclusive lexical marker appears in the
    # decoded text (e.g. 'आहे', 'कोण', 'सांगा', 'कुठे'), the user is
    # definitely speaking Marathi regardless of what the mel-spectrogram
    # language classifier said.
    if lang != "mr" and text and _is_marathi_by_transcript(text):
        print(f"[STT] Marathi lexical marker found in transcript — overriding lang: {lang} → mr")
        lang = "mr"

    t_end = time.time()
    print(f"[TIMING] capture: {capture_sec:.2f}s | whisper: {t_end - t_capture:.2f}s | total STT: {t_end - t_start:.2f}s")
    print(f"[STT] [{lang}] {text}")

    return text, lang


def calibrate_noise_floor(seconds: float = 3.0) -> float:
    """
    Measure ambient RMS over `seconds` of silence.
    Prints a suggested RMS_SPEECH_THRESH value.
    Run with:  python stt_engine.py --calibrate
    """
    print(f"[CAL] Measuring noise floor for {seconds}s — stay quiet...")
    time.sleep(1.5)   # wait for mic background thread to fill the queue
    # Drain queue first
    while not _audio_queue.empty():
        try: _audio_queue.get_nowait()
        except queue.Empty: break

    chunks = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            chunk = _audio_queue.get(timeout=1.0)
            chunks.append(_rms(chunk))
        except queue.Empty:
            pass

    if not chunks:
        print("[CAL] No audio received.")
        return 0.0

    avg = float(np.mean(chunks))
    peak = float(np.max(chunks))
    suggested = round(peak * 2.5, 4)
    print(f"[CAL] Noise floor — avg RMS: {avg:.4f}  peak RMS: {peak:.4f}")
    print(f"[CAL] Suggested RMS_SPEECH_THRESH = {suggested}  (set in stt_engine.py line ~21)")
    return suggested


def close_mic():
    _mic_stream.stop()
    _mic_stream.close()
    print("[STT] Microphone closed.")


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--calibrate" in sys.argv:
        calibrate_noise_floor(3.0)
        close_mic()
        sys.exit(0)

    if "--test-wake" in sys.argv:
        try:
            print("\nSay 'Adi' (alone, or 'Adi <question>'). Ctrl+C to stop.\n")
            while True:
                print("[*] Waiting for wake word...", flush=True)
                detected, leftover, lang = listen_for_wake_word()
                if detected:
                    if leftover:
                        print(f">>> WAKE + command in one breath [{lang}]: {leftover}\n")
                    else:
                        print(">>> WAKE detected, no leftover — would now call listen_and_transcribe()\n")
                else:
                    print("(no wake word in that utterance)\n")
        except KeyboardInterrupt:
            close_mic()
            print("Stopped.")
        sys.exit(0)

    try:
        print("\nSpeak in Hindi, Marathi, or English. Ctrl+C to stop.\n")
        while True:
            text, lang = listen_and_transcribe()
            if text:
                print(f"\n>>> [{lang}] {text}\n")
            else:
                print("(nothing detected)\n")
    except KeyboardInterrupt:
        close_mic()
        print("Stopped.")