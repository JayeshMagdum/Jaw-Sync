"""
test_stt_marathi_audio.py — Benchmark and test STT (faster-whisper) on Marathi audio files.

Reads audio files from faqmcad_wav/mr/ or a specified WAV file/folder,
runs language detection + transcription (using stt_engine settings),
applies domain corrections, and runs MCADKeywordMatcher.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from mcad_keyword_matcher import MCADKeywordMatcher
import stt_engine

BASE_DIR = Path(__file__).parent
DEFAULT_AUDIO_DIR = BASE_DIR / "faqmcad_wav" / "mr"
CSV_PATH = BASE_DIR / "mcad_solution_faq.csv"


def resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    # Simple linear interpolation resampling for test audio
    num_output_samples = int(len(audio) * target_sr / orig_sr)
    resampled = np.interp(
        np.linspace(0, len(audio), num_output_samples, endpoint=False),
        np.arange(len(audio)),
        audio
    )
    return resampled.astype(np.float32)


def transcribe_wav_file(file_path: Path, matcher: MCADKeywordMatcher | None = None, model_size: str = "medium"):
    data, sr = sf.read(str(file_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # convert stereo to mono

    audio_16k = resample_if_needed(data, sr, 16000)

    # Apply noise reduction (same as stt_engine)
    denoised_audio = stt_engine._denoise(audio_16k, sr=16000)

    # Transcribe using stt_engine's medium/small model
    if model_size == "medium":
        model = stt_engine._ensure_medium_loaded()
    else:
        model = stt_engine.whisper_small

    prompt = ("M CAD Solutions CATIA SolidWorks UG NX BIW फिक्स्चर डिझाईन "
              "प्लेसमेंट फी बॅच साईझ सर्टिफिकेट कोर्सेस वेळापत्रक एनरोल डेमो")

    t0 = time.perf_counter()
    try:
        segments_gen, info = model.transcribe(
            denoised_audio,
            beam_size=5,
            best_of=5,
            language="mr",
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            temperature=0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
        segments = list(segments_gen)
    except RuntimeError as e:
        if "cublas64_12.dll" in str(e) or "CUDA" in str(e):
            print(f"[STT] CUDA runtime error ({e}). Loading model on CPU...")
            cpu_model = stt_engine.WhisperModel(model_size, device="cpu", compute_type="int8")
            segments_gen, info = cpu_model.transcribe(
                denoised_audio,
                beam_size=5,
                best_of=5,
                language="mr",
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                temperature=0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )
            segments = list(segments_gen)
        else:
            raise e
    elapsed = time.perf_counter() - t0

    raw_text = " ".join(s.text.strip() for s in segments if getattr(s, 'avg_logprob', 0) >= -1.0).strip()
    corrected_text = stt_engine.apply_domain_corrections(raw_text)

    match_info = ""
    if matcher and corrected_text:
        res = matcher.match(corrected_text, "mr")
        match_info = f" -> Matched: [{res.topic}] conf={res.confidence:.2f}"

    return {
        "file": file_path.name,
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "lang_prob": getattr(info, "language_probability", 1.0),
        "elapsed": elapsed,
        "match_info": match_info
    }


def main():
    parser = argparse.ArgumentParser(description="Test Marathi STT on WAV files")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_AUDIO_DIR), help="Directory containing Marathi wav files")
    parser.add_argument("--file", type=str, default=None, help="Single wav file to transcribe")
    parser.add_argument("--model", type=str, choices=["small", "medium"], default="medium", help="Whisper model size to use")
    args = parser.parse_args()

    matcher = MCADKeywordMatcher(CSV_PATH)

    if args.file:
        files = [Path(args.file)]
    else:
        audio_dir = Path(args.dir)
        if not audio_dir.exists():
            print(f"Directory not found: {audio_dir}")
            sys.exit(1)
        files = sorted(list(audio_dir.glob("*.wav")))

    print(f"\n==================================================================")
    print(f"   MARATHI STT TRANSCRIPTION BENCHMARK ({len(files)} files, model={args.model})")
    print(f"==================================================================\n")

    results = []
    for idx, fpath in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Transcribing: {fpath.name}...")
        res = transcribe_wav_file(fpath, matcher=matcher, model_size=args.model)
        results.append(res)
        print(f"  Time     : {res['elapsed']:.2f}s")
        print(f"  Raw STT  : {res['raw_text']}")
        if res['corrected_text'] != res['raw_text']:
            print(f"  Corrected: {res['corrected_text']}")
        if res['match_info']:
            print(f"  Matcher  : {res['match_info']}")
        print("-" * 66)

    avg_time = sum(r['elapsed'] for r in results) / len(results) if results else 0
    print(f"\nSummary: Transcribed {len(results)} Marathi audio files. Average time: {avg_time:.2f}s per clip.")


if __name__ == "__main__":
    main()
