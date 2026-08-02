"""
slow_export.py

Reads existing WAVs from output/<lang>/ and writes time-stretched (slower)
copies into output_slow/<lang>/ WITHOUT re-running TTS.

Uses librosa phase-vocoder: pitch stays natural.

Usage:
    python slow_export.py --lang hi --rate 0.75
    python slow_export.py --lang all --rate 0.75
    python slow_export.py --lang mr --rate 0.70

Rate guide:
    1.0  = original speed
    0.80 = 20% slower
    0.75 = 25% slower  (recommended)
    0.65 = 35% slower
"""

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf


def slow_down(audio: np.ndarray, rate: float) -> np.ndarray:
    if rate == 1.0:
        return audio
    try:
        import librosa
        mono = audio[:, 0] if audio.ndim == 2 else audio
        stretched = librosa.effects.time_stretch(mono.astype(np.float32), rate=rate)
        if audio.ndim == 2:
            stretched = np.stack([stretched, stretched], axis=1)
        return stretched
    except ImportError:
        print("[warn] librosa not found - using resample trick (pitch will shift)")
        if audio.ndim == 2:
            audio = audio[:, 0]
        indices = np.round(np.arange(0, len(audio), rate)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]


def export_lang(lang: str, rate: float, overwrite: bool):
    src_dir = os.path.join("output", lang)
    dst_dir = os.path.join("output_slow", lang)
    os.makedirs(dst_dir, exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    if not wavs:
        print(f"[warn] No WAVs in {src_dir}")
        return

    for src in wavs:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            print(f"[skip] {dst}")
            continue
        audio, sr = sf.read(src, dtype="float32")
        audio_slow = slow_down(audio, rate)
        sf.write(dst, audio_slow, sr)
        print(f"[saved] {dst}  (rate={rate}x)")


def main():
    parser = argparse.ArgumentParser(description="Export slow WAV copies.")
    parser.add_argument("--lang", choices=["hi", "mr", "all"], default="all")
    parser.add_argument("--rate", type=float, default=0.75,
                        help="Speed ratio: 0.75 = 25%% slower (default)")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    try:
        import librosa
    except ImportError:
        print("[warn] librosa not installed. Run: pip install librosa")
        print("       Falling back to resample trick (pitch shift).")

    langs = ["hi", "mr"] if args.lang == "all" else [args.lang]
    for lang in langs:
        export_lang(lang, args.rate, args.overwrite)

    print(f"\nDone! Slowed files are in: output_slow/")


if __name__ == "__main__":
    main()
