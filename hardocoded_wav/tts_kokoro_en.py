"""
tts_kokoro_en.py

Generates English wavs using Kokoro-82M.

IMPORTANT LIMITATION: Kokoro does NOT support zero-shot voice cloning from
an arbitrary reference wav. It only ships fixed preset voices. So this will
NOT sound identical to full_name_mr.wav — it'll be the closest preset voice
(default: af_heart, a female voice, since your reference sample is female).

If you need the EXACT same cloned voice in English, use IndicF5 itself for
English too (it supports English text) with the same ref_audio - swap
VOICE_ENGINE below to "indicf5_en". That keeps one consistent voice across
hi/mr/en instead of mixing two different models.

Requirements (run on your GPU/CPU PC, not here):
    pip install kokoro soundfile

Usage:
    python tts_kokoro_en.py
"""

import csv
import os

import numpy as np
import soundfile as sf

VOICE = "af_heart"  # closest default female preset; change to taste
LANG_CODE = "a"  # 'a' = American English


def load_manifest():
    with open("manifests/manifest_en.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=LANG_CODE)

    rows = load_manifest()
    out_dir = os.path.join("output", "en")
    os.makedirs(out_dir, exist_ok=True)

    for row in rows:
        out_path = os.path.join(out_dir, row["audio_file"])
        if os.path.exists(out_path):
            print(f"[skip-exists] {out_path}")
            continue

        text = row["text"]
        print(f"[gen] {out_path}")
        generator = pipeline(text, voice=VOICE)
        chunks = [audio for _, _, audio in generator]
        if not chunks:
            print(f"[warn] no audio generated for {out_path}, skipping")
            continue
        # Kokoro splits long text into multiple chunks (sentence boundaries).
        # Concatenate all of them so longer FAQ answers aren't cut off after
        # just the first sentence.
        full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        sf.write(out_path, full_audio, 24000)


if __name__ == "__main__":
    main()