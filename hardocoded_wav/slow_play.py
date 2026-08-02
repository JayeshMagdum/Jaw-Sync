"""
slow_play.py

Plays WAV files from output/<lang>/ at a slower speed WITHOUT re-generating them.
Uses librosa's time-stretching (phase-vocoder) so pitch stays natural.

Usage:
    # Play ALL files for a language at 0.75x speed (slower):
    python slow_play.py --lang hi --rate 0.75

    # Play a single file:
    python slow_play.py --file output/hi/greeting_hi.wav --rate 0.70

    # Play all languages:
    python slow_play.py --lang all --rate 0.75

Speed / rate guide:
    1.0  = original speed (unchanged)
    0.80 = 20% slower  (good starting point)
    0.75 = 25% slower  (recommended for Hindi TTS)
    0.65 = 35% slower  (very slow, each syllable clear)

Requirements:
    pip install librosa sounddevice soundfile numpy
    (librosa is already used by f5-tts so it should be present)
"""

import argparse
import os
import sys
import glob

import numpy as np
import soundfile as sf


def _try_import_playback():
    """Return a play(audio, sr) function using the best available backend."""
    try:
        import sounddevice as sd
        def play(audio, sr):
            sd.play(audio, samplerate=sr)
            sd.wait()
        return play
    except ImportError:
        pass
    try:
        import pygame.mixer as mx
        import io
        import soundfile as sf2
        def play(audio, sr):
            mx.init(frequency=sr, size=-16, channels=1 if audio.ndim == 1 else 2)
            buf = io.BytesIO()
            sf2.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            sound = mx.Sound(buf)
            sound.play()
            import time
            time.sleep(sound.get_length() + 0.1)
            mx.quit()
        return play
    except ImportError:
        pass
    # Last resort: write to temp file and open with OS player
    import tempfile, subprocess, platform
    def play(audio, sr):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, sr)
        tmp.close()
        if platform.system() == "Windows":
            os.startfile(tmp.name)
        elif platform.system() == "Darwin":
            subprocess.run(["afplay", tmp.name])
        else:
            subprocess.run(["aplay", tmp.name])
    return play


def slow_down(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Time-stretch audio by rate (< 1.0 = slower, > 1.0 = faster).
    Pitch is preserved via librosa phase-vocoder.
    Falls back to resample trick if librosa unavailable.
    """
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
        print("[warn] librosa not found - using resample trick (pitch will shift slightly)")
        if audio.ndim == 2:
            audio = audio[:, 0]
        indices = np.round(np.arange(0, len(audio), rate)).astype(int)
        indices = indices[indices < len(audio)]
        return audio[indices]


def play_file(path: str, rate: float, play_fn):
    print(f"[play] {path}  (rate={rate}x)")
    audio, sr = sf.read(path, dtype="float32")
    audio_slow = slow_down(audio, rate)
    play_fn(audio_slow, sr)


def collect_files(lang: str) -> list:
    if lang == "all":
        langs = ["hi", "mr"]
    else:
        langs = [lang]
    files = []
    for l in langs:
        d = os.path.join("output", l)
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
        files.extend(wavs)
    return files


def main():
    parser = argparse.ArgumentParser(description="Play WAV files at reduced speed.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", choices=["hi", "mr", "all"],
                       help="Play all WAVs for a language folder")
    group.add_argument("--file", metavar="PATH",
                       help="Play a single WAV file")

    parser.add_argument("--rate", type=float, default=0.75,
                        help="Playback speed ratio: 1.0=normal, 0.75=25%% slower (default: 0.75)")
    args = parser.parse_args()

    if args.rate <= 0 or args.rate > 2.0:
        print("ERROR: --rate must be between 0.1 and 2.0")
        sys.exit(1)

    play_fn = _try_import_playback()

    if args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: file not found: {args.file}")
            sys.exit(1)
        play_file(args.file, args.rate, play_fn)
    else:
        files = collect_files(args.lang)
        if not files:
            print(f"No WAV files found in output/{args.lang}/")
            sys.exit(1)
        for f in files:
            play_file(f, args.rate, play_fn)


if __name__ == "__main__":
    main()
