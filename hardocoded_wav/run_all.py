"""
run_all.py - runs the full pipeline in order.

1. Build manifests (unique answers only, per language)
2. Generate hi + mr wavs with IndicF5 (voice cloned from full_name_mr.wav)
3. Generate en wavs with Kokoro

Safe to re-run any time: every step skips files that already exist.
"""

import subprocess
import sys


def run(cmd):
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run([sys.executable, "generate_manifest.py"])
    run([sys.executable, "tts_indicf5.py", "--lang", "all"])
    run([sys.executable, "tts_kokoro_en.py"])
    print("\nDone. Check output/en, output/hi, output/mr")