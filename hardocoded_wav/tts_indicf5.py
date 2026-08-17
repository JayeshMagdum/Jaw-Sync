"""
tts_indicf5.py

Generates Hindi + Marathi wavs using AI4Bharat IndicF5 (voice cloning),
cloning the voice from reference_voice/full_name_mr.wav.

WHY THIS DOESN'T USE `AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)`:
The HF-cached wrapper (model.py) for this model monkeypatches
`load_checkpoint()` into a no-op and never calls `load_state_dict(...)`
(that line is commented out in the cached file). So AutoModel loads fine
but produces an essentially untrained/random model -> garbage/noise audio,
no crash. Instead, this script builds the DiT architecture and loads the
REAL weights directly, using the same low-level f5_tts functions IndicF5's
own forward() uses internally (load_model, load_checkpoint, load_vocoder,
preprocess_ref_audio_text, infer_process).

Prerequisite: you must have already run fix_indicf5_ckpt.py to strip the
'_orig_mod.'/'ema_model.' prefixes from the raw model.safetensors into
model_fixed.safetensors (this was done previously for the Adi Robot
project on this same machine/snapshot).

Requirements (run on your GPU PC, not here):
    pip install transformers torch torchaudio soundfile

Usage:
    python tts_indicf5.py --lang mr
    python tts_indicf5.py --lang hi
    python tts_indicf5.py --lang all

Idempotent: skips any audio_file that already exists in output/<lang>/
so re-running never creates duplicate files.
"""

import argparse
import csv
import glob
import os

import soundfile as sf
import torch

REF_AUDIO = "reference_voice/full_name_mr.wav"
REF_AUDIO_FILENAME = os.path.basename(REF_AUDIO)  # "full_name_mr.wav"

# --- Locate the IndicF5 snapshot + fixed checkpoint automatically ----------
# (snapshot hash can change if the HF cache is ever cleared/re-downloaded)
_SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*"
)


def _find_snapshot_dir():
    matches = glob.glob(_SNAPSHOT_GLOB)
    if not matches:
        alt = os.path.join(
            os.environ.get("USERPROFILE", ""),
            ".cache", "huggingface", "hub",
            "models--ai4bharat--IndicF5", "snapshots", "*",
        )
        matches = glob.glob(alt)
    if not matches:
        raise RuntimeError(
            "Could not find the IndicF5 snapshot dir under the HF cache.\n"
            "Set SNAPSHOT_DIR manually near the top of tts_indicf5.py, e.g.:\n"
            r'  SNAPSHOT_DIR = r"C:\Users\Suraj\.cache\huggingface\hub\models--ai4bharat--IndicF5\snapshots\<hash>"'
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


SNAPSHOT_DIR = _find_snapshot_dir()
_ckpt_root = os.path.join(SNAPSHOT_DIR, "model_fixed.safetensors")
_ckpt_sub = os.path.join(SNAPSHOT_DIR, "checkpoints", "model_fixed.safetensors")
CKPT_FILE = _ckpt_root if os.path.exists(_ckpt_root) else _ckpt_sub
VOCAB_FILE = os.path.join(SNAPSHOT_DIR, "checkpoints", "vocab.txt")

# Same DiT config the IndicF5 wrapper uses (from its cached model.py)
DIT_CONFIG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def load_manifest(lang):
    path = f"manifests/manifest_{lang}.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_ref_text(lang):
    """
    The reference wav (full_name_mr.wav) corresponds to the 'mr' answer whose
    audio_file matches REF_AUDIO_FILENAME. Pull its exact text from the mr
    manifest so IndicF5 gets the correct ref_text for cloning.
    """
    rows = load_manifest("mr")
    for row in rows:
        if row["audio_file"] == REF_AUDIO_FILENAME:
            return row["text"]
    raise RuntimeError(
        f"Could not find ref_text: no row in manifest_mr.csv has audio_file={REF_AUDIO_FILENAME}"
    )


def _find_vocos_dir():
    glob_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/*")
    matches = glob.glob(glob_pattern)
    if not matches:
        alt = os.path.join(
            os.environ.get("USERPROFILE", ""),
            ".cache", "huggingface", "hub",
            "models--charactr--vocos-mel-24khz", "snapshots", "*",
        )
        matches = glob.glob(alt)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def get_model():
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    if not os.path.exists(CKPT_FILE):
        raise FileNotFoundError(
            f"Fixed checkpoint not found at:\n  {CKPT_FILE}\n"
            "Run fix_indicf5_ckpt.py first to strip '_orig_mod.'/'ema_model.' "
            "prefixes from the raw model.safetensors into model_fixed.safetensors."
        )
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"vocab.txt not found at:\n  {VOCAB_FILE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[snapshot] {SNAPSHOT_DIR}")
    print(f"[ckpt] {CKPT_FILE}")

    # load_model() now takes ckpt_path and handles checkpoint loading internally
    model = load_model(
        DiT,
        DIT_CONFIG,
        CKPT_FILE,
        mel_spec_type="vocos",
        vocab_file=VOCAB_FILE,
        use_ema=True,
        device=device,
    )

    vocos_dir = _find_vocos_dir()
    if vocos_dir and os.path.exists(os.path.join(vocos_dir, "config.yaml")):
        print(f"[vocos] loading from local cache: {vocos_dir}")
        vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=vocos_dir, device=device)
    else:
        vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    return model, vocoder, device


def synth_lang(model, vocoder, device, lang, ref_text, speed=1.0, overwrite=False,
               out_base_dir="output", only_file=None):
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

    rows = load_manifest(lang)
    out_dir = os.path.join(out_base_dir, lang)
    os.makedirs(out_dir, exist_ok=True)

    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(REF_AUDIO, ref_text)

    for row in rows:
        # --file filter: skip rows that don't match the requested filename
        if only_file and row["audio_file"] != only_file:
            continue

        out_path = os.path.join(out_dir, row["audio_file"])
        if os.path.exists(out_path) and not overwrite:
            print(f"[skip-exists] {out_path}")
            continue

        text = row["text"]
        print(f"[gen] {out_path} (speed={speed})")
        audio, sample_rate, _ = infer_process(
            ref_audio_processed,
            ref_text_processed,
            text,
            model,
            vocoder,
            mel_spec_type="vocos",
            speed=speed,
            nfe_step=64,            # 64 diffusion steps (vs default 32) for cleaner audio
            cross_fade_duration=0.25,  # longer crossfade to smooth chunk seams (vs 0.15)
            cfg_strength=2.0,
            device=device,
        )
        sf.write(out_path, audio, samplerate=sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["hi", "mr", "en", "all"], default="all")
    parser.add_argument("--speed", type=float, default=0.65,
                        help="Default speed for all languages unless specific speed arg is given (default: 0.65)")
    parser.add_argument("--speed-hi", type=float, default=None,
                        help="TTS generation speed for Hindi")
    parser.add_argument("--speed-mr", type=float, default=None,
                        help="TTS generation speed for Marathi")
    parser.add_argument("--speed-en", type=float, default=None,
                        help="TTS generation speed for English")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--out-dir", default="output_new",
                        help="Base output directory (default: output_new).")
    parser.add_argument("--file", default=None, metavar="AUDIO_FILE",
                        help="Only generate this single audio_file name. Requires --lang to be hi/mr/en.")
    args = parser.parse_args()

    if args.file and args.lang == "all":
        parser.error("--file requires --lang hi, mr, or en (not 'all')")

    ref_text = find_ref_text("mr")
    model, vocoder, device = get_model()

    langs = ["hi", "mr", "en"] if args.lang == "all" else [args.lang]
    for lang in langs:
        if lang == "hi":
            speed = args.speed_hi if args.speed_hi is not None else args.speed
        elif lang == "mr":
            speed = args.speed_mr if args.speed_mr is not None else args.speed
        elif lang == "en":
            speed = args.speed_en if args.speed_en is not None else args.speed
        else:
            speed = args.speed

        synth_lang(
            model, vocoder, device, lang, ref_text,
            speed=speed,
            overwrite=args.overwrite,
            out_base_dir=args.out_dir,
            only_file=args.file,
        )

    # copy the reference wav itself into <out_dir>/mr so full_name_mr.wav exists
    if not args.file:
        dst = os.path.join(args.out_dir, "mr", REF_AUDIO_FILENAME)
        if not os.path.exists(dst):
            import shutil
            shutil.copy(REF_AUDIO, dst)
            print(f"[copy] {REF_AUDIO} -> {dst}")


if __name__ == "__main__":
    main()