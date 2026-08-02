"""
fix_indicf5_model.py

Patches the HuggingFace-cached IndicF5 model.py so its call to f5_tts's
load_model() uses the correct keyword argument name.

Root cause:
    Installed f5-tts package defines:
        load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=..., vocab_file=...)
    But IndicF5's remote model.py (trust_remote_code=True) calls it with:
        load_model(..., ckpt_path=ckpt_path, ...)
    -> TypeError: load_model() got an unexpected keyword argument 'ckpt_path'

Fix: rename that one keyword from `ckpt_path=` to `ckpt_file=` in the cached
model.py (only inside the load_model(...) call, so no other ckpt_path usages
in the file are touched).

Usage:
    python fix_indicf5_model.py
"""

import glob
import os
import re
import sys

# Locate the cached model.py automatically (snapshot hash can change on updates)
PATTERN = os.path.expanduser(
    r"~/.cache/huggingface/modules/transformers_modules/ai4bharat/IndicF5/*/model.py"
)


def find_model_py():
    matches = glob.glob(PATTERN)
    if not matches:
        # Fallback for Windows-style home if expanduser didn't resolve as expected
        alt = os.path.join(
            os.environ.get("USERPROFILE", ""),
            ".cache", "huggingface", "modules", "transformers_modules",
            "ai4bharat", "IndicF5", "*", "model.py",
        )
        matches = glob.glob(alt)
    if not matches:
        print("[error] Could not find cached model.py. Searched:")
        print(f"        {PATTERN}")
        sys.exit(1)
    if len(matches) > 1:
        print(f"[warn] Multiple snapshots found, using the most recently modified one:")
        for m in matches:
            print(f"        {m}")
        matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def patch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "ckpt_file=" in content and "ckpt_path=" not in content:
        print(f"[skip] {path} already patched (no ckpt_path= found).")
        return

    # Only replace ckpt_path= as a keyword argument (ckpt_path=<something>),
    # not the variable name ckpt_path itself elsewhere in the file.
    new_content, count = re.subn(r"\bckpt_path=", "ckpt_file=", content)

    if count == 0:
        print(f"[warn] No 'ckpt_path=' keyword found in {path}. Nothing changed.")
        print("       The file may already differ from the expected version — check manually.")
        return

    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[backup] saved original to {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[ok] patched {count} occurrence(s) of 'ckpt_path=' -> 'ckpt_file=' in {path}")


if __name__ == "__main__":
    model_py = find_model_py()
    print(f"[found] {model_py}")
    patch_file(model_py)
    print("\nDone. Now re-run: python run_all.py")
