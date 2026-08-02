"""
generate_manifest.py

Reads mcad_solution_faq.csv (question+answer+audio_file per row) and builds
ONE manifest CSV per language containing ONLY unique answers that need a wav.

- Dedup key = (lang, answer_text). If same answer text repeats for a lang,
  only the FIRST audio_file name is kept and reused (no duplicate wav made).
- Output: manifests/manifest_en.csv, manifest_hi.csv, manifest_mr.csv
  columns -> audio_file, lang, text
"""

import csv
import os

SRC_CSV = "mcad_solution_faq.csv"
OUT_DIR = "manifests"

os.makedirs(OUT_DIR, exist_ok=True)


def build_manifests():
    seen = {}  # (lang, answer) -> audio_file already assigned
    rows_by_lang = {"en": [], "hi": [], "mr": []}

    with open(SRC_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang"].strip()
            answer = row["answer"].strip()
            audio_file = row["audio_file"].strip()

            key = (lang, answer)
            if key in seen:
                # duplicate answer text -> skip, don't add a new wav entry
                print(f"[skip-dup] {audio_file} duplicates {seen[key]}")
                continue

            seen[key] = audio_file
            rows_by_lang.setdefault(lang, []).append(
                {"audio_file": audio_file, "lang": lang, "text": answer}
            )

    for lang, rows in rows_by_lang.items():
        if not rows:
            continue
        out_path = os.path.join(OUT_DIR, f"manifest_{lang}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_file", "lang", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[ok] wrote {out_path} ({len(rows)} unique answers)")


if __name__ == "__main__":
    build_manifests()