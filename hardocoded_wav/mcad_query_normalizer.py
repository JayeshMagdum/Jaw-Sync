"""
mcad_query_normalizer.py  —  Pre-matching query normalization for the
M CAD Solutions FAQ bot (mcad_solution_faq.csv).

Sits between STT output and mcad_keyword_matcher:

    raw STT text
        -> normalize_query(text, lang)
            -> cleaned, vocab-bridged text
                -> MCADKeywordMatcher.match()

What this file does (in order):
  1. Unicode normalization (NFC) + lowercase
  2. Strip punctuation noise (keeps Devanagari matras, digits, spaces)
  3. "M CAD Solutions" name alias collapse (m cad / mcad / em kaad / ... -> mcad)
  4. Vocab bridge — common alternate spellings / cross-script forms for
     the terms that actually appear in mcad_solution_faq.csv (CATIA,
     SolidWorks, UG NX, BIW, fees, placement, timings, etc.)

Design notes
------------
* No fabricated STT-mishearing tables here — unlike the MMCOE admission
  normalizer (which encodes months of live transcript evidence), this
  is a fresh domain with no logged garbles yet. Keep this file's vocab
  bridge limited to plainly predictable alternate spellings /
  cross-script synonyms. Add real mishearing fixes here once you have
  logged evidence from actual STT runs (same pattern as query_normalizer.py).
* Zero external deps — stdlib + re only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# 1. "M CAD Solutions" name alias collapse
# ─────────────────────────────────────────────────────────────────────────────

_MCAD_ALIASES: list[tuple[re.Pattern, str]] = [
    # Latin-script variants / spaced-out letters
    (re.compile(r"\b(em\s*cad|m\s*-\s*cad|mkaad|m\s*kaad|mcaad)\b", re.IGNORECASE), "mcad"),
    (re.compile(r"\bm\s+c\s+a\s+d\b", re.IGNORECASE), "mcad"),
    (re.compile(r"\bmcad\s+solution(s)?\b", re.IGNORECASE), "mcad solutions"),
    # Devanagari transliterations of "M CAD" / "M CAD Solutions"
    (re.compile(r"एम\s*कैड|एम\s*कॅड|एम\s*सीएडी|एमकॅड|एमकैड"), "mcad"),
    (re.compile(r"mcad\s*सोल्यूशन्स|mcad\s*सोल्युशन्स|एमकॅड\s*सोल्यूशन्स|एमकैड\s*सोल्युशन्स"), "mcad solutions"),
]


def _collapse_mcad_aliases(text: str) -> str:
    for pat, repl in _MCAD_ALIASES:
        text = pat.sub(repl, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vocab bridge (REPLACEMENT, not additive) — plain, predictable
#    alternate spellings for the software/domain names used in the CSV.
#    Format: (regex_pattern_string, replacement_string)
# ─────────────────────────────────────────────────────────────────────────────

_RAW_VOCAB_BRIDGE: list[tuple[str, str]] = [
    # ── CATIA V5 ──────────────────────────────────────────────────────────
    (r"\bcatia\s*v\s*5\b", "catia v5"),
    (r"\bcaatia\b", "catia"),
    (r"\bkatia\b", "catia"),
    (r"\bकॅटिया\b", "catia"),
    (r"\bकटिया\b", "catia"),

    # ── SolidWorks ────────────────────────────────────────────────────────
    (r"\bsolid\s*works\b", "solidworks"),
    (r"\bsolidwork\b", "solidworks"),
    (r"\bसॉलिड\s*वर्क्स\b", "solidworks"),
    (r"\bसॉलिडवर्क्स\b", "solidworks"),
    (r"\bसोलिडवर्क्स\b", "solidworks"),

    # ── UG NX ─────────────────────────────────────────────────────────────
    (r"\bug\s*nx\b", "ug nx"),
    (r"\bunigraphics\b", "ug nx"),
    (r"\bयुजी\s*एनएक्स\b", "ug nx"),
    (r"\bयुजीएनएक्स\b", "ug nx"),
    (r"\bयूजी\s*एनएक्स\b", "ug nx"),

    # ── BIW Fixture Design ────────────────────────────────────────────────
    (r"\bb\s*i\s*w\b", "biw"),
    (r"\bbody\s*in\s*white\b", "biw"),
    (r"\bबीआयडब्ल्यू\b", "biw"),
    (r"\bबी\s*आय\s*डब्ल्यू\b", "biw"),

    # ── GD&T / OEM ────────────────────────────────────────────────────────
    (r"\bg\s*d\s*&?\s*t\b", "gd&t"),
    (r"\bo\s*e\s*m\b", "oem"),

    # ── ROS2 / Digital Twin / Industry 4.0 ───────────────────────────────
    (r"\br\s*o\s*s\s*2\b", "ros2"),
    (r"\bआरओएस\s*२\b", "ros2"),
    (r"\bआरओएस२\b", "ros2"),
    (r"\bindustry\s*4\.?0\b", "industry 4.0"),
    (r"\bindustry\s*5\.?0\b", "industry 5.0"),
    (r"\bdigital\s*twin\b", "digital twin"),
    (r"\bडिजिटल\s*ट्विन\b", "digital twin"),
    (r"\bरोबोटिक्स\b", "robotics"),

    # ── Fees / शुल्क / फीस ────────────────────────────────────────────────
    (r"\bfee\b", "fees"),
    (r"\bफी\b", "फीस"),
    (r"\bशुल्क\b", "फीस"),

    # ── Placement / प्लेसमेंट ────────────────────────────────────────────
    (r"\bplacements\b", "placement"),
    (r"\bप्लेसमेंट्स\b", "प्लेसमेंट"),
    (r"\bनौकरी\b", "प्लेसमेंट"),

    # ── Courses / कोर्स ──────────────────────────────────────────────────
    (r"\bcourses\b", "course"),
    (r"\bकोर्सेस\b", "कोर्स"),
    (r"\bकोर्सेज\b", "कोर्स"),

    # ── Batch size ────────────────────────────────────────────────────────
    (r"\bbatch\s*size\b", "batch size"),
    (r"\bबॅच\s*साईझ\b", "बॅच साईझ"),
    (r"\bबैच\s*साइज़?\b", "बैच साइज़"),

    # ── Certificate / प्रमाणपत्र ─────────────────────────────────────────
    (r"\bcertification\b", "certificate"),
    (r"\bसर्टिफिकेशन\b", "सर्टिफिकेट"),
    (r"\bप्रमाणपत्र\b", "सर्टिफिकेट"),

    # ── Timings / working hours / वेळ ────────────────────────────────────
    (r"\bworking\s*hours\b", "timings"),
    (r"\bopening\s*hours\b", "timings"),
    (r"\bटाइमिंग\b", "timings"),
    (r"\bटाइमिंग्स\b", "timings"),
    (r"\bटाईमिंग\b", "timings"),
    (r"\bऑफिस\s*टाइम\b", "कार्य समय"),
    (r"\bऑफिस\s*टाईम\b", "कामाचे तास"),

    # ── Demo / enrollment / एडमिशन ────────────────────────────────────────
    (r"\bfree\s*demo\b", "free demo"),
    (r"\benroll\b", "enroll"),
    (r"\benrol\b", "enroll"),
    (r"\bॲडमिशन\b", "admission"),
    (r"\bअ‍ॅडमिशन\b", "admission"),
    (r"\bएडमिशन\b", "admission"),
    (r"\bडेमो\b", "demo"),

    # ── Location & Address ────────────────────────────────────────────────
    (r"\bkarve\s*nagar\b", "karvenagar"),
    (r"\bकर्वे\s*नगर\b", "कर्वेनगर"),
    (r"\bपत्ता\b", "address"),
    (r"\bलोकेशन\b", "address"),
    (r"\bएड्रेस\b", "address"),

    # ── Founder / history ─────────────────────────────────────────────────
    (r"\bmanoj\s*potdar\b", "manoj potdar"),
    (r"\bमनोज\s*पोतदार\b", "manoj potdar"),
    (r"\bceo\b", "founder"),
    (r"\bमालक\b", "founder"),
    (r"\bसंस्थापक\b", "founder"),
    (r"\bस्थापना\b", "founded"),
    (r"\bइतिहास\b", "history"),

    # ── Intention Terms ───────────────────────────────────────────────────
    (r"\bफुल\s*फॉर्म\b", "full form"),
    (r"\bफुलफॉर्म\b", "full form"),
    (r"\bपूर्ण\s*नाव\b", "full name"),
    (r"\bऑनलाइन\b", "online"),
    (r"\bऑनलाईन\b", "online"),
    (r"\bऑफलाइन\b", "offline"),
    (r"\bऑफलाईन\b", "offline"),
]

_VOCAB_BRIDGE: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE | re.UNICODE), repl) for pat, repl in _RAW_VOCAB_BRIDGE
]


def _apply_vocab_bridge(text: str) -> str:
    for pat, repl in _VOCAB_BRIDGE:
        text = pat.sub(repl, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core cleaning
# ─────────────────────────────────────────────────────────────────────────────

_STRIP_PUNCT = re.compile(
    r"[^\w\s\u0900-\u097F\u0966-\u096F&.\-]",   # keep word chars, Devanagari, &, ., -
    re.UNICODE,
)
_MULTI_SPACE = re.compile(r"\s{2,}")


def _clean(text: str) -> str:
    """NFC normalize, lowercase, strip noise punctuation, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _STRIP_PUNCT.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedQuery:
    """Result of normalize_query()."""
    text: str   # cleaned, vocab-bridged text for the matcher
    lang: str   # pass-through from caller (unchanged)


def normalize_query(text: str, lang: str = "en") -> NormalizedQuery:
    """
    Normalize a raw STT (or typed) query before FAQ matching.

    Steps: clean -> collapse "M CAD" aliases -> apply vocab bridge.
    """
    text = _clean(text)
    text = _collapse_mcad_aliases(text)
    text = _apply_vocab_bridge(text)
    return NormalizedQuery(text=text, lang=lang)


def normalize_text_only(text: str, lang: str = "en") -> str:
    """Shortcut: returns only the normalized text string. Used at CSV
    keyword-load time by mcad_keyword_matcher.load_faq()."""
    return normalize_query(text, lang).text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quick CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _tests = [
        ("Tell me about the CATIA V5 course",           "en"),
        ("What is the em cad solutions contact number", "en"),
        ("solid works course kitne mahine ka hai",       "hi"),
        ("BIW course ki fee kya hai",                    "hi"),
        ("batch साईझ किती आहे",                          "mr"),
    ]
    print("\n-- mcad_query_normalizer smoke-test --------------------------")
    for raw, lang in _tests:
        result = normalize_query(raw, lang)
        print(f"\n  [{lang}] {raw!r}")
        print(f"      -> {result.text!r}")
