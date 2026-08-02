"""
mcad_keyword_matcher.py  —  Unique-match FAQ scorer for the M CAD
Solutions voice assistant (mcad_solution_faq.csv).

No RAG. No LLM. No cache. Pre-recorded WAV playback only (see app.py) —
this module's only job is: given user text, pick the best-matching FAQ
row and hand back its answer text, language, confidence, and the WAV
filename to play.

Scoring strategy (dual-pass, combined) — same proven approach used for
the Adi/MMCOE admission matcher, generalised to any (lang, question,
answer, keywords, audio_file) CSV:

  1. KEYWORD SCORE  — IDF-weighted keyword hit.
  2. QUESTION SCORE — token overlap between the user query and the FAQ
                      *question* text itself, weighted by per-token
                      discriminator power (a word that appears in only
                      one question gets weight 1.0; one in half the
                      rows gets a much smaller weight). Breaks ties
                      between rows that share keywords but differ in
                      the actual question asked (e.g. "CATIA V5 course"
                      vs "SolidWorks course" both mention "course").
  3. COMBINED       — 0.55 * keyword_score + 0.45 * question_score.
  4. MIXED-LANGUAGE — scored against all three language banks
                      simultaneously as a fallback, so Hinglish input
                      ("CATIA course kitna time lagta hai") still finds
                      the right row even if STT's detected lang is off.
  5. CONFIDENCE GATE — if no candidate clears MIN_COMBINED_SCORE,
                      return the fallback greeting/help text instead of
                      a low-confidence guess.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FAQEntry:
    lang:       str
    topic:      str
    question:   str
    answer:     str
    keywords:   list[str]                       # pre-normalised, split on '|'
    audio_file: str                             # WAV filename to play (from CSV)
    q_tokens:   list[str] = field(default_factory=list)


# ── Fallback answers when nothing matches ─────────────────────────────────────

_FALLBACK: dict[str, str] = {
    "en": (
        "Sorry, I didn't quite understand that. You can ask me about our "
        "CAD courses, domain training programs, placements, fees, timings, "
        "or contact details for M CAD Solutions."
    ),
    "hi": (
        "माफ़ करें, मैं समझ नहीं पाया। आप मुझसे M CAD Solutions के CAD कोर्स, "
        "डोमेन ट्रेनिंग प्रोग्राम, प्लेसमेंट, फीस, समय या संपर्क जानकारी के "
        "बारे में पूछ सकते हैं।"
    ),
    "mr": (
        "माफ करा, मला नीट समजले नाही. तुम्ही M CAD Solutions च्या CAD कोर्सेस, "
        "डोमेन ट्रेनिंग प्रोग्राम, प्लेसमेंट, फी, वेळापत्रक किंवा संपर्क "
        "माहितीबद्दल विचारू शकता."
    ),
}

# WAV to play alongside the fallback text above, keyed by lang.
# These filenames are NOT expected to exist yet in the audio/ folder for a
# fresh project — app.py handles a missing file by simply skipping playback
# (see play_wav()), so add fallback_en.wav / fallback_hi.wav / fallback_mr.wav
# to the audio/ folder whenever convenient; nothing breaks if you don't.
_FALLBACK_AUDIO: dict[str, str] = {
    "en": "fallback_en.wav",
    "hi": "fallback_hi.wav",
    "mr": "fallback_mr.wav",
}


# ── Synonym expansion ─────────────────────────────────────────────────────────
# Additive cross-script / alternate-form synonyms drawn from the CSV's own
# keyword columns (not invented). Each entry: normalised_input_token ->
# token_to_also_inject. Expansion is additive — both forms are searched.
_SYNONYM_MAP: dict[str, str] = {
    # Fees
    "fees":         "फीस",
    "fee":          "फीस",
    "फीस":          "fees",
    "शुल्क":        "फीस",
    "फी":           "फीस",
    # Placement
    "placement":    "प्लेसमेंट",
    "placements":   "प्लेसमेंट",
    "प्लेसमेंट":    "placement",
    "नौकरी":        "प्लेसमेंट",
    "job":          "प्लेसमेंट",
    # Courses
    "course":       "कोर्स",
    "courses":      "कोर्स",
    "कोर्स":        "course",
    "कोर्सेस":      "course",
    # Batch
    "batch":        "बॅच",
    "बैच":          "batch",
    "बॅच":          "batch",
    # Certificate
    "certificate":  "सर्टिफिकेट",
    "सर्टिफिकेट":   "certificate",
    # Timings
    "timings":      "समय",
    "hours":        "समय",
    "समय":          "timings",
    "तास":          "timings",
    "वेळ":          "timings",
    # Contact
    "contact":      "संपर्क",
    "संपर्क":       "contact",
    "phone":        "फोन",
    "फोन":          "phone",
    "number":       "नंबर",
    "नंबर":         "number",
    "email":        "ईमेल",
    "ईमेल":         "email",
    # Location
    "location":     "पता",
    "address":      "पत्ता",
    "पत्ता":        "address",
    "कहाँ":         "location",
    "कुठे":         "location",
    # Founder / history
    "founder":      "संस्थापक",
    "संस्थापक":     "founder",
    "history":      "स्थापना",
    "स्थापना":      "history",
    # Enrollment / demo
    "enroll":       "एनरोल",
    "एनरोल":        "enroll",
    "join":         "एनरोल",
    "demo":         "डेमो",
    # Online / offline
    "online":       "ऑनलाइन",
    "ऑनलाइन":       "online",
    "offline":      "क्लासरूम",
    "classroom":    "क्लासरूम",
    # Batch size / small batch
    "size":         "साईझ",
    "साईझ":         "size",
    "साइज़":        "size",
    # Why choose
    "why":          "क्यों",
    "क्यों":        "why",
    "का":           "why",
}

_MULTI_SYNONYM_MAP: dict[str, list[str]] = {
    # Placement companies keyword differs slightly per language phrasing
    "companies": ["कंपनियां", "कंपन्या"],
}


def _expand_synonyms(
    norm_text: str,
    synonym_map: dict[str, str] | None = None,
    multi_synonym_map: dict[str, list[str]] | None = None,
) -> str:
    """Inject synonym equivalents into the normalised input string.
    Additive and idempotent — original tokens are preserved. Checks both
    single tokens and adjacent word-pairs (bigrams)."""
    if synonym_map is None:
        synonym_map = _SYNONYM_MAP
    if multi_synonym_map is None:
        multi_synonym_map = _MULTI_SYNONYM_MAP

    extra: list[str] = []
    tokens = norm_text.split()
    for tok in tokens:
        mapped = synonym_map.get(tok)
        if mapped and mapped not in norm_text:
            extra.append(mapped)
        for multi in multi_synonym_map.get(tok, []):
            if multi not in norm_text:
                extra.append(multi)
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        mapped = synonym_map.get(bigram)
        if mapped and mapped not in norm_text:
            extra.append(mapped)
    if extra:
        return norm_text + " " + " ".join(extra)
    return norm_text


# Keep: Unicode letters/digits/marks, space, & and - (course names like
# "GD&T", "UG NX"). Devanagari matras (category Mc/Mn) need explicit
# inclusion since \w under re.UNICODE still misses some of them.
_HAS_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _normalise(text: str) -> str:
    """Lowercase + strip punctuation + collapse spaces (NFC)."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Loader ────────────────────────────────────────────────────────────────────

def load_faq(
    csv_path: str | Path,
    normalize_text_only_fn: Callable[[str, str], str] | None = None,
) -> list[FAQEntry]:
    """
    Load mcad_solution_faq.csv (lang, topic, question, answer, keywords,
    audio_file columns) into a FAQEntry list. Keywords are pre-normalised
    for fast matching; q_tokens stores normalised question words for
    question-level scoring.
    """
    if normalize_text_only_fn is None:
        from mcad_query_normalizer import normalize_text_only as normalize_text_only_fn

    entries: list[FAQEntry] = []
    skipped_no_answer = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_kws = row.get("keywords", "").strip()
            lang = row.get("lang", "en").strip()
            # Keywords are stored with a leading '!' marker in the CSV — strip it.
            kws = [
                normalize_text_only_fn(k.strip().lstrip("!"), lang)
                for k in raw_kws.split("|") if k.strip()
            ]
            answer = row.get("answer", "").strip()
            if not answer:
                skipped_no_answer += 1
                continue
            q_norm = _normalise(row["question"].strip())
            entries.append(FAQEntry(
                lang=lang,
                topic=row.get("topic", "").strip(),
                question=row["question"].strip(),
                answer=answer,
                keywords=kws,
                audio_file=row.get("audio_file", "").strip(),
                q_tokens=q_norm.split(),
            ))
    if skipped_no_answer:
        print(f"[Matcher] Skipped {skipped_no_answer} row(s) with no answer.")
    return entries


def _build_word_doc_freq(entries: list[FAQEntry]) -> dict[str, int]:
    """For every word inside any keyword phrase, count how many distinct
    rows use that word — powers an IDF-style weight in _score()."""
    doc_freq: dict[str, int] = {}
    for entry in entries:
        seen_in_entry: set[str] = set()
        for kw in entry.keywords:
            for word in kw.split():
                seen_in_entry.add(word)
        for word in seen_in_entry:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return doc_freq


# ── Question-level discriminator index ────────────────────────────────────────

_Q_STOPWORDS: frozenset[str] = frozenset({
    "what", "is", "the", "are", "at", "in", "of", "for", "does", "mcad",
    "how", "many", "there", "a", "an", "do", "be", "did", "was", "were",
    "to", "from", "with", "which", "who", "when", "where", "can", "i",
    "my", "and", "or", "it", "have", "has", "been", "will", "would",
    "could", "should", "this", "that", "its", "any", "all", "about",
    "get", "its", "not", "no", "on", "by", "if", "as", "up", "out", "me",
    "you", "your", "tell", "solutions", "solution",
    # Generic English words that appear in nearly every row and carry no
    # discriminating signal for question scoring (they still hit keyword
    # score when present as exact keywords in the CSV)
    "course", "courses", "program", "programs", "training",
    "m", "cad", "mcad",
    # Marathi/Hindi particles that appear in nearly every question
    "काय", "आहे", "का", "की", "है", "क्या", "में", "के", "की", "कि",
    "मध्ये", "ची", "चा", "चे", "ला", "ने", "ना", "हे", "हा", "हि",
    "कसे", "कसा", "कोण", "कुठे", "केव्हा", "किती",
    "कैसे", "कौन", "कहाँ", "कब", "बद्दल", "सांगा", "बताएं",
})


def _build_question_token_freq(entries: list[FAQEntry]) -> dict[str, int]:
    """For every CONTENT word in the question column, count how many
    distinct rows contain that word (stopwords excluded)."""
    freq: dict[str, int] = {}
    for entry in entries:
        seen: set[str] = set()
        for tok in entry.q_tokens:
            if tok not in _Q_STOPWORDS and len(tok) > 1:
                seen.add(tok)
        for tok in seen:
            freq[tok] = freq.get(tok, 0) + 1
    return freq


def _q_token_weight(tok: str, q_freq: dict[str, int], total: int) -> float:
    f = q_freq.get(tok, 1)
    w = 1.0 / (1 + (f - 1) * 0.25)
    return max(0.05, w)


def _word_weight(word: str, doc_freq: dict[str, int], total_rows: int) -> float:
    """IDF-style weight in (0, 1]: 1.0 for a word unique to one row,
    falling toward a 0.15 floor as the word appears in more rows."""
    freq = doc_freq.get(word, 1)
    weight = 1.0 / (1 + (freq - 1) * 0.35)
    return max(0.15, weight)


# Pure ASCII word pattern — used to detect English words in hi/mr queries
_ASCII_WORD = re.compile(r'^[a-z0-9&.\-]+$')


def _question_score(
    entry: FAQEntry,
    input_tokens: list[str],
    q_freq: dict[str, int],
    total_rows: int,
    lang: str = "en",
) -> float:
    """Score how well the user's query matches this entry's QUESTION TEXT.

    For hi/mr queries: strip pure ASCII tokens from the input before
    comparing against question tokens. Rationale: generic English words
    like 'course', 'tell', 'about' appear in every row and add noise to
    the question score without discriminating between topics. Technical
    English (CATIA, SolidWorks, etc.) is already captured by keyword score
    which is script-aware. Devanagari structural words ('बद्दल', 'किती',
    'कुठे') carry all the topical signal for hi/mr queries.
    """
    if not entry.q_tokens:
        return 0.0

    # For hi/mr queries, only use Devanagari tokens to match against question
    if lang in ("hi", "mr"):
        effective_input = [
            t for t in input_tokens
            if not _ASCII_WORD.match(t)  # keep Devanagari, drop pure ASCII
        ]
        # If nothing remains after stripping (e.g. fully English query hitting
        # hi/mr rows in cross-lang pass), fall back to full token list
        if not effective_input:
            effective_input = input_tokens
    else:
        effective_input = input_tokens

    input_set = set(effective_input)
    total_weight = 0.0
    matched_weight = 0.0

    for tok in entry.q_tokens:
        if tok in _Q_STOPWORDS or len(tok) <= 1:
            continue
        w = _q_token_weight(tok, q_freq, total_rows)
        total_weight += w
        if tok in input_set:
            matched_weight += w

    if total_weight == 0.0:
        return 0.0

    overlap = matched_weight / total_weight

    best_entry_tok_w = max(
        (_q_token_weight(t, q_freq, total_rows) for t in entry.q_tokens
         if t not in _Q_STOPWORDS and len(t) > 1),
        default=0.0,
    )
    best_matched_w = max(
        (_q_token_weight(t, q_freq, total_rows) for t in entry.q_tokens
         if t not in _Q_STOPWORDS and len(t) > 1 and t in input_set),
        default=0.0,
    )
    discriminator_bonus = 0.3 * (best_matched_w / best_entry_tok_w) if best_entry_tok_w > 0 else 0.0

    return min(1.0, overlap + discriminator_bonus)


# ── Keyword scorer ─────────────────────────────────────────────────────────────

def _base_len(word: str) -> int:
    """Length of word excluding Devanagari combining marks/diacritics."""
    clean = re.sub(r"[\u0900-\u0903\u093C-\u094F\u0951-\u0957\u0962-\u0963]", "", word)
    return len(clean)


def _is_fuzzy_garble(kw_word: str, tok: str, ratio: float) -> bool:
    """Plausible STT mis-spelling gate for Devanagari words: conservative
    (ratio >= 0.72, both words >= 3-4 base chars) to avoid coincidental
    collisions between unrelated short words."""
    if min(_base_len(kw_word), _base_len(tok)) < 3:
        return False
    if min(len(kw_word), len(tok)) < 4:
        return False
    return ratio >= 0.72


def _score(
    entry: FAQEntry,
    norm_input: str,
    input_tokens: list[str],
    doc_freq: dict[str, int],
    total_rows: int,
) -> tuple[float, int, float]:
    """Returns (weighted_match_score, total_keyword_chars, best_keyword_score).

    Short keywords (<=3 chars) are matched as whole words only, to avoid
    false positives (e.g. 'ai' inside another word). Longer keywords use
    substring matching first; Devanagari keywords also get a fuzzy STT-
    garble fallback via SequenceMatcher.
    """
    score = 0.0
    chars = 0
    best_kw_score = 0.0
    for kw in entry.keywords:
        if not kw:
            continue
        if len(kw) <= 3:
            pattern = r'(?<![^\s])' + re.escape(kw) + r'(?![^\s])'
            if re.search(pattern, norm_input):
                idf = _word_weight(kw, doc_freq, total_rows)
                score += idf
                chars += int(len(kw) * idf)
                best_kw_score = max(best_kw_score, idf)
            continue

        if kw in norm_input:
            kw_words_exact = kw.split()
            idf = min(_word_weight(w, doc_freq, total_rows) for w in kw_words_exact)
            score += idf
            chars += int(len(kw) * idf)
            best_kw_score = max(best_kw_score, idf)
            continue

        kw_words = kw.split()
        if not kw_words:
            continue

        word_weights: list[float] = []
        for kw_word in kw_words:
            matched_ratio = 0.0
            if kw_word in input_tokens:
                matched_ratio = 1.0
            else:
                kw_is_dev = _HAS_DEVANAGARI.search(kw_word)
                best_word_ratio = 0.0
                for tok in input_tokens:
                    tok_is_dev = _HAS_DEVANAGARI.search(tok)
                    if bool(kw_is_dev) != bool(tok_is_dev):
                        continue  # only compare within the same script
                    ratio = SequenceMatcher(None, kw_word, tok).ratio()
                    if ratio > best_word_ratio and _is_fuzzy_garble(kw_word, tok, ratio):
                        best_word_ratio = ratio
                if best_word_ratio > 0.0:
                    matched_ratio = best_word_ratio

            if matched_ratio > 0.0:
                idf = _word_weight(kw_word, doc_freq, total_rows)
                word_weights.append(idf * matched_ratio)
            else:
                word_weights.append(0.0)

        if any(w > 0 for w in word_weights):
            kw_score = sum(word_weights) / len(word_weights)
            if kw_score >= 0.25:
                score += kw_score
                chars += int(len(kw) * kw_score * 0.6)
                best_kw_score = max(best_kw_score, kw_score)
    return score, chars, best_kw_score


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    answer:      str
    lang:        str
    confidence:  float
    audio_file:  str
    topic:       str = ""


class MCADKeywordMatcher:
    """
    Instantiate once at startup, call match() per user utterance.

    Parameters
    ----------
    csv_path  : path to mcad_solution_faq.csv (lang, topic, question,
                answer, keywords, audio_file columns)
    normalize_query_fn      : callable(text, lang) -> object with a
                               .text attribute. Defaults to
                               mcad_query_normalizer.normalize_query.
    normalize_text_only_fn  : passed through to load_faq() for
                               per-keyword normalization at load time.
    audio_dir               : folder containing the pre-recorded WAV
                               files named in the CSV's audio_file
                               column. Stored on MatchResult as an
                               absolute path for app.py to play.
    """

    _KW_WEIGHT = 0.55
    _Q_WEIGHT = 0.45
    _MIN_SCORE = 0.12

    def __init__(
        self,
        csv_path: str | Path,
        normalize_query_fn: Callable | None = None,
        normalize_text_only_fn: Callable[[str, str], str] | None = None,
        audio_dir: str | Path | None = None,
    ) -> None:
        if normalize_query_fn is None:
            from mcad_query_normalizer import normalize_query as normalize_query_fn
        self._normalize_query_fn = normalize_query_fn

        self.entries = load_faq(csv_path, normalize_text_only_fn=normalize_text_only_fn)
        self.doc_freq = _build_word_doc_freq(self.entries)
        self.q_freq = _build_question_token_freq(self.entries)
        self.total_rows = len(self.entries)
        self.audio_dir = Path(audio_dir) if audio_dir else Path(csv_path).parent / "audio"
        print(f"[Matcher] Loaded {len(self.entries)} FAQ entries from {Path(csv_path).name}.")

    def match(self, text: str, lang: str = "en") -> MatchResult:
        """
        Find the best matching FAQ row for `text`.

        Returns a MatchResult(answer, lang, confidence, audio_file, topic).
        confidence == 0.0 means the fallback was used.
        """
        normalized = self._normalize_query_fn(text, lang)
        norm = normalized.text
        norm = _expand_synonyms(norm)
        if not norm:
            return MatchResult(
                answer=_FALLBACK.get(lang, _FALLBACK["en"]),
                lang=lang, confidence=0.0,
                audio_file=_FALLBACK_AUDIO.get(lang, _FALLBACK_AUDIO["en"]),
            )

        input_tokens = norm.split()

        def _combined(entry: FAQEntry) -> tuple[float, float, float]:
            kw_raw, _chars, kw_best = _score(
                entry, norm, input_tokens, self.doc_freq, self.total_rows
            )
            extra = max(0.0, kw_raw - kw_best)
            kw_score = min(1.0, kw_best + 0.15 * extra)
            # Pass lang so _question_score can strip ASCII tokens for hi/mr
            q_score = _question_score(entry, input_tokens, self.q_freq, self.total_rows, lang=lang)
            combined = self._KW_WEIGHT * kw_score + self._Q_WEIGHT * q_score
            return combined, kw_score, q_score

        def _best_in(entries: list[FAQEntry]):
            best_entry: Optional[FAQEntry] = None
            best_combined = 0.0
            best_kw = 0.0
            best_q = 0.0
            for entry in entries:
                combined, kw_score, q_score = _combined(entry)
                if combined > best_combined:
                    best_combined = combined
                    best_entry = entry
                    best_kw = kw_score
                    best_q = q_score
            return best_entry, best_combined, best_kw, best_q

        # Pass 1: same-language rows
        same_lang = [e for e in self.entries if e.lang == lang]
        best_entry, best_combined, best_kw, best_q = _best_in(same_lang)

        if best_entry is None or best_combined < self._MIN_SCORE:
            # Pass 2: cross-language fallback (Hinglish / mixed-script input)
            all_entry, all_combined, all_kw, all_q = _best_in(self.entries)
            if all_entry is not None and all_combined > best_combined:
                best_entry, best_combined, best_kw, best_q = all_entry, all_combined, all_kw, all_q

        if best_entry is None or best_combined < self._MIN_SCORE:
            return MatchResult(
                answer=_FALLBACK.get(lang, _FALLBACK["en"]),
                lang=lang, confidence=0.0,
                audio_file=_FALLBACK_AUDIO.get(lang, _FALLBACK_AUDIO["en"]),
            )

        print(
            f"[Matcher] '{text[:40]}' -> '{best_entry.question[:50]}' "
            f"(kw={best_kw:.2f} q={best_q:.2f} combined={best_combined:.2f})"
        )
        return MatchResult(
            answer=best_entry.answer,
            lang=best_entry.lang,
            confidence=best_combined,
            audio_file=best_entry.audio_file,
            topic=best_entry.topic,
        )

    def audio_path(self, audio_file: str, lang: str) -> Path:
        """Resolve a CSV audio_file value to a full path.

        Wav samples are stored one folder per language:
            <audio_dir>/<lang>/<audio_file>
        e.g. output_new/mr/batch_size_mr.wav, output_new/hi/phone_hi.wav
        """
        return self.audio_dir / lang / audio_file


# ── Quick CLI smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_p = Path(__file__).parent / "mcad_solution_faq.csv"
    matcher = MCADKeywordMatcher(csv_p)

    tests = [
        ("Hello!",                                        "en"),
        ("What is your name?",                            "en"),
        ("Tell me about the CATIA V5 course",              "en"),
        ("What is the BIW fixture design course about?",  "en"),
        ("solidworks course kitne din ka hai",             "hi"),
        ("UG NX कोर्सबद्दल सांगा",                          "mr"),
        ("What is the placement rate?",                    "en"),
        ("Which companies hire from you?",                 "en"),
        ("M CAD Solutions चा संपर्क क्रमांक काय आहे?",       "mr"),
        ("Do you offer online classes?",                    "en"),
        ("xyzabc nonsense query",                           "en"),
        ("Thanks a lot, bye!",                               "en"),
    ]

    print("\n-- MCAD matcher smoke-test ------------------------------------")
    for query, lang in tests:
        result = matcher.match(query, lang)
        status = "OK" if result.confidence >= 0.20 else "LOW"
        print(f"\n[{status}] Q [{lang}]: {query}")
        print(f"    A [{result.lang}] (conf={result.confidence:.2f}, wav={result.audio_file}): "
              f"{result.answer[:100]}")