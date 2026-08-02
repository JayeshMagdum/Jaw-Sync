"""
test_matcher.py — accuracy + speed test for mcad_keyword_matcher.

Only tests Hindi (hi) and Marathi (mr) rows for now — English is on
hold until it's asked for (matches ACTIVE_LANGS in app.py).

For every hi/mr row in mcad_solution_faq.csv, this feeds the row's own
question text back into MCADKeywordMatcher.match() and checks whether
the matcher lands back on the SAME row (by topic). This measures two
things per query:

  1. TIME   — wall-clock seconds for match() to return (time.perf_counter,
              so it's real elapsed time, not CPU time — matches what a
              user actually waits for).
  2. ACCURACY — did it come back with the correct topic (and language)?

Usage
-----
    python test_matcher.py                # run hi+mr accuracy/speed test
    python test_matcher.py --verbose       # print every single query, not just failures
    python test_matcher.py --lang hi       # test only Hindi
    python test_matcher.py --lang mr       # test only Marathi

Reading the output
-------------------
    Each line looks like:

    [PASS] hi  0.00041s  conf=0.91  topic=phone
    [FAIL] mr  0.00038s  conf=0.18  topic=batch_size -> got 'founder'  Q: ...

    Summary at the end gives:
      - total queries, pass/fail counts, accuracy %
      - min / avg / max / median response time (seconds)
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from mcad_keyword_matcher import MCADKeywordMatcher, load_faq

CSV_PATH = Path(__file__).parent / "mcad_solution_faq.csv"

# Confidence floor used only to flag a PASS as "weak" in the summary —
# a correct topic match with very low confidence is still counted as a
# pass here (matcher.py's own MIN_SCORE gate already decides fallback
# vs real answer); this is just visibility into how close it was.
WEAK_CONF_THRESHOLD = 0.30


def run_test(lang_filter: tuple[str, ...], verbose: bool) -> None:
    matcher = MCADKeywordMatcher(CSV_PATH)
    all_entries = load_faq(CSV_PATH)
    test_entries = [e for e in all_entries if e.lang in lang_filter]

    print(f"\n-- Testing {len(test_entries)} rows for lang(s) {lang_filter} --------------\n")

    results = []          # (pass: bool, elapsed_seconds: float, entry, matched_topic)
    weak_passes = []

    for entry in test_entries:
        t0 = time.perf_counter()
        result = matcher.match(entry.question, entry.lang)
        elapsed = time.perf_counter() - t0

        passed = (result.topic == entry.topic) and (result.lang == entry.lang)
        results.append((passed, elapsed, entry, result))

        if passed and result.confidence < WEAK_CONF_THRESHOLD:
            weak_passes.append((entry, result, elapsed))

        if passed:
            if verbose:
                print(f"[PASS] {entry.lang}  {elapsed:.5f}s  conf={result.confidence:.2f}  "
                      f"topic={entry.topic}")
        else:
            print(f"[FAIL] {entry.lang}  {elapsed:.5f}s  conf={result.confidence:.2f}  "
                  f"topic={entry.topic} -> got {result.topic!r}  Q: {entry.question}")

    # ── Summary ─────────────────────────────────────────────────────────────
    total = len(results)
    passed_count = sum(1 for p, _, _, _ in results if p)
    failed_count = total - passed_count
    accuracy = (passed_count / total * 100) if total else 0.0

    times = [t for _, t, _, _ in results]
    avg_t = statistics.mean(times) if times else 0.0
    min_t = min(times) if times else 0.0
    max_t = max(times) if times else 0.0
    median_t = statistics.median(times) if times else 0.0

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Total queries   : {total}")
    print(f"  Passed          : {passed_count}")
    print(f"  Failed          : {failed_count}")
    print(f"  Accuracy        : {accuracy:.2f}%")
    if weak_passes:
        print(f"  Weak passes     : {len(weak_passes)}  (correct topic but confidence < {WEAK_CONF_THRESHOLD})")
    print("-" * 64)
    print(f"  Min time        : {min_t:.5f}s")
    print(f"  Avg time        : {avg_t:.5f}s")
    print(f"  Median time     : {median_t:.5f}s")
    print(f"  Max time        : {max_t:.5f}s")
    print("=" * 64)

    if failed_count:
        print(f"\n{failed_count} quer{'y' if failed_count == 1 else 'ies'} did not match its own "
              f"row's topic — see [FAIL] lines above. Common causes: two rows sharing near-identical "
              f"keywords/questions, or a synonym map entry pulling the score toward the wrong row.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy + timing test for the M CAD matcher")
    parser.add_argument("--verbose", action="store_true", help="Print every PASS line too, not just FAILs")
    parser.add_argument("--lang", choices=("hi", "mr", "both"), default="both",
                         help="Restrict the test to one language (default: both hi and mr)")
    args = parser.parse_args()

    lang_filter = ("hi", "mr") if args.lang == "both" else (args.lang,)
    run_test(lang_filter, args.verbose)


if __name__ == "__main__":
    main()
