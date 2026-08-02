"""
test_marathi.py — Dedicated Marathi accuracy and performance benchmark suite
for M CAD Solutions FAQ Assistant.

Supports testing:
  1. Canonical CSV Marathi queries
  2. Spoken/colloquial Marathi variation dataset (marathi_test_queries.json)

Usage:
    python test_marathi.py                    # Run both canonical + variation tests
    python test_marathi.py --canonical-only   # Run only CSV canonical questions
    python test_marathi.py --variation-only   # Run only variation test queries
    python test_marathi.py --verbose          # Print details for every single test case
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

# Force UTF-8 standard output for Windows CLI environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcad_keyword_matcher import MCADKeywordMatcher, load_faq

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "mcad_solution_faq.csv"
VARIATION_PATH = BASE_DIR / "marathi_test_queries.json"

WEAK_CONF_THRESHOLD = 0.30


def test_canonical(matcher: MCADKeywordMatcher, verbose: bool) -> tuple[int, int, list]:
    all_entries = load_faq(CSV_PATH)
    mr_entries = [e for e in all_entries if e.lang == "mr"]

    print(f"\n[1/2] Testing {len(mr_entries)} Canonical Marathi CSV Questions...")
    print("=" * 64)

    results = []
    for entry in mr_entries:
        t0 = time.perf_counter()
        res = matcher.match(entry.question, "mr")
        elapsed = time.perf_counter() - t0

        passed = (res.topic == entry.topic)
        results.append({
            "passed": passed,
            "elapsed": elapsed,
            "topic": entry.topic,
            "query": entry.question,
            "got_topic": res.topic,
            "confidence": res.confidence
        })

        tag = "PASS" if passed else "FAIL"
        if not passed or verbose:
            got_str = f" -> got '{res.topic}'" if not passed else ""
            print(f"  [{tag}] {elapsed:.5f}s  conf={res.confidence:.2f}  topic={entry.topic:<22}{got_str} | Q: {entry.question}")

    passed_cnt = sum(1 for r in results if r["passed"])
    print(f"Canonical Result: {passed_cnt}/{len(mr_entries)} passed ({(passed_cnt/len(mr_entries))*100:.1f}%)\n")
    return passed_cnt, len(mr_entries), results


def test_variations(matcher: MCADKeywordMatcher, verbose: bool) -> tuple[int, int, list]:
    if not VARIATION_PATH.exists():
        print(f"\n[2/2] Warning: Variation dataset {VARIATION_PATH.name} not found. Skipping.")
        return 0, 0, []

    with open(VARIATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    total_queries = sum(len(tc["queries"]) for tc in test_cases)

    print(f"[2/2] Testing {total_queries} Spoken Marathi Query Variations across {len(test_cases)} topics...")
    print("=" * 64)

    results = []
    for tc in test_cases:
        expected_topic = tc["topic"]
        for query in tc["queries"]:
            t0 = time.perf_counter()
            res = matcher.match(query, "mr")
            elapsed = time.perf_counter() - t0

            passed = (res.topic == expected_topic)
            results.append({
                "passed": passed,
                "elapsed": elapsed,
                "topic": expected_topic,
                "query": query,
                "got_topic": res.topic,
                "confidence": res.confidence
            })

            tag = "PASS" if passed else "FAIL"
            if not passed or verbose:
                arrow = f" -> GOT '{res.topic}'" if not passed else ""
                print(f"  [{tag}] {elapsed:.5f}s  conf={res.confidence:.2f}  topic={expected_topic:<22}{arrow} | Q: {query}")

    passed_cnt = sum(1 for r in results if r["passed"])
    print(f"Variation Result: {passed_cnt}/{total_queries} passed ({(passed_cnt/total_queries)*100:.1f}%)\n")
    return passed_cnt, total_queries, results


def print_summary(all_results: list[dict]) -> None:
    if not all_results:
        print("No test results to summarize.")
        return

    total = len(all_results)
    passed_count = sum(1 for r in all_results if r["passed"])
    failed_count = total - passed_count
    accuracy = (passed_count / total * 100) if total else 0.0

    weak_passes = [r for r in all_results if r["passed"] and r["confidence"] < WEAK_CONF_THRESHOLD]
    times = [r["elapsed"] for r in all_results]
    avg_t = statistics.mean(times)
    min_t = min(times)
    max_t = max(times)
    median_t = statistics.median(times)

    print("=" * 64)
    print("MARATHI MATCHING BENCHMARK SUMMARY")
    print("=" * 64)
    print(f"  Total Marathi Queries : {total}")
    print(f"  Passed                : {passed_count}")
    print(f"  Failed                : {failed_count}")
    print(f"  Accuracy Rate         : {accuracy:.2f}%")
    if weak_passes:
        print(f"  Weak Passes           : {len(weak_passes)} (confidence < {WEAK_CONF_THRESHOLD})")
    print("-" * 64)
    print(f"  Min Response Time     : {min_t:.5f}s ({min_t*1000:.2f} ms)")
    print(f"  Avg Response Time     : {avg_t:.5f}s ({avg_t*1000:.2f} ms)")
    print(f"  Median Response Time  : {median_t:.5f}s ({median_t*1000:.2f} ms)")
    print(f"  Max Response Time     : {max_t:.5f}s ({max_t*1000:.2f} ms)")
    print("=" * 64)

    if failed_count > 0:
        print("\nFailed Queries Breakdown:")
        for r in all_results:
            if not r["passed"]:
                print(f"  - [{r['topic']}] Expected '{r['topic']}', got '{r['got_topic']}' (conf={r['confidence']:.2f}) | Q: '{r['query']}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Marathi Matcher Accuracy & Speed Benchmark")
    parser.add_argument("--canonical-only", action="store_true", help="Run only canonical CSV questions")
    parser.add_argument("--variation-only", action="store_true", help="Run only variation benchmark queries")
    parser.add_argument("--verbose", action="store_true", help="Print details for every test query")
    args = parser.parse_args()

    matcher = MCADKeywordMatcher(CSV_PATH)
    all_results = []

    if not args.variation_only:
        p1, t1, r1 = test_canonical(matcher, args.verbose)
        all_results.extend(r1)

    if not args.canonical_only:
        p2, t2, r2 = test_variations(matcher, args.verbose)
        all_results.extend(r2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
