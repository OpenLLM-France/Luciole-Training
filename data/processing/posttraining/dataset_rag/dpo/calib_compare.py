#!/usr/bin/env python3
"""Agreement between a new answer judge deployment (e.g. Qwen3.8-27B served on Jean Zay) and verdicts already made
with reasoning on the endpoint / DGX (dpo/data/jz_run/calib/<dataset>/reference_verified.jsonl, build_jz_run.py).
Same completions, same prompts, reasoning on both sides. Exit code 1 when answer_correct agreement < --min.
Reference: the DGX local judge agreed 95-100 % with the endpoint on these booleans (2026-10-01).
    python dpo/calib_compare.py --dir dpo/data/jz_run/calib/hotpotqa --dir dpo/data/jz_run/calib/tatqa [--min 0.9]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot_pairs import is_good  # noqa: E402

FIELDS = ("refused", "mixed", "answer_correct", "answer_supported", "reasoning_supported", "citation_supported")


def last(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("judge"):
                out[(r["pid"], r["k"])] = r["judge"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--new", default="verified_jz.jsonl")
    ap.add_argument("--min", type=float, default=0.9)
    a = ap.parse_args()
    agree, n = collections.Counter(), collections.Counter()
    for d in map(Path, a.dir):
        ref, new = last(d / "reference_verified.jsonl"), last(d / a.new)
        for key in ref.keys() & new.keys():
            r, w = ref[key], new[key]
            for f in FIELDS + ("is_good",):
                try:
                    x, y = (is_good(r, False), is_good(w, False)) if f == "is_good" else (r.get(f), w.get(f))
                except KeyError:
                    continue
                if x is None or y is None:
                    continue
                n[f] += 1
                agree[f] += x == y
        n["_compared"] += len(ref.keys() & new.keys())
        n["_reference"] += len(ref)
    rep = {"compared": n["_compared"], "reference": n["_reference"],
           **{f: round(agree[f] / n[f], 3) for f in FIELDS + ("is_good",) if n[f]}}
    print(json.dumps(rep, indent=1))
    if n["_compared"] < 0.9 * n["_reference"]:
        sys.exit(f"!! only {n['_compared']}/{n['_reference']} completions judged")
    if rep.get("answer_correct", 0) < a.min:
        sys.exit(f"!! answer_correct agreement {rep.get('answer_correct')} < {a.min}")


if __name__ == "__main__":
    main()
