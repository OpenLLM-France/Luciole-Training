#!/usr/bin/env python3
"""Second pass of the two-pass judging: list the next chosen candidates to verify with reasoning.

For every prompt whose first-pass (no reasoning) judgments offer a pair (a good sample and a
rejected candidate) but no chosen confirmed yet in --verified, emit its next good sample in the
chosen order (greedy first). Run in rounds: judge the emitted list with reasoning, append to
--verified, select again; a prompt stops once one chosen is confirmed or none is left.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot_pairs import candidates, rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--scored", required=True, help="first-pass judgments")
    ap.add_argument("--verified", required=True, help="second-pass judgments so far (may not exist)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompts = {p["pid"]: p for p in rows(args.prompts)}
    contexts = {r["pid"]: r for r in rows(args.contexts)}
    scored = collections.defaultdict(dict)
    for r in rows(args.scored):
        scored[r["pid"]][r["k"]] = r
    verified = {(r["pid"], r["k"]): r for r in rows(args.verified)} if os.path.exists(args.verified) else {}
    todo, stats = [], collections.Counter()
    for pid, p in prompts.items():
        wanted = "unanswerable" if p["is_unanswerable"] else "answerable"
        if contexts.get(pid, {}).get("verdict") != wanted:
            continue
        cand = candidates(p, list(scored[pid].values()), verified)
        if cand is None or not any(cand[c] for c in cand["bad_keys"]):
            continue
        if cand["good"]:
            stats["confirmed"] += 1
        elif cand["good_unverified"]:
            todo.append({"pid": pid, "k": cand["good_unverified"][0]["k"]})
            stats["to_verify"] += 1
        else:
            stats["exhausted" if cand["good_first_pass"] else "no_first_pass_good"] += 1
    with open(args.out, "w", encoding="utf-8") as f:
        for t in todo:
            f.write(json.dumps(t) + "\n")
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
