#!/usr/bin/env python3
"""Strict-instruction prompts for heldout_full_sft4152: the imposed-refusal-phrase variants
(add_strict_variants.py, pid = base pid + "_s") whose base prompt is in the full run.

A strict variant has the same question and documents as its base prompt: the base context
verdict is copied (no new judge call) and the train/val split of the base is kept. Variants
whose base verdict does not match their label are dropped, as build_pilot_pairs would.

    python3 dpo/build_strict_prompts.py --pool dpo/data/prompts_hpv4heldout.jsonl \
        --base-dir dpo/data/heldout_full_sft4152 --out-dir dpo/data/heldout_strict_sft4152
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    base = {p["pid"]: p for p in rows(Path(args.base_dir) / "prompts.jsonl")}
    ctx = {r["pid"]: r for r in rows(Path(args.base_dir) / "context_verdicts.jsonl")}
    out, verdicts, skipped = [], [], collections.Counter()
    for p in rows(args.pool):
        if not p.get("is_strict_variant"):
            continue
        b = base.get(p["pid"].removesuffix("_s"))
        if b is None:
            skipped["base_not_in_full_run"] += 1
            continue
        assert b["question"] == p["question"] and b["messages"][1] == p["messages"][1]
        v = ctx[b["pid"]]
        if v["verdict"] != ("unanswerable" if p["is_unanswerable"] else "answerable"):
            skipped[f"base_context_{v['verdict']}"] += 1
            continue
        out.append({**p, "pilot_split": b["pilot_split"], "base_pid": b["pid"]})
        verdicts.append({**v, "pid": p["pid"], "copied_from": b["pid"]})
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in (("prompts.jsonl", out), ("context_verdicts.jsonl", verdicts)):
        with (d / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"pool": args.pool, "n": len(out), "skipped": dict(skipped),
           "label": dict(collections.Counter("U" if p["is_unanswerable"] else "A" for p in out)),
           "unans_mode": dict(collections.Counter(p["unans_mode"] for p in out if p["is_unanswerable"])),
           "split": dict(collections.Counter(p["pilot_split"] for p in out)),
           "strict_phrase": dict(collections.Counter(p["strict_phrase"][:40] for p in out)),
           "prompt_sha256": hashlib.sha256((d / "prompts.jsonl").read_bytes()).hexdigest()}
    (d / "prompts.manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
