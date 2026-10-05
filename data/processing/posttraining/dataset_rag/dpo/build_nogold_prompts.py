#!/usr/bin/env python3
"""Refusal-pair prompts: held-out answerable HotpotQA prompts with ALL gold chunks removed.
ATTENTION (2026-10-05) : les chunks gold sont retirés SANS remplacement, la version sans réponse a toujours 1-2 documents
de moins que sa jumelle (biais « moins de chunks => refuser »). Ne plus utiliser pour de nouvelles données : passer par
dpo/build_unans_dag.py (code du DAG, remplacement un pour un). Voir DIAGNOSTIC_dpo_2026-10-05.md § 8.


The distractors are kept as they are (same order, same OpenRAG separator), only the chunks
whose title is a gold supporting title are dropped. Every gold title must be found exactly
once in the prompt, and at least --min-docs chunks must remain, otherwise the prompt is
skipped. The answerable twin's answer is kept as `gold_answer` (reference for the judges:
the context audit must confirm it is NOT derivable from the remaining documents).

    python3 dpo/build_nogold_prompts.py --pool dpo/data/prompts_hpv4heldout.base.jsonl \
        --splits dpo/data/heldout_full_sft4152/all.jsonl --n 300 \
        --out dpo/data/refusal_nogold_pilot/prompts.jsonl
"""
import argparse
import collections
import hashlib
import html
import json
import random
import re
from pathlib import Path

MARKER = "Here are the retrieved documents : `"
SEPARATOR = "\n" + "-" * 10 + "\n\n"  # formatting_sft.CONTEXT_SEPARATOR
CHUNK_RE = re.compile(r"\*\s*filename:\s*(.*?)\n\n\[CHUNK_START\].*?\[CHUNK_END\]", re.DOTALL)


def norm_title(t):
    return " ".join(html.unescape(t).split()).casefold()


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strip_gold(system, gold_titles):
    """Return (new_system, kept_titles) or (None, reason)."""
    i = system.find(MARKER)
    if i < 0 or not system.endswith("`"):
        return None, "no_context_block"
    head, block = system[:i + len(MARKER)], system[i + len(MARKER):-1]
    chunks = [(m.group(1), m.group(0)) for m in CHUNK_RE.finditer(block)]
    if SEPARATOR.join(c for _, c in chunks) != block:
        return None, "unparsed_context"
    gold = {norm_title(g) for g in gold_titles}
    hits = collections.Counter(norm_title(t) for t, _ in chunks if norm_title(t) in gold)
    if set(hits) != gold or any(n != 1 for n in hits.values()):
        return None, "gold_not_found_once"
    kept = [(t, c) for t, c in chunks if norm_title(t) not in gold]
    return head + SEPARATOR.join(c for _, c in kept) + "`", [t for t, _ in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="held-out base prompts (prompts_hpv4heldout.base.jsonl)")
    ap.add_argument("--splits", required=True, help="prompts carrying pilot_split, to reuse the train/val split by qid")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None, help="prompts.jsonl of an earlier batch: skip its source prompts")
    args = ap.parse_args()

    split_of = {r["qid"]: r["pilot_split"] for r in rows(args.splits)}
    pool = [p for p in rows(args.pool) if not p["is_unanswerable"] and not p.get("strict_refusal_instruction")
            and p["qid"] in split_of]
    random.Random(args.seed).shuffle(pool)
    if args.exclude:
        seen = {r["source_pid"] for r in rows(args.exclude)}
        pool = [p for p in pool if p["pid"] not in seen]
    out, skipped = [], collections.Counter()
    for p in pool:
        if len(out) >= args.n:
            break
        system, kept = strip_gold(p["messages"][0]["content"], p["gold_titles"])
        if system is None:
            skipped[kept] += 1
            continue
        if len(kept) < args.min_docs:
            skipped["too_few_docs_left"] += 1
            continue
        pid = hashlib.sha256(f"nogold:{p['pid']}".encode()).hexdigest()[:16]
        out.append({**p, "pid": pid, "source_pid": p["pid"],
                    "messages": [{"role": "system", "content": system}, p["messages"][1]],
                    "is_unanswerable": True, "unans_mode": "no_gold", "context_titles": kept,
                    "removed_gold_titles": p["gold_titles"], "gold_titles": [],
                    "pilot_split": split_of[p["qid"]], "sft_target": None, "sft_target_reliable": False})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"pool": args.pool, "n": len(out), "seed": args.seed, "min_docs": args.min_docs,
                "skipped": dict(skipped),
                "hotpot_type": dict(collections.Counter(r["hotpot_type"] for r in out)),
                "split": dict(collections.Counter(r["pilot_split"] for r in out)),
                "docs_left": dict(sorted(collections.Counter(len(r["context_titles"]) for r in out).items())),
                "prompt_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}
    Path(args.out).with_name("prompts.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
