#!/usr/bin/env python3
"""Imposed-phrase balance over a DPO MIX (strict refusal instruction), and capping of one set to restore it.

A strict prompt imposes a refusal phrase. A pair whose chosen IS that phrase pushes towards writing it, a pair whose
rejected is the bare phrase (false refusal) pushes away from it. When the phrase is rejected more often than chosen,
DPO stops copying it (hotpotqa v1.1: imposed-phrase recall 0.66 -> 0.04); far more chosen than rejected risks false
refusals with the phrase. The on-policy configs balance each config; the Jean Zay strict delta run cannot (the Qwen
models almost never refuse wrongly with the phrase), so its phrase-chosen pairs are capped against the whole mix.

Counts per (dataset, phrase): chosen = pairs whose chosen is the imposed phrase (build_strict_pairs.is_exact),
rejected = pairs whose rejected is. The phrase of a pair is `strict_phrase`, else the pool phrase found in the strict
rule of its system prompt.

  report:  python dpo/phrase_balance.py --pairs A.parquet B.jsonl hf:tatqa_offpolicy_qwen35_35b_a3b@0f77a34 ...
  cap:     ... --adjust STRICT.jsonl --out STRICT_balanced.jsonl [--slack 0]
           drops phrase-chosen pairs of STRICT.jsonl (hash order of the pid) until, for every (dataset, phrase),
           chosen <= rejected + slack over the whole mix (all --pairs plus the kept part of --adjust).
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_strict_pairs import is_exact  # noqa: E402
from strict_phrases import EN_PHRASES, RULE_PREFIX  # noqa: E402

REPO = "OpenLLM-France/RAG_DPO_onpolicy"


def norm(t):
    return " ".join(t.lower().split())


def load(spec):
    if spec.startswith("hf:"):  # hf:<config>@<revision>, train + val
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        cfg, _, rev = spec[3:].partition("@")
        out = []
        for s in ("train", "val"):
            p = hf_hub_download(REPO, f"{cfg}/{s}.parquet", repo_type="dataset", revision=rev or None)
            out += pq.read_table(p).to_pylist()
        return out
    if spec.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(spec).to_pylist()
    return [json.loads(line) for line in open(spec, encoding="utf-8") if line.strip()]


def text(m):
    return m[-1]["content"] if isinstance(m, list) else m


def phrase_of(r):
    if r.get("strict_phrase"):
        return r["strict_phrase"]
    system = next((m["content"] for m in r["prompt"] if m["role"] == "system"), "")
    rule = next((line for line in system.splitlines() if line.startswith(RULE_PREFIX)), "")
    hits = [p for p in EN_PHRASES if norm(p) in norm(rule)]
    return max(hits, key=len) if hits else None


def dataset_of(r):
    return r.get("dataset") or ("tatqa" if "_" in str(r.get("qid", "")) else "hotpotqa")


def tally(rows, counts):
    for r in rows:
        ph = phrase_of(r)
        if not ph:
            continue
        key = (dataset_of(r), ph)
        counts[key]["chosen"] += is_exact(text(r["chosen"]), ph)
        counts[key]["rejected"] += is_exact(text(r["rejected"]), ph)


def report(counts):
    out = {}
    for ds in sorted({k[0] for k in counts}):
        cells = {k[1]: v for k, v in counts.items() if k[0] == ds and (v["chosen"] or v["rejected"])}
        out[ds] = {"chosen": sum(v["chosen"] for v in cells.values()),
                   "rejected": sum(v["rejected"] for v in cells.values()),
                   "phrases_rejected_gt_chosen": sum(v["rejected"] > v["chosen"] for v in cells.values()),
                   "phrases_chosen_gt_rejected": sum(v["chosen"] > v["rejected"] for v in cells.values()),
                   "phrases": len(cells)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=[])
    ap.add_argument("--adjust")
    ap.add_argument("--out")
    ap.add_argument("--slack", type=int, default=0)
    a = ap.parse_args()
    base = collections.defaultdict(collections.Counter)
    for spec in a.pairs:
        tally(load(spec), base)
    res = {"mix_without_adjusted": report(base)}
    if a.adjust:
        rows = load(a.adjust)
        counts = collections.defaultdict(collections.Counter, {k: collections.Counter(v) for k, v in base.items()})
        keep, dropped = [], collections.Counter()
        # non-phrase-chosen pairs of the adjusted set first (they may add phrase-rejected pairs), then the phrase-chosen
        # ones in hash order, each kept only while its (dataset, phrase) cell stays within chosen <= rejected + slack
        def chosen_phrase(r):
            ph = phrase_of(r)
            return bool(ph and is_exact(text(r["chosen"]), ph))
        rest = [r for r in rows if not chosen_phrase(r)]
        tally(rest, counts)
        keep += rest
        for r in sorted((r for r in rows if chosen_phrase(r)),
                        key=lambda r: hashlib.sha256(f"balance:{r['pid']}".encode()).hexdigest()):
            key = (dataset_of(r), phrase_of(r))
            if counts[key]["chosen"] + 1 <= counts[key]["rejected"] + a.slack:
                counts[key]["chosen"] += 1
                counts[key]["rejected"] += is_exact(text(r["rejected"]), key[1])
                keep.append(r)
            else:
                dropped[key[0]] += 1
        res.update(adjusted_in=len(rows), adjusted_kept=len(keep), dropped_phrase_chosen=dict(dropped),
                   mix_with_adjusted=report(counts))
        if a.out:
            order = {id(r): i for i, r in enumerate(rows)}
            with open(a.out, "w", encoding="utf-8") as f:
                for r in sorted(keep, key=lambda r: order[id(r)]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
