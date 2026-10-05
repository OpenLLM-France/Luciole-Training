#!/usr/bin/env python3
"""Deterministic checks of DPO pairs taken from the colleague's Delta Learning generation (JZ,
`create_preference_data.py` DPOFilter), applied to our pairs as a report: nothing is filtered.

- lengths in tokens of the policy tokenizer (completion only) and characters: chosen longer rate,
  median ratio, and his length filter (disabled in his runs): |ratio| > 1.3 AND |diff| > 100 tokens;
- identical answers after his normalization (lowercase, whitespace, punctuation);
- "Qwen" mentioned in the chosen (identity leak of a Qwen-written chosen);
- CJK characters in the chosen.

    .venv/bin/python dpo/check_pairs_static.py --pairs dpo/data/publish_onpolicy_sft4152_v2/*/*.parquet \
        --tokenizer models/sft_newrag_step4152 --out dpo/data/jz_methods_audit/static.json
"""
import argparse
import collections
import json
import re
import statistics
from pathlib import Path

_CJK = re.compile(r"[一-鿿]")


def read_pairs(path):
    if str(path).endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(path).to_dict("records")
        for r in rows:  # parquet gives numpy arrays of dicts
            for k in ("prompt", "chosen", "rejected"):
                r[k] = [dict(m) for m in r[k]]
            # published parquets have no `dataset` column: <dataset>/<split>.parquet
            r.setdefault("dataset", Path(path).parent.name)
        return rows
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def normalize(text):
    text = " ".join(text.lower().strip().split())
    return re.sub(r"[^\w\s]", "", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--group-by", default="dataset,category")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    keys = args.group_by.split(",")

    groups = collections.defaultdict(list)
    for p in args.pairs:
        for r in read_pairs(p):
            c, rj = r["chosen"][-1]["content"], r["rejected"][-1]["content"]
            tc = len(tok(c, add_special_tokens=False).input_ids)
            tr = len(tok(rj, add_special_tokens=False).input_ids)
            x = {"tc": tc, "tr": tr, "cc": len(c), "cr": len(rj),
                 "ratio": tc / tr if tr else float("inf"),
                 "same": normalize(c) == normalize(rj),
                 "qwen": "qwen" in c.lower(), "cjk": bool(_CJK.search(c))}
            x["len_filter"] = (x["ratio"] > 1.3 or x["ratio"] < 1 / 1.3) and abs(tc - tr) > 100
            groups["/".join(str(r.get(k)) for k in keys)].append(x)
            groups["ALL"].append(x)

    res = {}
    for k, xs in sorted(groups.items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        n = len(xs)
        res[k] = {
            "n": n,
            "chosen_longer_tok": round(sum(x["tc"] > x["tr"] for x in xs) / n, 3),
            "median_ratio_tok": round(statistics.median(x["ratio"] for x in xs), 2),
            "median_tok_chosen": statistics.median(x["tc"] for x in xs),
            "median_tok_rejected": statistics.median(x["tr"] for x in xs),
            "median_chars_chosen": statistics.median(x["cc"] for x in xs),
            "median_chars_rejected": statistics.median(x["cr"] for x in xs),
            "len_filter_would_drop": round(sum(x["len_filter"] for x in xs) / n, 3),
            "len_filter_drop_chosen_longer": round(sum(x["len_filter"] and x["tc"] > x["tr"] for x in xs) / n, 3),
            "identical": sum(x["same"] for x in xs),
            "chosen_mentions_qwen": sum(x["qwen"] for x in xs),
            "chosen_has_cjk": sum(x["cjk"] for x in xs),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{'groupe':38s} {'n':>5s} {'c>r':>6s} {'ratio':>6s} {'tok c':>6s} {'tok r':>6s} "
          f"{'filtre':>7s} {'dont c+':>7s} {'ident':>5s} {'qwen':>5s} {'cjk':>4s}")
    for k, x in res.items():
        print(f"{k:38s} {x['n']:5d} {x['chosen_longer_tok']:6.3f} {x['median_ratio_tok']:6.2f} "
              f"{x['median_tok_chosen']:6.0f} {x['median_tok_rejected']:6.0f} {x['len_filter_would_drop']:7.3f} "
              f"{x['len_filter_drop_chosen_longer']:7.3f} {x['identical']:5d} {x['chosen_mentions_qwen']:5d} "
              f"{x['chosen_has_cjk']:4d}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
