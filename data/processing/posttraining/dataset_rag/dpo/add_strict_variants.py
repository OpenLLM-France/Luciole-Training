#!/usr/bin/env python3
"""Add strict-refusal-instruction variants to the DPO prompt pool (stdlib only).

Probe on a GPU server, 2026-09-14, fullft_hotpotqav4_unans010, answerable prompts:
  generic instruction ("state that the documents do not allow you to answer")
      -> ~1 % of prompts with at least one refusal in 5 samples
  strict instruction ("respond exactly with: <phrase>")
      -> 8 % greedy false refusals, 14 % of prompts with at least one refusal
Without these variants the pool yields almost only "refusing is better" pairs.

A variant = same question and context, generic refusal rule replaced by one of the 5
strict rules of hotpotqa_v4 (--phrases v4, default, reproduces the existing pools), or by a
phrase and a wording drawn from the pid in strict_phrases.py (--phrases pool: 30 phrases,
7 wordings, lighteval's phrase held out); pid + "_s", `strict_phrase` set. For unanswerable
variants the SFT fallback target becomes the phrase itself (obeying the instruction).
The pool is rewritten shuffled; sampling/scoring resume by pid, so rows already
processed are kept.

    python3 dpo/add_strict_variants.py --pool dpo/data/prompts_hpv4.jsonl \
        --n-answerable 3000 --n-unanswerable 800
"""
import argparse
import json
import random
import re

from strict_phrases import pick_strict

GENERIC = ("   * If the context is **insufficient** to answer the question, state that the retrieved "
           "documents do not allow you to answer and invite the user to rephrase the query or add "
           "relevant documents.")
PHRASES = [
    "The available documents do not contain sufficient information to answer this question. You may want to rephrase it or expand your document base.",
    "I'm unable to answer your question based on the provided context. Please clarify your query or provide additional keywords.",
    "Based on the retrieved context, I cannot provide an answer to your question. Consider rephrasing it or adding more relevant documents.",
    "The context provided does not allow me to answer your question. Could you reformulate your query or add additional relevant documents?",
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?",
]
STRICT_RE = re.compile(r'respond exactly with: "(.+?)"\s*$', re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-answerable", type=int, default=3000)
    ap.add_argument("--n-unanswerable", type=int, default=800)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--phrases", choices=("v4", "pool"), default="v4",
                    help="v4: the 5 phrases of hotpotqa_v4, one wording; pool: strict_phrases.pick_strict")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    base = [r for r in rows if not r["pid"].endswith("_s")]
    for r in base:  # strict phrase of the native strict prompts of v4
        m = STRICT_RE.search(r["messages"][0]["content"])
        r["strict_phrase"] = m.group(1) if m else None
        r["is_strict_variant"] = False
    eligible = [r for r in base if GENERIC in r["messages"][0]["content"]]
    ans = [r for r in eligible if not r["is_unanswerable"]]
    una = [r for r in eligible if r["is_unanswerable"]]
    rng.shuffle(ans)
    rng.shuffle(una)
    variants = []
    for r in ans[: args.n_answerable] + una[: args.n_unanswerable]:
        if args.phrases == "v4":
            ph, rule_id = rng.choice(PHRASES), "v4"
            rule = f'   * If the context is **insufficient** to answer the question, respond exactly with: "{ph}"'
        else:
            ph, rule, rule_id = pick_strict(r["pid"])
        v = json.loads(json.dumps(r))
        v["pid"] = r["pid"] + "_s"
        v["messages"][0]["content"] = r["messages"][0]["content"].replace(GENERIC, rule)
        v["strict_phrase"] = ph
        v["strict_rule_id"] = rule_id
        v["strict_refusal_instruction"] = True
        v["is_strict_variant"] = True
        if v["is_unanswerable"]:
            v["sft_target"] = ph
            v["sft_target_short_refusal"] = True
        variants.append(v)
    out = base + variants
    rng.shuffle(out)
    with open(args.pool, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"base={len(base)} variants: answerable={min(len(ans), args.n_answerable)} "
          f"unanswerable={min(len(una), args.n_unanswerable)} total={len(out)}")


if __name__ == "__main__":
    main()
