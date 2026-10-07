#!/usr/bin/env python3
"""Prompts to generate on Jean Zay with Qwen3.5-35B-A3B (chosen) and Qwen3.5-4B (rejected), for pairs whose two
sides are written by the two Qwen3.5 models (no policy text), with a large share of refusal pairs.

Source: the held-out on-policy pools, generic refusal instruction (HotpotQA: heldout_full_sft4152, TAT-QA:
heldout_tatqa_generic_sft4152), questions the SFT policy never saw. Every question comes as answerable (gold documents
present) and, for part of them, unanswerable (gold document removed: drop_gold / partial_hop).
Written: the prompts usable for training (all the pool prompts with --all), with three flags:
context_verdict_ok (the context verdict of Qwen3.8-27B agrees with the construction), in_8802410_validation (the
validation questions of OpenLLM-France/RAG_DPO_onpolicy@8802410 (HotpotQA: pilot_split val; TAT-QA:
the validation tables, i.e. the release hash split), in_paired_eval (the questions of the paired evaluation with and
without the key document, dpo/data/p9_paired_eval; TAT-QA: their tables).
Prompts are the published ones (refusal rule without its "invite the user to rephrase ..." clause, as in the release).

Output dpo/data/jz_gen_prompts/: prompts_<dataset>.jsonl (full records: pid, messages + the fields the judges need),
context_verdicts_<dataset>.jsonl, manifest.json, README.md.
    cd $RAG_ROOT && .venv/bin/python dpo/build_jz_prompts.py [--all]
"""
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_text import published_prompt  # noqa: E402

D = HERE / "data"
POOLS = {"hotpotqa": D / "heldout_full_sft4152", "tatqa": D / "heldout_tatqa_generic_sft4152"}
OUT = Path(os.environ.get("JZ_PROMPTS_OUT", D / "jz_gen_prompts"))  # env: Airflow run dir


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def table(qid):
    return qid.split("_", 1)[0]


def main():
    val_q = {r["qid"] for r in rows(D / "rag_dpo_onpolicy_8802410" / "val.jsonl")}
    ev_q = {p["qid"] for p in rows(D / "p9_paired_eval" / "prompts.jsonl")}
    bad_tables = {table(q) for q in val_q | ev_q if "_" in q}
    OUT.mkdir(exist_ok=True)
    man = {"source_pools": {k: str(v.relative_to(HERE.parent)) for k, v in POOLS.items()}, "datasets": {}}
    for ds, pool in POOLS.items():
        cv = {r["pid"]: r for r in rows(pool / "context_verdicts.jsonl")}
        st, kept = collections.Counter(), []
        for p in rows(pool / "prompts.jsonl"):
            # left out (decision 2026-10-05): context verdict contradicting the construction, 8802410 validation,
            # paired evaluation; --all keeps them (flags only)
            q = p["qid"]
            want = "unanswerable" if p["is_unanswerable"] else "answerable"
            flags = {"context_verdict_ok": cv.get(p["pid"], {}).get("verdict") == want,
                     "in_8802410_validation": (q in val_q or p.get("pilot_split") == "val") if ds == "hotpotqa" else
                     (table(q) in bad_tables or int(hashlib.sha256(f"42:{table(q)}".encode()).hexdigest(), 16) % 100 < 10),
                     "in_paired_eval": q in ev_q or (ds == "tatqa" and table(q) in {table(x) for x in ev_q if "_" in x})}
            usable = flags["context_verdict_ok"] and not flags["in_8802410_validation"] and not flags["in_paired_eval"]
            if not usable and "--all" not in sys.argv:
                st["left_out"] += 1
                continue
            kept.append({**p, "messages": published_prompt(p["messages"]), "dataset": ds, **flags})
            st["prompts/" + (p.get("unans_mode") or "answerable")] += 1
            st["usable_for_training/" + (p.get("unans_mode") or "answerable")] += flags["context_verdict_ok"] and \
                not flags["in_8802410_validation"] and not flags["in_paired_eval"]
            for k, v in flags.items():
                st[k] += v
        with open(OUT / f"prompts_{ds}.jsonl", "w", encoding="utf-8") as f:
            for p in kept:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(OUT / f"context_verdicts_{ds}.jsonl", "w", encoding="utf-8") as f:
            for p in kept:
                if p["pid"] in cv:
                    f.write(json.dumps(cv[p["pid"]], ensure_ascii=False) + "\n")
        qs = collections.defaultdict(set)
        for p in kept:
            qs[p["qid"]].add(p["is_unanswerable"])
        man["datasets"][ds] = {**dict(sorted(st.items())), "prompts": len(kept), "questions": len(qs),
                               "questions_with_both_versions": sum(len(v) == 2 for v in qs.values()),
                               "sha256": hashlib.sha256((OUT / f"prompts_{ds}.jsonl").read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
