#!/usr/bin/env python3
"""Strict twins of the Jean Zay delta prompts: every prompt of dpo/data/jz_run (generic refusal rule) again, with the
rule replaced by a strict one ("respond exactly with: <phrase>"), same question and documents.

The 2026-10-06 run had generic prompts only (build_jz_prompts.py reads the generic pools, build_unans_dag.py redraws the
strict slice). Twins rather than the older strict pools: exclusions already applied, unanswerable versions from the
DAG code (one-for-one replacement, no nogold clones), identical contexts (context verdicts reused), and generic/strict
pairs on the same data. Phrase and wording drawn from the pid in strict_phrases.pick_strict (30 phrases, 7 wordings,
the lighteval phrases held out), as heldout_strictpool_sft4152.

Record: pid + "_s", strict_phrase, strict_rule_id, strict_refusal_instruction / is_strict_variant True, generic_pid,
chosen_kind ("phrase" for unanswerable: the chosen is the imposed phrase, checked lexically, no judge call; "answer"
otherwise).
Output dpo/data/jz_run_strict/<gen>/<dataset>/{prompts,context_verdicts}.jsonl, calib/ (inputs only), manifest.json.
    venv/bin/python dpo/build_jz_strict_prompts.py [--src dpo/data/jz_run] [--out dpo/data/jz_run_strict] [--share 1.0]
--share < 1: keep that share of the QUESTIONS (hash of the qid), both versions of a question together.
--quota hotpotqa:A=500,U=300 --quota tatqa:A=500,U=650: that many answerable / unanswerable twins per dataset, drawn by
hash of the pid (decision 2026-10-08: ~2,000 strict prompts; the unanswerable ones give the phrase-chosen pairs that
balance the phrase-rejected excess of tatqa_offpolicy_qwen35_35b_a3b, the answerable ones keep answering under a strict
rule; the Qwen models almost never refuse wrongly with the phrase, so phrase-rejected pairs will be rare here).
"""
import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strict_phrases import HELD_OUT, RULE_PREFIX, pick_strict  # noqa: E402

D = HERE / "data"
GENS = ("qwen35_35b_a3b", "qwen35_4b")
GENERIC_RULE = RULE_PREFIX + "state that the retrieved documents do not allow you to answer."
CALIB_INPUTS = ("prompts.jsonl", "context_verdicts.jsonl", "chosen_qwen35_35b_a3b.jsonl", "reference_verified.jsonl",
                "only.jsonl")


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(p, rs):
    with open(p, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(D / "jz_run"))
    ap.add_argument("--out", default=str(D / "jz_run_strict"))
    ap.add_argument("--share", type=float, default=1.0)
    ap.add_argument("--quota", action="append", default=[], help="dataset:A=n,U=m")
    a = ap.parse_args()
    quota = {}
    for q in a.quota:
        ds, spec = q.split(":")
        quota[ds] = {k: int(v) for k, v in (x.split("=") for x in spec.split(","))}
    src, out = Path(a.src), Path(a.out)
    held = [" ".join(h.lower().split()) for h in HELD_OUT]
    man = {"source": str(src), "share": a.share, "quota": quota, "datasets": {}}
    for ds in ("hotpotqa", "tatqa"):
        st = collections.Counter()
        base = list(rows(src / GENS[0] / ds / "prompts.jsonl"))
        assert base == list(rows(src / GENS[1] / ds / "prompts.jsonl")), f"{ds}: prompts differ between generators"
        cv = {r["pid"]: r for r in rows(src / GENS[0] / ds / "context_verdicts.jsonl")}
        if ds in quota:  # quota draw: hash order of the pid, per answerable / unanswerable
            order = sorted(base, key=lambda p: hashlib.sha256(f"strictq:{p['pid']}".encode()).hexdigest())
            want = {"A": quota[ds].get("A", 0), "U": quota[ds].get("U", 0)}
            keep = set()
            for p in order:
                k = "U" if p["is_unanswerable"] else "A"
                if want[k] > 0:
                    keep.add(p["pid"])
                    want[k] -= 1
            st["skipped_quota"] = len(base) - len(keep)
            base = [p for p in base if p["pid"] in keep]
        twins, verdicts = [], []
        for p in base:
            if a.share < 1 and int(hashlib.sha256(f"strict:{p['qid']}".encode()).hexdigest(), 16) % 10_000 >= a.share * 10_000:
                st["skipped_share"] += 1
                continue
            system = p["messages"][0]["content"]
            if system.count(GENERIC_RULE) != 1 or p.get("strict_refusal_instruction"):
                st["skipped_no_generic_rule"] += 1
                continue
            phrase, rule, rule_id = pick_strict(p["pid"] + "_s")
            assert " ".join(phrase.lower().split()) not in held
            t = json.loads(json.dumps(p))
            t["pid"] = p["pid"] + "_s"
            t["messages"][0]["content"] = system.replace(GENERIC_RULE, rule)
            t.update(strict_phrase=phrase, strict_rule_id=rule_id, strict_refusal_instruction=True,
                     is_strict_variant=True, generic_pid=p["pid"],
                     chosen_kind="phrase" if p["is_unanswerable"] else "answer")
            twins.append(t)
            verdicts.append({**cv[p["pid"]], "pid": t["pid"], "copied_from_generic": p["pid"]})
            st["strict/" + ("unanswerable" if p["is_unanswerable"] else "answerable")] += 1
            st[f"rule/{rule_id}"] += 1
        for gen in GENS:
            d = out / gen / ds
            d.mkdir(parents=True, exist_ok=True)
            write(d / "prompts.jsonl", twins)
            write(d / "context_verdicts.jsonl", verdicts)
        phrases = collections.Counter(t["strict_phrase"] for t in twins)
        man["datasets"][ds] = {**dict(sorted(st.items())), "prompts": len(twins), "distinct_phrases": len(phrases),
                               "max_prompts_per_phrase": max(phrases.values()),
                               "sha256": hashlib.sha256((out / GENS[0] / ds / "prompts.jsonl").read_bytes()).hexdigest()}
        # calibration of the judge job: the same answer completions as the generic run (inputs only, judged afresh)
        c = out / "calib" / ds
        c.mkdir(parents=True, exist_ok=True)
        for name in CALIB_INPUTS:
            shutil.copy(src / "calib" / ds / name, c / name)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
