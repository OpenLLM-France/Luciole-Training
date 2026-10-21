#!/usr/bin/env python3
"""Refine built DPO pairs with the audit verdicts (v1 -> v1.1 of heldout_full_sft4152).

Drops, in this order:
  u_chosen_flawed_body          unanswerable pair whose chosen refusal was labelled in the manual
                                review (--u-review) with one of its drop_labels: invented fact or
                                false statement about the documents
  rejected_acceptable_on_recheck rejected response that the reasoning judge (--rejected-recheck,
                                score_pilot_samples.py output) now accepts (build_pilot_pairs.is_good)
  u_rejected_supported_on_recheck  optional (--drop-u-supported): unanswerable pair whose rejected answer
                                the reasoning judge now finds supported by the documents (answer_supported,
                                not refused): the context probably holds the answer. Added 2026-10-01 for the
                                strict and TatQA configs, whose U rejected were judged once without reasoning.
  excluded_<reason>             optional (--exclude-pids JSON {reason: [pid, ...]}): pairs excluded after a
                                manual review (e.g. a context found answerable)
  cap_u_correct_guess           optional (--cap-u-correct-guess N): keep at most N unanswerable
                                pairs whose rejected states the reference answer of the answerable
                                twin, chosen deterministically by pid hash

Kept pairs get `rejected_states_twin_gold` (unanswerable pairs) and, when rechecked,
`rejected_judge_recheck`. Needs the dataset_rag venv (utils.normalize_answer).

    venv/bin/python dpo/refine_pairs.py --pairs-dir dpo/data/heldout_full_sft4152/pairs \
        --prompts dpo/data/heldout_full_sft4152/prompts.jsonl \
        --rejected-recheck dpo/data/audit_heldout_sft4152/rejected_think.jsonl \
        --u-review dpo/data/audit_heldout_sft4152/u_chosen_manual_review.json \
        --out-dir dpo/data/heldout_full_sft4152/pairs_v1.1
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_pilot_pairs import fix_quote, is_good, rows, squash  # noqa: E402
from utils import normalize_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rejected-recheck", required=True)
    ap.add_argument("--u-review", default=None, help="manual review of the chosen refusal bodies (none: no drop)")
    ap.add_argument("--drop-u-supported", action="store_true")
    ap.add_argument("--exclude-pids", default=None, help="JSON {reason: [pid, ...]} from a manual review")
    ap.add_argument("--cap-u-correct-guess", type=int, default=-1, help="-1 = no cap")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    prompts = {p["pid"]: p for p in rows(args.prompts)}
    drop_u = set()
    if args.u_review:
        review = json.load(open(args.u_review))
        drop_u = {it["pid"] for it in review["items"] if it["label"] in review["drop_labels"]}
    exclude = {}
    if args.exclude_pids:
        for reason, pids in json.load(open(args.exclude_pids)).items():
            exclude.update({pid: reason for pid in pids})
    recheck = {}
    for r in rows(args.rejected_recheck):  # readers keep the last row of a (pid, k)
        if r["judge"] is not None:
            recheck[(r["pid"], r["k"])] = r["judge"]

    kept, excluded = [], collections.Counter()
    for split in ("train", "val"):
        for p in rows(Path(args.pairs_dir) / f"{split}.jsonl"):
            pr = prompts[p["pid"]]
            u = pr["is_unanswerable"]
            if p["pid"] in exclude:
                excluded[f"excluded_{exclude[p['pid']]}"] += 1
                continue
            if u and p["pid"] in drop_u:
                excluded["u_chosen_flawed_body"] += 1
                continue
            j = recheck.get((p["pid"], p["rejected_k"]))
            if j is not None:
                fix_quote(j, squash(pr["messages"][0]["content"]))
                # In U_strict_inexact / U_phrase_over_grounded the rejected IS a grounded refusal (it only
                # ignores the imposed phrase): a good refusal on recheck is expected, not an inversion.
                phrase_pair = p["category"] in ("U_strict_inexact", "U_phrase_over_grounded")
                if is_good(j, u) and not phrase_pair:
                    excluded["rejected_acceptable_on_recheck"] += 1
                    continue
                if u and args.drop_u_supported and j["answer_supported"] and not j["refused"]:
                    excluded["u_rejected_supported_on_recheck"] += 1
                    continue
                p["rejected_judge_recheck"] = j
            if u:
                # TatQA prompts carry the judge-facing gold "x (scale: ...)" and the raw one
                gold = normalize_answer(pr.get("gold_answer_raw", pr["gold_answer"]))
                p["rejected_states_twin_gold"] = bool(gold) and gold in normalize_answer(p["rejected"][0]["content"])
            kept.append(p)

    if args.cap_u_correct_guess >= 0:
        guess = sorted((p for p in kept if p.get("rejected_states_twin_gold")),
                       key=lambda p: hashlib.sha256(p["pid"].encode()).hexdigest())
        dropped = {id(p) for p in guess[args.cap_u_correct_guess:]}
        excluded["cap_u_correct_guess"] = len(dropped)
        kept = [p for p in kept if id(p) not in dropped]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for p in kept:
                if p["split"] == split:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    report = {"source": args.pairs_dir, "n_pairs": len(kept), "excluded": dict(excluded),
              "category": dict(collections.Counter(p["category"] for p in kept)),
              "split": dict(collections.Counter(p["split"] for p in kept)),
              "rejected_rechecked": sum("rejected_judge_recheck" in p for p in kept),
              "u_rejected_states_twin_gold": sum(bool(p.get("rejected_states_twin_gold")) for p in kept),
              "cap_u_correct_guess": args.cap_u_correct_guess, "drop_u_supported": args.drop_u_supported,
              "u_review": args.u_review, "exclude_pids": args.exclude_pids}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
