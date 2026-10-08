#!/usr/bin/env python3
"""Configs hotpotqa_full_offpolicy_qwen35_35b_a3b / tatqa_full_offpolicy_qwen35_35b_a3b of OpenLLM-France/RAG_DPO_onpolicy
from dpo/data/jz_delta_v2 (Jean Zay run, 2026-10-06/07): chosen Qwen3.5-35B-A3B, rejected Qwen3.5-4B, both judged by
Qwen3.8-27B. Schema of the off-policy configs, plus rejected_source and chosen_mistral_judge.

Splits by unit (HotpotQA question, TAT-QA table), never a unit in train and validation across ALL configs of the repo:
units already in a validation split of the repo -> validation; units in a train split -> train; the others (never
published) are drawn by hash until validation holds ~10 % of the pairs. Pairs over 4,096 tokens (policy tokenizer and
chat template, as the other configs) are removed.
    venv/bin/python dpo/publish_jz_full_offpolicy.py --tokenizer <policy dir> --out <dir> [--revision bf40cf4]
        [--pairs-dir dpo/data/jz_delta_v2 --jz-run dpo/data/jz_run --crossjudge <judged.jsonl>]
--revision none: no split reference from the hub (validation drawn by hash among all units). Writes local parquets
only; uploading is a separate, manual step. Configs missing at the reference revision are skipped (the full off-policy
configs themselves exist from 0f77a34 on, so a later run keeps their validation units in validation).
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen_chosen_all as Q  # noqa: E402

D = HERE / "data"
REPO = "OpenLLM-France/RAG_DPO_onpolicy"
EXISTING = ["hotpotqa", "tatqa", "hotpotqa_offpolicy_qwen35_35b_a3b", "tatqa_offpolicy_qwen35_35b_a3b",
            "hotpotqa_offpolicy_luciole23b", "tatqa_offpolicy_luciole23b",
            "hotpotqa_full_offpolicy_qwen35_35b_a3b", "tatqa_full_offpolicy_qwen35_35b_a3b"]
VAL_SHARE = 0.10


def n_tokens(tokenizer, messages):
    """As dpo_train.n_tokens (not imported: it pulls trl); transformers 5 returns a BatchEncoding."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def unit(ds, qid):
    return qid.split("_", 1)[0] if ds == "tatqa" else qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="bf40cf4")
    ap.add_argument("--pairs-dir", default=str(D / "jz_delta_v2"))
    ap.add_argument("--jz-run", default=str(D / "jz_run"))
    ap.add_argument("--crossjudge", default=str(D / "crossjudge_mistral_jz_delta" / "judged.jsonl"),
                    help="judged.jsonl of crossjudge_mistral.py; none: no Mistral verdict (chosen_mistral_judge null)")
    ap.add_argument("--val-share", type=float, default=VAL_SHARE)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    seen = {"train": collections.defaultdict(set), "val": collections.defaultdict(set)}
    for cfg in (EXISTING if a.revision != "none" else []):
        ds = "tatqa" if cfg.startswith("tatqa") else "hotpotqa"
        for s in ("train", "val"):
            try:
                p = hf_hub_download(REPO, f"{cfg}/{s}.parquet", repo_type="dataset", revision=a.revision)
            except Exception as e:  # config absent at this revision
                if "404" not in str(e) and "EntryNotFound" not in type(e).__name__:
                    raise
                continue
            seen[s][ds].update(unit(ds, q) for q in pq.read_table(p, columns=["qid"]).column("qid").to_pylist())

    mistral = {}
    for r in (rows(a.crossjudge) if a.crossjudge != "none" else []):
        if r["group"] == "jz_chosen_35b" and r["mistral_judge"]:
            mistral[r["key"]] = r["mistral_judge"]

    out = Path(a.out)
    pairs_dir, run = Path(a.pairs_dir), Path(a.jz_run)
    man = {"source": str(pairs_dir), "source_sha256": json.loads((pairs_dir / "manifest.json").read_text())
           ["files"]["delta_pairs.jsonl"]["sha256"], "repo_revision_for_splits": a.revision, "configs": {}}
    pairs = list(rows(pairs_dir / "delta_pairs.jsonl"))
    for ds in ("hotpotqa", "tatqa"):
        st = collections.Counter()
        prompts, samples, verdict = {}, {}, {}
        for gen in ("qwen35_35b_a3b", "qwen35_4b"):
            Q.GEN = gen
            P, S, sc, ve, rj = Q.state(run / gen / ds)
            prompts.update(P)
            samples[gen] = S
            cit = Q.last_by_key(run / gen / ds / "scored_citfail.jsonl")
            verdict[gen] = lambda pid, k, P=P, sc=sc, ve=ve, rj=rj, cit=cit: (
                (rj.get((pid, k)) or {}).get("judge") if P[pid]["is_unanswerable"] else
                ((ve.get((pid, k)) or sc.get((pid, k)) or cit.get((pid, k)) or {}).get("judge")))
        recs = []
        for r in (r for r in pairs if r["dataset"] == ds):
            p = prompts[r["pid"]]
            n = max(n_tokens(tok, r["prompt"] + r["chosen"]), n_tokens(tok, r["prompt"] + r["rejected"]))
            if n > 4096:
                st["dropped_over_4096_tokens"] += 1
                continue
            mj = mistral.get(f"{ds}:{r['pid']}")
            recs.append({
                "prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"],
                "category": r["category"], "pair_type": "full_offpolicy",
                "chosen_source": "qwen35_35b_a3b", "rejected_source": "qwen35_4b",
                "refusal_instruction": r["refusal_instruction"], "pid": r["pid"], "qid": r["qid"],
                "hotpot_type": p.get("hotpot_type") if ds == "hotpotqa" else None,
                "answer_type": p.get("answer_type") if ds == "tatqa" else None,
                "answer_from": p.get("answer_from") if ds == "tatqa" else None,
                "unans_mode": p.get("unans_mode"),
                "chosen_k": r["chosen_k"], "rejected_k": r["rejected_k"],
                # chosen_k None: strict unanswerable prompt, the chosen is the imposed phrase itself (no sample, no judge)
                "chosen_judge": verdict["qwen35_35b_a3b"](r["pid"], r["chosen_k"]) if r["chosen_k"] is not None else None,
                "rejected_judge": verdict["qwen35_4b"](r["pid"], r["rejected_k"]),
                "chosen_mistral_judge": {k: mj[k] for k in ("good", "refused", "mixed", "missing_fact_correct",
                                                            "false_statement_about_documents", "reason")} if mj else None,
                "chosen_raw": samples["qwen35_35b_a3b"][r["pid"]][r["chosen_k"]]["text"] if r["chosen_k"] is not None
                else r["chosen"][0]["content"],
                "rejected_raw": samples["qwen35_4b"][r["pid"]][r["rejected_k"]]["text"],
                # strict prompts only (columns of the other configs); absent from the generic configs, unchanged
                **({"strict_phrase": r["strict_phrase"], "chosen_is_imposed_phrase": r["chosen_is_imposed_phrase"],
                    "rejected_is_exact_phrase": r["rejected_is_exact_phrase"]} if r.get("strict_phrase") else {}),
                "_n_tokens": n})
        # splits by unit
        by_unit = collections.defaultdict(list)
        for x in recs:
            by_unit[unit(ds, x["qid"])].append(x)
        split = {}
        for u in by_unit:
            if u in seen["val"][ds]:
                split[u] = "val"
            elif u in seen["train"][ds]:
                split[u] = "train"
        target = round(a.val_share * len(recs))
        n_val = sum(len(by_unit[u]) for u, s in split.items() if s == "val")
        fresh = sorted((u for u in by_unit if u not in split), key=lambda u: hashlib.sha256(f"jz42:{u}".encode()).hexdigest())
        for u in fresh:
            split[u] = "val" if n_val < target else "train"
            n_val += len(by_unit[u]) if split[u] == "val" else 0
        cfg = f"{ds}_full_offpolicy_qwen35_35b_a3b"
        (out / cfg).mkdir(parents=True, exist_ok=True)
        info = {"n": len(recs), **dict(st)}
        for s in ("train", "val"):
            rs = [{k: v for k, v in x.items() if not k.startswith("_")} for u in sorted(by_unit) if split[u] == s
                  for x in by_unit[u]]
            keys = list(dict.fromkeys(k for x in rs for k in x))  # generic and strict rows: same columns everywhere
            pq.write_table(pa.Table.from_pylist([{k: x.get(k) for k in keys} for x in rs]), out / cfg / f"{s}.parquet")
            info[s] = len(rs)
            info[f"{s}_units"] = sum(1 for u in split if split[u] == s)
            info[f"{s}_by_category"] = dict(sorted(collections.Counter(x["category"] for x in rs).items()))
        info["val_units_from_existing_val"] = sum(1 for u in split if u in seen["val"][ds])
        info["units_in_existing_train"] = sum(1 for u in split if u in seen["train"][ds])
        info["median_tokens"] = sorted(x["_n_tokens"] for x in recs)[len(recs) // 2]
        info["chosen_longer_chars"] = round(sum(len(x["chosen"][0]["content"]) > len(x["rejected"][0]["content"])
                                                for x in recs) / len(recs), 3)
        u_recs = [x for x in recs if x["category"].startswith("U_")]
        mj_u = [x["chosen_mistral_judge"] for x in u_recs if x["chosen_mistral_judge"]]
        info["chosen_mistral_good_U"] = f"{sum(j['good'] for j in mj_u)}/{len(mj_u)} judged of {len(u_recs)}"
        info["sha256"] = {s: hashlib.sha256((out / cfg / f"{s}.parquet").read_bytes()).hexdigest() for s in ("train", "val")}
        man["configs"][cfg] = info
    man["models"] = {"chosen": "Qwen/Qwen3.5-35B-A3B (bf16)", "rejected": "Qwen/Qwen3.5-4B (bf16)",
                     "judge": "Qwen/Qwen3.8-27B (bf16, reasoning)", "cross_judge": "Mistral-Small-3.2-24B-Instruct-2506-FP8"}
    (out / "manifest_full_offpolicy_qwen35_35b_a3b.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
