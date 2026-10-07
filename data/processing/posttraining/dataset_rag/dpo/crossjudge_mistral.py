#!/usr/bin/env python3
"""Mistral second opinion on the refusals of the Delta pairs (column chosen_mistral_judge of the release).

The chosen refusal of every unanswerable (U) pair, written by Qwen3.5-35B-A3B and accepted by the Qwen3.8-27B judge,
is judged again by a model of another family (Mistral); with JZ_CROSSJUDGE_REJECTED=1 (default) the Qwen3.5-4B
rejected refusal of every U_bad_refusal pair is judged too (an audit: Mistral "good" = the pair's contrast is
doubtful). Informational: no pair is removed on it. Same refusal judge prompt as the pipeline
(judge_refusal_candidates). Each distinct response is judged once (resumable: rows already in --out are skipped).

env (Airflow run dirs): JZ_RUN (prompts, default dpo/data/jz_run), JZ_PAIRS (default dpo/data/jz_delta/delta_pairs.jsonl).
    venv/bin/python dpo/crossjudge_mistral.py run --out dpo/data/crossjudge_mistral_jz_delta/judged.jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dpo_common import _as_bool, _extract_json, chat_call, context_block, read_jsonl  # noqa: E402
from judge_refusal_candidates import SYSTEM as REFUSAL_SYSTEM  # noqa: E402

D = HERE / "data"


def text_of(m):
    return m[-1]["content"] if isinstance(m, list) else m


def _item(group, key, r, chosen, kind, p, verdict):
    chosen = chosen.strip()
    return {"group": group, "key": key, "sha": hashlib.sha256(chosen.encode()).hexdigest()[:16],
            "category": r["category"], "refusal_instruction": r.get("refusal_instruction"), "split": r.get("split"),
            "len_chosen": len(chosen), "kind": kind, "qwen_judge": r.get("chosen_judge"),
            "_p": p, "_verdict": verdict, "_text": chosen}


def items_jz_delta():
    """Delta pairs: chosen of the U pairs, then rejected of the U_bad_refusal pairs (both refusals).
    JZ_CROSSJUDGE_REJECTED=0 to judge the chosen only (the rejected verdicts are an audit, not used downstream)."""
    run = Path(os.environ.get("JZ_RUN", D / "jz_run"))
    prompts = {r["pid"]: r for ds in ("hotpotqa", "tatqa")
               for r in read_jsonl(run / "qwen35_35b_a3b" / ds / "prompts.jsonl")}
    pairs = [r for r in read_jsonl(os.environ.get("JZ_PAIRS", D / "jz_delta" / "delta_pairs.jsonl"))
             if r["category"].startswith("U_")]
    for r in pairs:
        yield _item("jz_chosen_35b", f"{r['dataset']}:{r['pid']}", {**r, "split": r["dataset"]}, text_of(r["chosen"]),
                    "refusal", prompts[r["pid"]], "unanswerable")
    for r in pairs:
        if r["category"] == "U_bad_refusal" and os.environ.get("JZ_CROSSJUDGE_REJECTED", "1") == "1":
            yield _item("jz_rejected_4b", f"{r['dataset']}:{r['pid']}", {**r, "split": r["dataset"]},
                        text_of(r["rejected"]), "refusal", prompts[r["pid"]], "unanswerable")


SETS = {"jz_delta": items_jz_delta}


def items(source="jz_delta"):
    """One judging job per distinct (group, key, chosen)."""
    seen, out = set(), []
    for it in SETS[source]():
        if (it["group"], it["key"], it["sha"]) not in seen:
            seen.add((it["group"], it["key"], it["sha"]))
            out.append(it)
    return out


def judge(it, url, key, model, max_tokens):
    p, text = it["_p"], it["_text"]
    body = context_block(p["messages"][0]["content"])
    user = (f"Question:\n{p['question']}\n\nReference answer (NOT supported by the documents):\n{p['gold_answer']}"
            f"\n\nRetrieved documents:\n{body}\n\nAssistant response:\n{text}")
    raw = chat_call(url, key, model, [{"role": "system", "content": REFUSAL_SYSTEM}, {"role": "user", "content": user}],
                    max_tokens=max_tokens, timeout=600)
    j = None
    try:
        d = _extract_json(raw)
        claims = [c for c in d.get("unsupported_claims") or [] if isinstance(c, str) and c.strip()]
        j = {"refused": _as_bool(d.get("refused")), "mixed": _as_bool(d.get("mixed")),
             "missing_fact_correct": _as_bool(d.get("missing_fact_correct")), "unsupported_claims": claims,
             "false_statement_about_documents": _as_bool(d.get("false_statement_about_documents")),
             "reason": str(d.get("reason") or "")}
        j["good"] = j["good_noquote"] = bool(j["refused"] and not j["mixed"] and j["missing_fact_correct"]
                                             and not claims and not j["false_statement_about_documents"])
    except Exception:
        pass
    row = {k: v for k, v in it.items() if not k.startswith("_")}
    row.update(judge_model=model, mistral_judge=j, raw=None if j else raw)
    return row


def cmd_run(args):
    its = items(args.set)
    done = set()
    if os.path.exists(args.out):
        done = {(r["group"], r["key"], r["sha"]) for r in read_jsonl(args.out) if r["mistral_judge"] is not None}
    jobs = [it for it in its if (it["group"], it["key"], it["sha"]) not in done]
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    print(f"[crossjudge] {len(its)} distinct chosen, {len(jobs)} to judge "
          f"{dict(collections.Counter(it['group'] for it in jobs))}", flush=True)
    key = (Path(args.key_file).read_text().strip() if args.key_file else os.environ.get("LLM_API_KEY", ""))
    url = args.url.rstrip("/") + "/chat/completions"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as ex:
        for i, r in enumerate(ex.map(lambda it: judge(it, url, key, args.model, args.max_tokens), jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(jobs):
                print(f"[crossjudge] {i}/{len(jobs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=os.environ.get("LLM_API_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default="Mistral-Small-3.2-24B-Instruct-2506-FP8")
    ap.add_argument("--key-file", default=None, help="else env LLM_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=10, help="Mistral: 12 concurrent calls max, all jobs together")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", choices=tuple(SETS), default="jz_delta")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
