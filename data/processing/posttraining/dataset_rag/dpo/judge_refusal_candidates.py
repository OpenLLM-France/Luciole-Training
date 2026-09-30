#!/usr/bin/env python3
"""Judge candidate CHOSEN refusals (from any generator) on unanswerable prompts, in one call.

One Qwen call with reasoning per completion checks at once what build_pilot_pairs needs
(refused, mixed, missing_fact_correct) and what it misses (audit_refusal_bodies): claims
the documents do not support, and false statements about what the documents contain.
A completion is a usable chosen when it refuses, is not mixed, names the missing fact
correctly, asserts nothing unsupported and says nothing false about the documents.

    python dpo/judge_refusal_candidates.py --prompts D/prompts.jsonl --samples D/chosen_x.jsonl \
        --contexts D/context_verdicts.jsonl --out D/judged_chosen_x.jsonl --judge-key-file .judge_key --only-greedy
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dpo_common import _as_bool, _extract_json, chat_call, context_block, read_jsonl

SYSTEM = """You audit the response of a RAG assistant to a question that the retrieved documents
do NOT allow to answer. Use ONLY the retrieved documents; the reference answer is given only
to recognise it, it is not evidence.
Decide:
- refused: the response declines to give an answer to the question.
- mixed: the response both declines and still gives (or guesses) an answer.
- missing_fact_correct: the response correctly identifies what information is missing.
- unsupported_claims: every factual claim about the world that no document states (ignore
  statements about what the documents do or do not contain, and mere restatements of the
  question's own premises).
- false_statement_about_documents: the response says a document lacks information that it
  does contain, or misdescribes what the documents contain.
Return only JSON: {"refused":bool,"mixed":bool,"missing_fact_correct":bool,
"unsupported_claims":["short claim",...],"false_statement_about_documents":bool,
"reason":"one short sentence"}."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--samples", required=True, help="sample_policy.py output")
    ap.add_argument("--contexts", required=True, help="only prompts judged unanswerable are audited")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-greedy", action="store_true")
    ap.add_argument("--ks", default=None, help="comma-separated sample indices to judge (e.g. -1 or 0,1)")
    ap.add_argument("--pids", default=None, help="file with one pid per line: judge only these prompts")
    ap.add_argument("--judge-url", default=os.environ.get("LLM_API_URL", "http://localhost:8000/v1"))
    ap.add_argument("--judge-model", default="Qwen3.8-27B")
    ap.add_argument("--judge-key-file", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=6000, help="reasoning counts: raise if verdicts come back empty")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    prompts = {p["pid"]: p for p in read_jsonl(args.prompts)}
    unans = {r["pid"] for r in read_jsonl(args.contexts) if r.get("verdict") == "unanswerable"}
    ks = {int(k) for k in args.ks.split(",")} if args.ks else None
    if args.pids:
        keep = set(Path(args.pids).read_text().split())
        unans &= keep
    done = {(r["pid"], r["k"]) for r in read_jsonl(args.out) if r["judge"] is not None} if os.path.exists(args.out) else set()
    jobs = [(prompts[r["pid"]], s) for r in read_jsonl(args.samples) if r["pid"] in unans and r["pid"] in prompts
            for s in r["samples"] if (not args.only_greedy or s["k"] == -1) and (ks is None or s["k"] in ks)
            and (r["pid"], s["k"]) not in done]
    key = Path(args.judge_key_file).read_text().strip()
    url = args.judge_url.rstrip("/") + "/chat/completions"
    print(f"[refusal-judge] {len(jobs)} completions to judge", flush=True)

    def work(job):
        p, s = job
        text = s.get("text") or ""
        out = {"pid": p["pid"], "k": s["k"], "text": text, "finish_reason": s.get("finish_reason"),
               "n_chars": len(text), "judge": None}
        if not text.strip() or s.get("finish_reason") == "length":
            return out
        user = (f"Question:\n{p['question']}\n\nReference answer (NOT supported by the documents):\n{p['gold_answer']}"
                f"\n\nRetrieved documents:\n{context_block(p['messages'][0]['content'])}\n\nAssistant response:\n{text}")
        raw = chat_call(url, key, args.judge_model, [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}], max_tokens=args.max_tokens, timeout=args.timeout)
        try:
            d = _extract_json(raw)
            claims = [c for c in d.get("unsupported_claims") or [] if isinstance(c, str) and c.strip()]
            j = {"refused": _as_bool(d.get("refused")), "mixed": _as_bool(d.get("mixed")),
                 "missing_fact_correct": _as_bool(d.get("missing_fact_correct")),
                 "unsupported_claims": claims,
                 "false_statement_about_documents": _as_bool(d.get("false_statement_about_documents")),
                 "reason": str(d.get("reason") or "")}
            j["good_chosen"] = bool(j["refused"] and not j["mixed"] and j["missing_fact_correct"]
                                    and not claims and not j["false_statement_about_documents"])
            out["judge"] = j
        except Exception:
            pass
        if out["judge"] is None:  # why the verdict is missing (rejudged on the next run)
            out["judge_error"] = "no_response" if raw is None else "unparsable: " + str(raw)[-300:]
        return out

    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[refusal-judge] {i}/{len(jobs)}", flush=True)
    print(f"[refusal-judge] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
