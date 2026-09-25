#!/usr/bin/env python3
"""Judge pilot completions against documents, reasoning and cited passages."""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dpo_common import _as_bool, _extract_json, chat_call, citation_metrics, context_block, read_jsonl

SYSTEM = """Evaluate one RAG assistant response using ONLY the retrieved documents.
The supplied reference answer is a target, not evidence. Inspect every factual step.
Check whether each cited document passage actually supports the nearby claim; a
matching title by itself is insufficient. A refusal is good only when it describes
the missing information accurately. A response containing both an answer and a
refusal is mixed and cannot be preferred automatically.
Return only JSON with boolean fields:
{"refused":bool,"mixed":bool,"answer_correct":bool,"answer_supported":bool,
 "reasoning_supported":bool,"citation_supported":bool,"citation_style_ok":bool,
 "missing_fact_correct":bool,"support_quote":"exact short quote or empty",
 "reason":"short explanation"}.
For a refusal set answer_correct, answer_supported, reasoning_supported and
citation_supported to false. For an answer set missing_fact_correct to false.
If unsure, set the relevant boolean to false and explain why."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default=os.environ.get("LLM_API_URL", "http://localhost:8000/v1"))
    ap.add_argument("--judge-model", default="Qwen3.8-27B")
    ap.add_argument("--judge-key-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=6000, help="raise for reasoning judges (reasoning counts)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-thinking", action="store_true", help="disable the judge's reasoning (Qwen chat_template_kwargs)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--only", default=None, help="JSONL of {pid, k}: judge only these completions")
    args = ap.parse_args()
    prompts = {p["pid"]: p for p in read_jsonl(args.prompts)}
    contexts = {r["pid"]: r for r in read_jsonl(args.contexts)}
    samples = {}
    for r in read_jsonl(args.samples):
        if r.get("samples") and not r.get("errors"):
            samples[r["pid"]] = r
    only = {(r["pid"], r["k"]) for r in read_jsonl(args.only)} if args.only else None
    # a row whose judge call failed is judged again (readers keep the last row of a (pid, k))
    done = {(r["pid"], r["k"]) for r in read_jsonl(args.out)
            if r["judge"] is not None or r["finish_reason"] == "length" or not r["text"].strip()} \
        if os.path.exists(args.out) else set()
    jobs = [(prompts[pid], s) for pid, row in samples.items() if pid in prompts
            and contexts.get(pid, {}).get("verdict") == ("unanswerable" if prompts[pid]["is_unanswerable"] else "answerable")
            for s in row["samples"] if (pid, s["k"]) not in done and (only is None or (pid, s["k"]) in only)]
    key = Path(args.judge_key_file).read_text().strip()
    url = args.judge_url.rstrip("/") + "/chat/completions"
    print(f"[score] {len(samples)} sampled prompts, {len(jobs)} completions to judge", flush=True)

    def work(job):
        p, s = job
        body = context_block(p["messages"][0]["content"])
        text = s.get("text") or ""
        out = {"pid": p["pid"], "qid": p["qid"], "k": s["k"], "text": text,
               "finish_reason": s.get("finish_reason"), "n_chars": len(text),
               "judge": None, "context_verdict": contexts[p["pid"]]["verdict"],
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               **citation_metrics(text, p["gold_titles"], p["context_titles"], p["citation_style"])}
        if not text.strip() or s.get("finish_reason") == "length":
            return out
        user = (f"Question:\n{p['question']}\n\nReference answer:\n{p['gold_answer']}\n\n"
                f"Context sufficiency verdict:\n{contexts[p['pid']]['verdict']}\n\n"
                f"Retrieved documents:\n{body}\n\nAssistant response:\n{text}")
        raw = chat_call(url, key, args.judge_model, [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_tokens=args.max_tokens, timeout=args.timeout,
            thinking=False if args.no_thinking else None)
        try:
            d = _extract_json(raw)
            keys = ("refused", "mixed", "answer_correct", "answer_supported", "reasoning_supported",
                    "citation_supported", "citation_style_ok", "missing_fact_correct")
            flags = {k: _as_bool(d.get(k)) for k in keys}
            if all(v is not None for v in flags.values()):
                quote = str(d.get("support_quote") or "").strip()
                out["judge"] = {**flags, "support_quote": quote, "support_quote_in_context": bool(quote and quote in body),
                                "reason": str(d.get("reason") or "")}
        except Exception:
            pass
        if out["judge"] is None:  # why the verdict is missing (rejudged on the next run)
            out["judge_error"] = "no_response" if raw is None else "unparsable: " + str(raw)[-300:]
        return out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, r in enumerate(ex.map(work, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(jobs):
                f.flush()
                print(f"[score] {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
