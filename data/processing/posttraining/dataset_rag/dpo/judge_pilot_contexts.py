#!/usr/bin/env python3
"""Check evidence sufficiency for every RAG pilot prompt before pairing.

The judge receives the exact retrieved context and gold answer.  A verdict without
verbatim evidence for an answerable context, or without a named missing fact for an
unanswerable context, is treated as ambiguous.  Output is resumable by pid.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dpo_common import _as_bool, _extract_json, chat_call, context_block, read_jsonl

SYSTEM = """You audit whether retrieved documents alone suffice to answer a question.
Use no external knowledge and do not assume the reference answer is supported.
For a bridge or comparison question, check every required fact and link.
Reply with only a JSON object with these fields:
{"answerable": true|false, "evidence": ["exact short quote from a document"],
 "missing": "specific missing fact or empty", "reason": "one short sentence"}.
When answerable=true, give quotes that jointly support the answer. When false,
name the indispensable missing fact and quote any partial evidence available.
If the documents are ambiguous, explain this in reason and use an empty evidence list."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default=os.environ.get("LLM_API_URL", "http://localhost:8000/v1"))
    ap.add_argument("--judge-model", default="Qwen3.8-27B")
    ap.add_argument("--judge-key-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=6000, help="raise for reasoning judges (reasoning counts)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-thinking", action="store_true", help="disable the judge's reasoning (Qwen chat_template_kwargs)")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    prompts = list(read_jsonl(args.prompts))
    done = {r["pid"] for r in read_jsonl(args.out)} if os.path.exists(args.out) else set()
    todo = [p for p in prompts if p["pid"] not in done]
    key = Path(args.judge_key_file).read_text().strip()
    url = args.judge_url.rstrip("/") + "/chat/completions"
    print(f"[context] {len(prompts)} prompts, {len(todo)} to judge", flush=True)

    def work(p):
        context = context_block(p["messages"][0]["content"])
        user = (f"Question:\n{p['question']}\n\nReference answer (may be unsupported):\n"
                f"{p['gold_answer']}\n\nRetrieved documents:\n{context}")
        raw = chat_call(url, key, args.judge_model, [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            max_tokens=args.max_tokens, timeout=args.timeout,
            thinking=False if args.no_thinking else None)
        out = {"pid": p["pid"], "qid": p["qid"], "context_sha256": __import__("hashlib").sha256(context.encode()).hexdigest(),
               "source_label": "U" if p["is_unanswerable"] else "A", "verdict": "ambiguous"}
        try:
            d = _extract_json(raw)
            answerable = _as_bool(d.get("answerable"))
            quotes = d.get("evidence")
            if not isinstance(quotes, list):
                quotes = []
            quotes = [q for q in quotes if isinstance(q, str) and q.strip()]
            valid = bool(quotes) and all(q in context for q in quotes)
            missing = str(d.get("missing") or "").strip()
            if answerable is True and valid:
                out["verdict"] = "answerable"
            elif answerable is False and missing:
                out["verdict"] = "unanswerable"
            out.update(answerable=answerable, evidence=quotes, evidence_in_context=valid,
                       missing=missing, reason=str(d.get("reason") or ""))
        except Exception:
            out["error"] = "judge_parse_failure"
        return out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, r in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(todo):
                f.flush()
                print(f"[context] {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
