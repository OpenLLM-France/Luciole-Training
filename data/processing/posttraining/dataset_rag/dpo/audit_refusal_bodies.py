#!/usr/bin/env python3
"""Audit of chosen refusals: does the refusal body assert facts the documents do not support?

build_pilot_pairs only checks `refused` and `missing_fact_correct` for an unanswerable chosen,
so a grounded refusal may still carry invented claims. One Qwen call (with reasoning) per pair.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dpo_common import _as_bool, _extract_json, chat_call, context_block, read_jsonl

SYSTEM = """You audit a refusal written by a RAG assistant, using ONLY the retrieved documents.
List every factual claim the response makes about the world (ignore statements about what the
documents do or do not contain). A claim is unsupported when no document states it.
Also check the statements about the documents: saying a document lacks some information when it
does contain it is a false statement.
Return only JSON: {"unsupported_claims": ["short claim", ...], "false_statement_about_documents": bool,
"reason": "one short sentence"}."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--categories", default="U_unsupported_answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-model", default="Qwen3.8-27B")
    ap.add_argument("--judge-key-file", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--pids", default=None, help="file with one pid per line: audit only these pairs")
    args = ap.parse_args()
    cats = set(args.categories.split(","))
    pairs = [p for f in args.pairs for p in read_jsonl(f) if p["category"] in cats]
    if args.pids:
        keep = set(Path(args.pids).read_text().split())
        pairs = [p for p in pairs if p["pid"] in keep]
    key = Path(args.judge_key_file).read_text().strip()
    url = args.judge_url.rstrip("/") + "/chat/completions"

    def work(p):
        user = (f"Question:\n{p['prompt'][-1]['content']}\n\nRetrieved documents:\n{context_block(p['prompt'][0]['content'])}"
                f"\n\nAssistant response:\n{p['chosen'][0]['content']}")
        raw = chat_call(url, key, args.judge_model, [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}], max_tokens=6000, timeout=900)
        out = {"pid": p["pid"], "category": p["category"], "unans_mode": p["unans_mode"], "judge": None}
        try:
            d = _extract_json(raw)
            out["judge"] = {"unsupported_claims": [c for c in d.get("unsupported_claims") or [] if isinstance(c, str)],
                            "false_statement_about_documents": _as_bool(d.get("false_statement_about_documents")),
                            "reason": str(d.get("reason") or "")}
        except Exception:
            pass
        return out

    with open(args.out, "w", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as ex:
        for r in ex.map(work, pairs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[audit] {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
