#!/usr/bin/env python3
"""On-policy sampling of the SFT policy through a vLLM OpenAI endpoint.

For each prompt: K samples at temperature T (one request with n=K, the prompt prefix
is shared in the KV cache) + optionally one greedy completion (k = -1), which is the
behaviour lighteval measures.

Output: one JSON line per prompt, appended and flushed -> resumable (prompts already
present in --out are skipped). Long timeouts on purpose: the thermal pauser may
freeze the vLLM container for minutes.

    python dpo/sample_policy.py --prompts dpo/data/prompts_hpv4.jsonl \
        --out dpo/data/samples_hpv4.jsonl --model luciole_policy --k 4
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp


async def post(session, url, payload, retries=6):
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload) as r:
                if r.status == 200:
                    return await r.json(), None
                txt = await r.text()
                if r.status == 400:  # context too long / bad request: do not retry
                    return None, f"400: {txt[:300]}"
                err = f"{r.status}: {txt[:200]}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(min(60, 5 * (attempt + 1)))
    return None, err


async def sample_one(session, sem, args, rec):
    url = args.api_url.rstrip("/") + "/chat/completions"
    base = {"model": args.model, "messages": rec["messages"], "max_tokens": args.max_tokens}
    if args.chat_template_kwargs:  # e.g. {"enable_thinking": false} for Qwen3
        base["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    out = {"pid": rec["pid"], "samples": [], "errors": []}
    async with sem:
        reqs = []
        if args.k > 0:
            extra = {k: v for k, v in (("top_k", args.top_k), ("min_p", args.min_p)) if v is not None}
            reqs.append(("sampled", {**base, "n": args.k, "temperature": args.temperature,
                                     "top_p": args.top_p, "seed": args.seed, **extra}))
        if args.greedy:
            reqs.append(("greedy", {**base, "n": 1, "temperature": 0.0}))
        results = await asyncio.gather(*(post(session, url, p) for _, p in reqs))
    for (kind, _), (res, err) in zip(reqs, results):
        if res is None:
            out["errors"].append({"kind": kind, "error": err})
            continue
        for i, ch in enumerate(res["choices"]):
            out["samples"].append({
                "k": -1 if kind == "greedy" else i,
                "text": ch["message"]["content"] or "",
                "finish_reason": ch.get("finish_reason"),
            })
        out.setdefault("usage", {})[kind] = res.get("usage")
    return out


async def main_async(args):
    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    done = set()
    try:
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["samples"] and not d["errors"]:
                    done.add(d["pid"])
    except FileNotFoundError:
        pass
    todo = [p for p in prompts if p["pid"] not in done]
    print(f"[sample] {len(prompts)} prompts, {len(done)} already done, {len(todo)} to go", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    t0, n_ok, n_err, gen_tokens = time.time(), 0, 0, 0
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if os.environ.get(args.api_key_env) else {}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        with open(args.out, "a", encoding="utf-8") as f:
            tasks = [asyncio.create_task(sample_one(session, sem, args, p)) for p in todo]
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                res = await fut
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += not res["errors"]
                n_err += bool(res["errors"])
                gen_tokens += sum((u or {}).get("completion_tokens", 0) for u in res.get("usage", {}).values())
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"[sample] {i}/{len(todo)} ok={n_ok} err={n_err} "
                          f"{i / el:.2f} prompts/s {gen_tokens / el:.0f} gen tok/s "
                          f"ETA {(len(todo) - i) / max(i / el, 1e-9) / 60:.0f} min", flush=True)
    print(f"[sample] done in {(time.time() - t0) / 60:.1f} min, errors={n_err} "
          "(rerun the same command to retry failed prompts)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--api-key-env", default="SAMPLER_API_KEY",
                    help="env var holding a bearer key (unset for the local vLLM)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None, help="vLLM extra sampling parameter (unset = server default)")
    ap.add_argument("--min-p", type=float, default=None, help="vLLM extra sampling parameter (unset = server default)")
    ap.add_argument("--no-greedy", dest="greedy", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chat-template-kwargs", default=None, help="JSON passed as chat_template_kwargs")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
