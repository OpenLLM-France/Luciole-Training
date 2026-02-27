#!/usr/bin/env python3
"""
Augment converted TATQA rows with reasoning traces.
"""

import argparse
import asyncio
import json
import random
import warnings
from collections.abc import Coroutine
from pathlib import Path
from datetime import datetime

import aiohttp
from utils import (
    LLM_API_URL,
    LLM_MAX_CONCURRENT,
    LLM_MODEL,
    Metrics,
    call_llm,
    format_context_chunks,
    jsonl_read,
    load_checkpoint_ids,
    load_env_key,
    log,
    save_checkpoint_ids,
)

LLM_API_KEY = load_env_key("LLM_API_KEY", base_dir=Path(__file__).parent)

SYSTEM_PROMPT = """Answer the user question using only the provided context (table and paragraphs).
- Provide step-by-step reasoning.
- For each factual statement copied from the context, wrap the quote in ##begin_quote## ... ##end_quote## and cite the source as ##Cite "Table" ## or ##Cite "Paragraph N" ##.
- For arithmetic/count questions, show the computation steps explicitly.
- End in this exact format: **Final Answer:** [your answer here]"""

UNANSWERABLE_REFUSAL = (
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?"
)


def create_unanswerable_row(row: dict) -> dict | None:
    context = row["context"]
    titles = context["title"]
    sentences = context["sentences"]
    relevant = set(row.get("supporting_facts", {}).get("title", []))
    answer_from = row.get("answer_from", "")

    keep_titles = []
    keep_sentences = []
    for title, sents in zip(titles, sentences):
        should_keep = title not in relevant
        if answer_from == "table" and title == "Table":
            should_keep = False
        if should_keep:
            keep_titles.append(title)
            keep_sentences.append(sents)

    if len(keep_titles) < 2:
        return None

    return {
        **row,
        "id": f"{row['id']}_unanswerable",
        "answer": "",
        "answer_raw": [],
        "supporting_facts": {"title": [], "sent_id": []},
        "_is_unanswerable": True,
        "_unanswerable_reasoning": UNANSWERABLE_REFUSAL,
        "context": {"title": keep_titles, "sentences": keep_sentences},
    }


def prepare_rows_with_unanswerable(
    rows: list[dict], unanswerable_ratio: float
) -> list[dict]:
    answerable = []
    existing_unanswerable = []
    for row in rows:
        if row.get("_is_unanswerable", False):
            existing_unanswerable.append(row)
        else:
            answerable.append(row)

    total = len(rows)
    target = int(total * unanswerable_ratio)
    need_new = max(0, target - len(existing_unanswerable))

    new_unanswerable = []
    if need_new > 0 and answerable:
        need_new = min(need_new, len(answerable))
        indices = random.sample(range(len(answerable)), need_new)
        for i in indices:
            u = create_unanswerable_row(answerable[i])
            if u is not None:
                new_unanswerable.append(u)

    all_rows = answerable + existing_unanswerable + new_unanswerable
    random.shuffle(all_rows)
    return all_rows


async def process_row(
    session: aiohttp.ClientSession,
    row: dict,
    semaphore: asyncio.Semaphore,
    shuffle: bool,
    max_remove_background_ratio: float,
    max_retries: int,
    request_timeout: int,
    max_tokens: int,
    debug_http: bool,
) -> dict | None:
    is_unanswerable = row.get("_is_unanswerable", False)
    context_str, chunk_stats = format_context_chunks(
        context=row["context"],
        relevant_titles=set(row.get("supporting_facts", {"title": [], "sent_id": []}).get("title", [])),
        shuffle=shuffle,
        max_remove_background_ratio=max_remove_background_ratio,
        min_chunks=2,
        protected_background_titles={"Table"},
    )

    if is_unanswerable:
        reasoning = row.get("_unanswerable_reasoning", UNANSWERABLE_REFUSAL)
    else:
        reasoning = await call_llm(
            session,
            row["question"],
            context_str,
            semaphore,
            api_key=LLM_API_KEY,
            system_prompt=SYSTEM_PROMPT,
            max_retries=max_retries,
            timeout_seconds=request_timeout,
            max_tokens=max_tokens,
            debug_http=debug_http,
        )
        if reasoning is None:
            log(f"Row failed: id={row.get('id')}")
            return None

    sf_titles = list(dict.fromkeys(row.get("supporting_facts", {}).get("title", [])))
    return {
        "id": row["id"],
        "question": row["question"],
        "context": context_str,
        "answer": row.get("answer", ""),
        "answer_raw": row.get("answer_raw", []),
        "answer_type": row.get("answer_type", ""),
        "answer_from": row.get("answer_from", ""),
        "scale": row.get("scale", ""),
        "reasoning_trace": reasoning,
        "is_unanswerable": is_unanswerable,
        "supporting_facts_titles": sf_titles,
        "type": row.get("type", row.get("answer_type", "unknown")),
        "level": row.get("level", "tatqa"),
        **chunk_stats,
    }


async def run(
    dataset_path: str,
    output_prefix: str,
    limit: int | None,
    concurrency: int,
    unanswerable_ratio: float,
    shuffle: bool,
    max_remove_background_ratio: float,
    seed: int | None,
    max_retries: int,
    request_timeout: int,
    max_tokens: int,
    debug_http: bool,
) -> None:
    if seed is not None:
        random.seed(seed)
        log(f"Random seed: {seed}")

    output_prefix = output_prefix.removesuffix(".jsonl")
    output_file = f"{output_prefix}.jsonl"
    checkpoint_file = f"{output_prefix}_checkpoint.json"

    rows = jsonl_read(dataset_path, limit=limit)
    rows = prepare_rows_with_unanswerable(rows, unanswerable_ratio=unanswerable_ratio)

    processed = load_checkpoint_ids(checkpoint_file)
    rows = [r for r in rows if r["id"] not in processed]

    log(f"Rows to process: {len(rows)}")
    if not rows:
        log("Nothing to process.")
        return

    needs_llm = any(not r.get("_is_unanswerable", False) for r in rows)
    if needs_llm and not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required to process answerable rows (set in .env file or environment)")

    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    semaphore = asyncio.Semaphore(concurrency)
    metrics = Metrics(total_rows=len(rows))

    connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        with open(output_file, "a", encoding="utf-8") as out_f:
            batch_size = 100
            total_batches = (len(rows) + batch_size - 1) // batch_size
            for bidx, start in enumerate(range(0, len(rows), batch_size), start=1):
                batch = rows[start:start + batch_size]
                tasks = [
                    process_row(
                        session, row, semaphore,
                        shuffle=shuffle,
                        max_remove_background_ratio=max_remove_background_ratio,
                        max_retries=max_retries,
                        request_timeout=request_timeout,
                        max_tokens=max_tokens,
                        debug_http=debug_http,
                    )
                    for row in batch
                ]
                results = await asyncio.gather(*tasks)
                for row, result in zip(batch, results):
                    if result is None:
                        metrics.failed += 1
                        continue
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed.add(row["id"])
                    metrics.processed += 1
                    if result.get("is_unanswerable", False):
                        metrics.unanswerable_count += 1
                save_checkpoint_ids(processed, checkpoint_file, sort_ids=True, indent=2)
                out_f.flush()
                llm_calls = metrics.processed - metrics.unanswerable_count
                log(
                    f"Batch {bidx}/{total_batches} — "
                    f"Progress {metrics.done}/{metrics.total_rows} ({metrics.progress_pct:.1f}%) "
                    f"| processed={metrics.processed} failed={metrics.failed} "
                    f"| unanswerable={metrics.unanswerable_count} llm_calls={llm_calls} "
                    f"| speed={metrics.speed:.2f} rows/s"
                )

    log(f"Done in {metrics.elapsed:.1f}s — {metrics.processed} processed, {metrics.failed} failed, {metrics.unanswerable_count} unanswerable")


def run_async(coro: Coroutine[object, object, object]) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
        loop.run_until_complete(loop.shutdown_asyncgens())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loop.run_until_complete(loop.shutdown_default_executor(timeout=2.0))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment TATQA converted JSONL with reasoning traces")
    parser.add_argument("--dataset", required=True, help="Converted TATQA JSONL path")
    parser.add_argument("--output", default="tatqa_augmented", help="Output prefix (default: tatqa_augmented)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows before augmentation")
    parser.add_argument("--concurrency", type=int, default=LLM_MAX_CONCURRENT, help="Async concurrency")
    parser.add_argument("--unanswerable-ratio", type=float, default=0.1, help="Target unanswerable ratio")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable chunk shuffling")
    parser.add_argument("--max-retries", type=int, default=3, help="LLM call retries per row")
    parser.add_argument("--request-timeout", type=int, default=120, help="HTTP timeout per LLM call (seconds)")
    parser.add_argument("--max-tokens", type=int, default=1200, help="LLM max_tokens")
    parser.add_argument("--debug-http", action="store_true", help="Verbose LLM failure diagnostics")
    parser.add_argument(
        "--max-remove-background-ratio",
        type=float,
        default=0.7,
        help="Max fraction of background chunks to remove",
    )
    args = parser.parse_args()

    run_async(
        run(
            dataset_path=args.dataset,
            output_prefix=args.output,
            limit=args.limit,
            concurrency=args.concurrency,
            unanswerable_ratio=args.unanswerable_ratio,
            shuffle=not args.no_shuffle,
            max_remove_background_ratio=args.max_remove_background_ratio,
            seed=args.seed,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            max_tokens=args.max_tokens,
            debug_http=args.debug_http,
        )
    )


if __name__ == "__main__":
    main()
