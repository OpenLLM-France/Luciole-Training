"""Augment HotpotQA with reasoning traces."""

import json
import asyncio
import aiohttp
import random
from datasets import load_dataset, Dataset
from pathlib import Path
from datetime import datetime
from utils import (
    LLM_API_URL,
    LLM_MAX_CONCURRENT,
    LLM_MODEL,
    Metrics,
    call_llm,
    format_context_chunks,
    load_checkpoint_ids,
    load_env_key,
    log,
    save_checkpoint_ids,
)

LLM_API_KEY = load_env_key("LLM_API_KEY", base_dir=Path(__file__).parent)
DATASET_CACHE_DIR = Path(__file__).parent / "hotpotqa_cache"
MAX_CONCURRENT = LLM_MAX_CONCURRENT


SYSTEM_PROMPT = """Answer the user question using only the provided context. Follow these rules:
- Provide step-by-step reasoning on how to answer the question.
- If you quote the context, wrap the quote in ##begin_quote## and ##end_quote## and add a citation marker ##Cite "title" ##.
- End your response with the final answer in this exact format: **Final Answer:** [your answer here]"""

SYSTEM_PROMPT_FR = """Répondez à la question de l'utilisateur en utilisant les informations fournies dans le contexte donné. Voici les points auxquels vous devez prêter attention :
- Fournissez un raisonnement étape par étape sur la manière de répondre à la question.
- Dans le raisonnement, si vous devez copier-coller certaines phrases du contexte, incluez-les entre ##begin_quote## et ##end_quote## et ajoutez une balise ##Cite "titre" ##.
- Terminez votre réponse par la réponse finale dans ce format exact : **Réponse finale :** [votre réponse ici]"""

UNANSWERABLE_REASONING_FR = "Les documents récupérés ne me permettent pas de répondre à votre question, pourriez vous la reformuler ou le cas échéant ajouter des documents dans votre partition"
UNANSWERABLE_REFUSAL_EN = (
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?"
)


def load_or_cache_dataset(dataset_path: str | None = None):
    if dataset_path:
        custom_path = Path(dataset_path)
        if custom_path.exists():
            log(f"Loading dataset from: {custom_path}")
            return Dataset.load_from_disk(str(custom_path))
        raise FileNotFoundError(f"Dataset not found: {custom_path}")

    if DATASET_CACHE_DIR.exists():
        log(f"Loading dataset from cache: {DATASET_CACHE_DIR}")
        return Dataset.load_from_disk(str(DATASET_CACHE_DIR))

    log("Downloading dataset from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds.save_to_disk(str(DATASET_CACHE_DIR))
    return ds


def create_unanswerable_row(row: dict, language: str = "en") -> dict | None:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    relevant_titles = set(row["supporting_facts"]["title"])

    new_titles = []
    new_sentences = []
    for title, sents in zip(titles, sentences):
        if title not in relevant_titles:
            new_titles.append(title)
            new_sentences.append(sents)

    if not new_titles:
        return None

    reasoning = {"en": UNANSWERABLE_REFUSAL_EN, "fr": UNANSWERABLE_REASONING_FR}.get(language, UNANSWERABLE_REFUSAL_EN)

    return {
        "id": f"{row['id']}_unanswerable",
        "question": row["question"],
        "answer": "",
        "type": row["type"],
        "level": row["level"],
        "context": {"title": new_titles, "sentences": new_sentences},
        "supporting_facts": {"title": [], "sent_id": []},
        "_is_unanswerable": True,
        "_unanswerable_reasoning": reasoning,
    }


def prepare_dataset_with_unanswerable(
    ds,
    unanswerable_ratio: float = 0.1,
    language: str = "en",
    limit: int | None = None,
) -> list:
    rows = []
    existing_unanswerable = []

    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        row_dict = dict(row)
        if row_dict.get("_is_unanswerable", False):
            existing_unanswerable.append(row_dict)
        else:
            rows.append(row_dict)

    total = len(rows) + len(existing_unanswerable)
    num_to_create = max(0, int(total * unanswerable_ratio) - len(existing_unanswerable))
    num_to_create = min(num_to_create, len(rows))

    new_unanswerable = []
    if num_to_create > 0:
        for idx in random.sample(range(len(rows)), num_to_create):
            u = create_unanswerable_row(rows[idx], language)
            if u:
                new_unanswerable.append(u)

    all_rows = rows + existing_unanswerable + new_unanswerable
    random.shuffle(all_rows)

    total_unanswerable = len(existing_unanswerable) + len(new_unanswerable)
    log(f"Dataset: {len(rows)} answerable + {total_unanswerable} unanswerable = {len(all_rows)} total")
    return all_rows


async def process_row(
    session: aiohttp.ClientSession,
    row: dict,
    semaphore: asyncio.Semaphore,
    language: str = "en",
    shuffle: bool = True,
    max_remove_background_ratio: float = 0.7,
) -> dict | None:
    is_unanswerable = row.get("_is_unanswerable", False)

    context_str, chunk_stats = format_context_chunks(
        context=row["context"],
        relevant_titles=set(row["supporting_facts"]["title"]),
        shuffle=shuffle,
        max_remove_background_ratio=max_remove_background_ratio,
        min_chunks=3,
    )

    if is_unanswerable:
        reasoning = row.get("_unanswerable_reasoning", UNANSWERABLE_REFUSAL_EN)
    else:
        system_prompt = {"en": SYSTEM_PROMPT, "fr": SYSTEM_PROMPT_FR}.get(language, SYSTEM_PROMPT)
        reasoning = await call_llm(
            session, row["question"], context_str, semaphore,
            api_key=LLM_API_KEY, system_prompt=system_prompt,
        )

    if reasoning is None:
        return None

    sf_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    return {
        "id": row["id"],
        "question": row["question"],
        "context": context_str,
        "answer": row["answer"],
        "reasoning_trace": reasoning,
        "type": row["type"],
        "level": row["level"],
        "is_unanswerable": is_unanswerable,
        "supporting_facts_titles": sf_titles,
        **chunk_stats,
    }


async def main(
    limit: int | None = None,
    concurrency: int = MAX_CONCURRENT,
    output_prefix: str = "hotpotqa_augmented",
    dataset_path: str | None = None,
    language: str = "en",
    unanswerable_ratio: float = 0.1,
    shuffle: bool = True,
    max_remove_background_ratio: float = 0.7,
    seed: int | None = None,
):
    if seed is not None:
        random.seed(seed)
        log(f"Random seed: {seed}")

    ds = load_or_cache_dataset(dataset_path)

    output_prefix = output_prefix.removesuffix(".jsonl")
    output_file = f"{output_prefix}.jsonl"
    checkpoint_file = f"{output_prefix}_checkpoint.json"

    rows_to_process = prepare_dataset_with_unanswerable(
        ds, unanswerable_ratio=unanswerable_ratio, language=language, limit=limit
    )

    processed_ids = load_checkpoint_ids(checkpoint_file)
    rows_to_process = [r for r in rows_to_process if r["id"] not in processed_ids]
    log(f"Rows to process: {len(rows_to_process)} (checkpoint: {len(processed_ids)} already done)")

    if not rows_to_process:
        log("Nothing to process.")
        return

    needs_llm = any(not r.get("_is_unanswerable", False) for r in rows_to_process)
    if needs_llm and not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required to process answerable rows (set in .env file or environment)")

    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metrics = Metrics(total_rows=len(rows_to_process))
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        with open(output_file, "a", encoding="utf-8") as out_f:
            batch_size = 100
            total_batches = (len(rows_to_process) + batch_size - 1) // batch_size

            for bidx, batch_start in enumerate(range(0, len(rows_to_process), batch_size), start=1):
                batch = rows_to_process[batch_start:batch_start + batch_size]
                tasks = [
                    process_row(
                        session, row, semaphore,
                        language=language,
                        shuffle=shuffle,
                        max_remove_background_ratio=max_remove_background_ratio,
                    )
                    for row in batch
                ]
                results = await asyncio.gather(*tasks)

                for row, result in zip(batch, results):
                    if result is None:
                        metrics.failed += 1
                        continue
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_ids.add(row["id"])
                    metrics.processed += 1
                    if result.get("is_unanswerable", False):
                        metrics.unanswerable_count += 1

                save_checkpoint_ids(processed_ids, checkpoint_file)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment HotpotQA with reasoning traces")

    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT, help=f"Number of parallel requests (default: {MAX_CONCURRENT})")
    parser.add_argument("--output", type=str, default="hotpotqa_augmented", help="Output file prefix (without extension)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to custom dataset cache directory")
    parser.add_argument("--language", type=str, choices=["en", "fr"], default="en", help="Language for prompts (default: en)")
    parser.add_argument("--unanswerable-ratio", type=float, default=0.1, help="Ratio of unanswerable examples (default: 0.1)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable chunk shuffling")
    parser.add_argument("--max-remove-background-ratio", type=float, default=0.7, help="Max fraction of background chunks to remove")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    asyncio.run(main(
        limit=args.limit,
        concurrency=args.concurrency,
        output_prefix=args.output,
        dataset_path=args.dataset,
        language=args.language,
        unanswerable_ratio=args.unanswerable_ratio,
        shuffle=not args.no_shuffle,
        max_remove_background_ratio=args.max_remove_background_ratio,
        seed=args.seed,
    ))
