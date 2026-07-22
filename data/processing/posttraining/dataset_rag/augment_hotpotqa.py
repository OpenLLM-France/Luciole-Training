"""Augment HotpotQA with reasoning traces."""

import json
import asyncio
import aiohttp
import random
from datasets import load_dataset, Dataset
from pathlib import Path
from datetime import datetime
from utils import (
    LLM_MAX_CONCURRENT,
    LLM_MODEL,
    Metrics,
    call_llm,
    contextualize_chunks,
    detect_refusal,
    format_context_chunks,
    load_checkpoint_ids,
    load_env_key,
    log,
    save_checkpoint_ids,
)
# Shared generation==inference prompt builder (single source of truth).
from formatting_sft import (
    DEFAULT_CITATION_STYLE,
    EOC_PROB,
    _generic_refusal_instruction,
    build_chat_system_prompt,
)

LLM_API_KEY = load_env_key("LLM_API_KEY", base_dir=Path(__file__).parent)
DATASET_CACHE_DIR = Path(__file__).parent / "hotpotqa_cache"
MAX_CONCURRENT = LLM_MAX_CONCURRENT


# Reasoning traces are generated with the shared OpenRAG prompt built by
# formatting_sft.build_chat_system_prompt, so the generation prompt matches the
# inference prompt (per-row citation placement, EN/FR persona). The canonical
# markup style is used at generation; markup variety is a formatting-only aug.

UNANSWERABLE_REASONING_FR = "Les documents récupérés ne me permettent pas de répondre à votre question, pourriez vous la reformuler ou le cas échéant ajouter des documents dans votre partition"
UNANSWERABLE_REFUSAL_EN = (
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?"
)


def load_or_cache_dataset(dataset_path: str | None = None):
    if dataset_path:
        custom_path = Path(dataset_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Dataset not found: {custom_path}")
        if custom_path.suffix == ".jsonl":
            log(f"Loading dataset from JSONL: {custom_path}")
            rows = []
            with open(custom_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        # Preserve unanswerable flags when loading heterogeneous JSONL rows.
                        row["_is_unanswerable"] = bool(row.get("_is_unanswerable", False))
                        if "_unanswerable_reasoning" in row:
                            row["_unanswerable_reasoning"] = row.get("_unanswerable_reasoning")
                        rows.append(row)
            all_keys = set().union(*(row.keys() for row in rows)) if rows else set()
            for row in rows:
                for key in all_keys:
                    row.setdefault(key, None)
            return Dataset.from_list(rows)
        log(f"Loading dataset from: {custom_path}")
        return Dataset.load_from_disk(str(custom_path))

    if DATASET_CACHE_DIR.exists():
        log(f"Loading dataset from cache: {DATASET_CACHE_DIR}")
        return Dataset.load_from_disk(str(DATASET_CACHE_DIR))

    log("Downloading dataset from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds.save_to_disk(str(DATASET_CACHE_DIR))
    return ds


def build_backfill_pool(rows: list[dict]) -> list[dict]:
    """Collect the non-gold paragraphs of every row, keyed by title.

    Dropping the supporting documents without replacing them shortens the
    context, so its length alone predicts the refusal. Since an unanswerable row
    is otherwise a clone of an answerable one, that shortcut separates the pair
    perfectly and spares the model from ever reading the documents. HotpotQA's
    own distractor paragraphs are the natural filler: real Wikipedia passages,
    about unrelated entities, with titles unique across the dataset.
    """
    pool: dict[str, list] = {}
    for row in rows:
        gold = set(row.get("supporting_facts", {}).get("title", []))
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in gold and title not in pool:
                pool[title] = sents
    return [{"title": t, "sentences": s} for t, s in pool.items()]


def _select_backfill(
    row: dict, pool: list[dict], *, present_titles: set[str], count: int
) -> list[dict]:
    """Draw ``count`` replacement paragraphs about other entities."""
    if count <= 0 or not pool:
        return []
    excluded = set(present_titles)
    picked: list[dict] = []
    # Rejection sampling: the pool holds every distractor of the whole dataset,
    # so scanning it per row would dominate a step that is otherwise LLM-bound.
    for _ in range(count * 50):
        if len(picked) == count:
            break
        candidate = random.choice(pool)
        if candidate["title"] in excluded:
            continue
        excluded.add(candidate["title"])
        picked.append(candidate)
    return picked


def create_unanswerable_row(
    row: dict,
    language: str = "en",
    mode: str = "drop_gold",
    backfill_pool: list[dict] | None = None,
) -> dict | None:
    """Turn an answerable row into an unanswerable one.

    ``drop_gold`` removes every supporting document (easy refusal: the topic
    entity may vanish from the context entirely). ``partial_hop`` removes a
    single supporting document of a multi-hop question, leaving genuine but
    insufficient evidence in context, the classic over-answering trap, and a
    much harder refusal signal. Returns None when the mode is not applicable.

    ``backfill_pool`` replaces the dropped documents one-for-one with
    paragraphs about unrelated entities, so the twin rows keep the same document
    count and roughly the same context length. The replacements are also flagged
    as protected background: supporting documents are never dropped by
    ``format_context_chunks``, so without protection the unanswerable row would
    expose a larger removable pool and end up shorter anyway.
    """
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    relevant_titles = set(row["supporting_facts"]["title"])

    if mode == "partial_hop":
        present_gold = [t for t in dict.fromkeys(titles) if t in relevant_titles]
        if len(present_gold) < 2:
            return None
        drop_titles = {random.choice(present_gold)}
    elif mode == "drop_gold":
        drop_titles = relevant_titles
    else:
        raise ValueError(f"Unknown unanswerable mode: {mode!r}")

    new_titles = []
    new_sentences = []
    for title, sents in zip(titles, sentences):
        if title not in drop_titles:
            new_titles.append(title)
            new_sentences.append(sents)

    if not new_titles or len(new_titles) == len(titles):
        return None

    replacements = _select_backfill(
        row,
        backfill_pool or [],
        present_titles=set(titles),
        count=len(titles) - len(new_titles),
    )
    for candidate in replacements:
        new_titles.append(candidate["title"])
        new_sentences.append(candidate["sentences"])

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
        "_unanswerable_mode": mode,
        # Stand in for the protection the supporting documents enjoyed, so both
        # twins expose the same number of removable background chunks. The gold
        # document a partial_hop row keeps is protected for the same reason, and
        # because dropping it at random would silently downgrade a hard refusal
        # with partial evidence into a plain drop_gold one.
        "_backfilled_titles": [c["title"] for c in replacements],
        "_protected_background_titles": [c["title"] for c in replacements]
        + [t for t in new_titles if t in relevant_titles],
        "_unanswerable_reasoning": reasoning,
    }


def prepare_dataset_with_unanswerable(
    ds,
    unanswerable_ratio: float = 0.1,
    language: str = "en",
    limit: int | None = None,
    partial_hop_ratio: float = 0.0,
    backfill: bool = True,
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
    partial_count = 0
    if num_to_create > 0:
        pool = build_backfill_pool(rows) if backfill else []
        for idx in random.sample(range(len(rows)), num_to_create):
            u = None
            if random.random() < partial_hop_ratio:
                u = create_unanswerable_row(rows[idx], language, mode="partial_hop", backfill_pool=pool)
                if u:
                    partial_count += 1
            if u is None:
                u = create_unanswerable_row(rows[idx], language, mode="drop_gold", backfill_pool=pool)
            if u:
                new_unanswerable.append(u)

    all_rows = rows + existing_unanswerable + new_unanswerable
    random.shuffle(all_rows)

    total_unanswerable = len(existing_unanswerable) + len(new_unanswerable)
    log(f"Dataset: {len(rows)} answerable + {total_unanswerable} unanswerable = {len(all_rows)} total")
    if partial_count:
        log(f"  ({partial_count} partial-hop unanswerable rows)")
    if new_unanswerable:
        backfilled = sum(len(u["_backfilled_titles"]) for u in new_unanswerable)
        log(f"  ({backfilled} documents backfilled into {len(new_unanswerable)} unanswerable rows)")
    return all_rows


async def process_row(
    session: aiohttp.ClientSession,
    row: dict,
    semaphore: asyncio.Semaphore,
    language: str = "en",
    shuffle: bool = True,
    max_remove_background_ratio: float = 0.7,
    contextualize: bool = False,
) -> dict | None:
    is_unanswerable = row.get("_is_unanswerable", False)

    # Optionally compute contextual summaries. Unanswerable rows are
    # contextualised too: skipping them would make the presence of a [CONTEXT]
    # block a perfect predictor of "answerable" and teach the model to never
    # refuse when the context is contextualised.
    summaries: dict[str, str] | None = None
    if contextualize:
        chunks = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            text = " ".join(sents).strip() if isinstance(sents, list) else str(sents).strip()
            chunks.append({"title": title, "text": text})
        summaries = await contextualize_chunks(
            session, chunks, semaphore,
            api_key=LLM_API_KEY, language=language,
        )

    context_str, chunk_stats = format_context_chunks(
        context=row["context"],
        relevant_titles=set(row["supporting_facts"]["title"]),
        shuffle=shuffle,
        max_remove_background_ratio=max_remove_background_ratio,
        min_chunks=3,
        protected_background_titles=set(row.get("_protected_background_titles", [])),
        contextual_summaries=summaries,
    )

    # Citation placement decided per row (deterministic by id) so the generation
    # prompt matches the inference prompt; formatting reuses this stored value.
    end_of_completion = random.Random(f"{row['id']}:eoc").random() < EOC_PROB
    # Answerable AND unanswerable rows are generated with the SAME OpenRAG prompt
    # (generic refusal instruction). On an insufficient context the model is
    # expected to refuse; the verification step keeps only the rows where it did.
    system_prompt = build_chat_system_prompt(
        context=context_str,
        language=language,
        style=DEFAULT_CITATION_STYLE,
        end_of_completion=end_of_completion,
        refusal_instruction=_generic_refusal_instruction(language),
    )
    reasoning = await call_llm(
        session, row["question"], context_str, semaphore,
        api_key=LLM_API_KEY, system_prompt=system_prompt,
        context_in_system=True,
    )
    # Whether the model actually refused (informational; the filter enforces it).
    grounded_refusal = bool(is_unanswerable and reasoning and detect_refusal(reasoning))

    if reasoning is None:
        return None

    title_mapping = chunk_stats.pop("title_mapping", {})
    sf_titles = list(dict.fromkeys(
        title_mapping.get(t, t)
        for t in row["supporting_facts"]["title"]
    ))
    return {
        "id": row["id"],
        "question": row["question"],
        "context": context_str,
        "answer": row["answer"],
        "reasoning_trace": reasoning,
        "type": row["type"],
        "level": row["level"],
        "is_unanswerable": is_unanswerable,
        "grounded_refusal": grounded_refusal,
        "citation_end_of_completion": end_of_completion,
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
    contextualize_ratio: float = 0.0,
    unanswerable_partial_hop_ratio: float = 0.0,
    unanswerable_backfill: bool = True,
):
    if seed is not None:
        random.seed(seed)
        log(f"Random seed: {seed}")

    if not 0.0 <= contextualize_ratio <= 1.0:
        raise ValueError("--contextualize-ratio must be between 0.0 and 1.0")
    if contextualize_ratio > 0:
        raise ValueError(
            "HotpotQA contextualization is disabled: its row context is a "
            "multi-document retrieval result, not the complete ordered source "
            "document required by contextual retrieval. Use 0.0 or run the "
            "source-backed TAT-QA pipeline."
        )

    ds = load_or_cache_dataset(dataset_path)

    output_prefix = output_prefix.removesuffix(".jsonl")
    output_file = f"{output_prefix}.jsonl"
    checkpoint_file = f"{output_prefix}_checkpoint.json"

    rows_to_process = prepare_dataset_with_unanswerable(
        ds, unanswerable_ratio=unanswerable_ratio, language=language, limit=limit,
        partial_hop_ratio=unanswerable_partial_hop_ratio,
        backfill=unanswerable_backfill,
    )

    processed_ids = load_checkpoint_ids(checkpoint_file)
    rows_to_process = [r for r in rows_to_process if r["id"] not in processed_ids]
    log(f"Rows to process: {len(rows_to_process)} (checkpoint: {len(processed_ids)} already done)")

    if not rows_to_process:
        log("Nothing to process.")
        return

    # Unanswerable rows now also call the LLM (to generate query-grounded refusals).
    needs_llm = bool(rows_to_process)
    if needs_llm and not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required to process rows (set in .env file or environment)")

    if contextualize_ratio > 0:
        log(f"Contextualize ratio: {contextualize_ratio:.0%}")

    ctx_flags = [random.random() < contextualize_ratio for _ in rows_to_process]

    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metrics = Metrics(total_rows=len(rows_to_process))
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        with open(output_file, "a", encoding="utf-8") as out_f:
            batch_size = 100
            total_batches = (len(rows_to_process) + batch_size - 1) // batch_size

            for bidx, batch_start in enumerate(range(0, len(rows_to_process), batch_size), start=1):
                batch = rows_to_process[batch_start:batch_start + batch_size]
                batch_flags = ctx_flags[batch_start:batch_start + batch_size]
                tasks = [
                    process_row(
                        session, row, semaphore,
                        language=language,
                        shuffle=shuffle,
                        max_remove_background_ratio=max_remove_background_ratio,
                        contextualize=flag,
                    )
                    for row, flag in zip(batch, batch_flags)
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
    parser.add_argument("--unanswerable-partial-hop-ratio", type=float, default=0.0,
                        help="Fraction of created unanswerable rows that drop only ONE supporting "
                             "document of a multi-hop question (hard refusal with partial evidence). "
                             "Falls back to drop_gold when not applicable (default: 0.0)")
    parser.add_argument("--no-unanswerable-backfill", action="store_true",
                        help="Do not replace the dropped supporting documents with paragraphs about "
                             "other entities. Leaves unanswerable rows shorter than their answerable "
                             "twin, which makes the context length predict the refusal "
                             "(default: backfill)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable chunk shuffling")
    parser.add_argument("--max-remove-background-ratio", type=float, default=0.7, help="Max fraction of background chunks to remove")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--contextualize-ratio", type=float, default=0.0,
                        help="Reserved; must remain 0 because complete HotpotQA source documents are unavailable")

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
        contextualize_ratio=args.contextualize_ratio,
        unanswerable_partial_hop_ratio=args.unanswerable_partial_hop_ratio,
        unanswerable_backfill=not args.no_unanswerable_backfill,
    ))
