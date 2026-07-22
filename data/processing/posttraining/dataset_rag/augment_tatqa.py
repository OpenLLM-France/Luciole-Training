#!/usr/bin/env python3
"""
Augment converted TATQA rows with reasoning traces.
"""

import argparse
import asyncio
import hashlib
import json
import random
import warnings
from collections.abc import Coroutine
from pathlib import Path
from datetime import datetime

import aiohttp
from convert_tatqa_to_augment_format import load_tatqa_docs, tatqa_document_chunks
from distractor_generation import LEAK_MIN_LENGTH, content_hash, normalize_for_leak_check
from utils import (
    CONTEXTUALIZER_SYSTEM_PROMPT,
    LLM_MAX_CONCURRENT,
    LLM_MODEL,
    Metrics,
    call_llm,
    contextualize_chunks,
    detect_refusal,
    format_context_chunks,
    jsonl_read,
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

# Reasoning traces are generated with the shared OpenRAG prompt built by
# formatting_sft.build_chat_system_prompt, so the generation prompt matches the
# inference prompt (per-row citation placement, canonical markup). TATQA-specific
# guidance (tables, explicit arithmetic) is dropped in favour of that shared
# prompt; the verification/filter step keeps only the traces that still answered
# correctly.

UNANSWERABLE_REFUSAL = (
    "The retrieved documents do not allow me to answer your question. Could you rephrase it or add relevant documents?"
)


def _answer_leak_strings(row: dict) -> list[str]:
    """Normalized answer strings a backfilled chunk must not contain.

    Mirrors the converter: only textual answers are checked. Arithmetic and
    count answers are derived numbers whose value may legitimately appear in an
    unrelated report, and checking them would reject most of the pool.
    """
    if row.get("answer_type") not in {"span", "multi-span"}:
        return []
    raw = row.get("answer_raw") or row.get("answer") or ""
    values = raw if isinstance(raw, list) else [raw]
    normalized = [normalize_for_leak_check(v) for v in values if v]
    return [n for n in normalized if len(n) >= LEAK_MIN_LENGTH]


def _chunk_kind(title: str) -> str:
    return "table" if str(title).startswith("Table") else "paragraph"


def build_backfill_pool(rows: list[dict]) -> dict[str, list[dict]]:
    """Collect every cross-doc distractor present in the converted rows.

    Dropping the gold chunks without replacing them shortens the context, and
    the length alone then predicts the refusal, the same class of leak as a
    [CONTEXT] block that only ever appears on answerable rows, and a worse one
    here because the unanswerable row is otherwise identical to its twin. The
    distractors the converter already attached are the natural filler: real
    chunks, from other reports, with globally unique content-hash titles.

    The pool is split by kind: dropping a gold table and putting a paragraph
    back would leave unanswerable rows with systematically fewer tables, which
    is the same leak one modality further down.
    """
    pool: dict[str, list[dict]] = {"table": [], "paragraph": []}
    seen: set[str] = set()
    for row in rows:
        for chunk in row.get("chunks", []):
            title = chunk.get("id")
            if not title or title in seen:
                continue
            if chunk.get("is_gold") or not str(chunk.get("source", "")).endswith("distractor"):
                continue
            seen.add(title)
            text = chunk.get("text", "")
            pool[_chunk_kind(title)].append({
                "id": title,
                "text": text,
                "document_id": chunk.get("document_id", ""),
                "document_chunk_id": chunk.get("document_chunk_id", ""),
                "norm": normalize_for_leak_check(text),
            })
    return pool


def _draw_backfill(
    candidates: list[dict],
    *,
    excluded_titles: set[str],
    own_document: str,
    leaks: list[str],
) -> dict | None:
    """Draw one usable replacement chunk by rejection sampling.

    The pool holds every distractor of the whole dataset, so scanning it once
    per row would dominate a step that is otherwise LLM-bound.
    """
    for _ in range(50):
        candidate = random.choice(candidates) if candidates else None
        if candidate is None:
            return None
        if candidate["id"] in excluded_titles:
            continue
        if candidate["document_id"] == own_document:
            continue
        if any(leak in candidate["norm"] for leak in leaks):
            continue
        return candidate
    return None


def _select_backfill(
    row: dict,
    pool: dict[str, list[dict]] | None,
    *,
    present_titles: set[str],
    dropped_titles: list[str],
) -> list[dict]:
    """Draw one replacement chunk per dropped chunk, of the same kind."""
    if not dropped_titles or not pool:
        return []
    own_document = row.get("document_id", "")
    leaks = _answer_leak_strings(row)
    excluded = set(present_titles)
    picked: list[dict] = []
    for dropped in dropped_titles:
        kind = _chunk_kind(dropped)
        candidate = _draw_backfill(
            pool.get(kind, []),
            excluded_titles=excluded,
            own_document=own_document,
            leaks=leaks,
        )
        if candidate is None:
            # No distractor of that kind left: any chunk still beats a hole.
            other = "paragraph" if kind == "table" else "table"
            candidate = _draw_backfill(
                pool.get(other, []),
                excluded_titles=excluded,
                own_document=own_document,
                leaks=leaks,
            )
        if candidate is None:
            continue
        excluded.add(candidate["id"])
        picked.append(candidate)
    return picked


def create_unanswerable_row(
    row: dict, mode: str = "drop_gold", backfill_pool: dict[str, list[dict]] | None = None
) -> dict | None:
    """Turn an answerable row into an unanswerable one.

    ``drop_gold`` removes every gold chunk (table and/or paragraphs).
    ``partial`` applies only to ``table-text`` questions (both modalities are
    required to answer): it removes either the table or the gold paragraphs,
    leaving genuine but insufficient evidence in context, a hard refusal.
    Returns None when the mode is not applicable.

    ``backfill_pool`` replaces the dropped chunks one-for-one with distractors
    from other reports, so the twin rows keep the same chunk count and roughly
    the same context length. The replacements are also flagged as protected
    background: gold chunks are never dropped by ``format_context_chunks``, so
    without protection the unanswerable row would offer a larger removable pool
    and end up shorter anyway.
    """
    context = row["context"]
    titles = context["title"]
    sentences = context["sentences"]
    relevant = set(row.get("supporting_facts", {}).get("title", []))
    answer_from = row.get("answer_from", "")

    if mode == "partial":
        if answer_from != "table-text" or "Table" not in relevant or len(relevant) < 2:
            return None
        if random.random() < 0.5:
            drop = {"Table"}
        else:
            drop = relevant - {"Table"}
    elif mode == "drop_gold":
        drop = set(relevant)
        if answer_from == "table":
            drop.add("Table")
    else:
        raise ValueError(f"Unknown unanswerable mode: {mode!r}")

    keep_titles = []
    keep_sentences = []
    for title, sents in zip(titles, sentences):
        if title not in drop:
            keep_titles.append(title)
            keep_sentences.append(sents)

    if len(keep_titles) < 2 or len(keep_titles) == len(titles):
        return None

    kept = set(keep_titles)
    replacements = _select_backfill(
        row,
        backfill_pool,
        present_titles=set(titles),
        dropped_titles=[t for t in titles if t not in kept],
    )
    for chunk in replacements:
        keep_titles.append(chunk["id"])
        keep_sentences.append([chunk["text"]])

    return {
        **row,
        "id": f"{row['id']}_unanswerable",
        "answer": "",
        "answer_raw": [],
        "supporting_facts": {"title": [], "sent_id": []},
        "_is_unanswerable": True,
        "_unanswerable_mode": mode,
        "_unanswerable_reasoning": UNANSWERABLE_REFUSAL,
        # Stand in for the protection the gold chunks enjoyed, so both twins
        # expose the same number of removable background chunks. The gold
        # modality a partial row keeps is protected for the same reason, and
        # because dropping it at random would silently downgrade a hard refusal
        # with partial evidence into a plain drop_gold one.
        "_backfilled_titles": [c["id"] for c in replacements],
        "_protected_background_titles": [c["id"] for c in replacements]
        + [t for t in keep_titles if t in relevant],
        "context": {"title": keep_titles, "sentences": keep_sentences},
        # Keep provenance aligned with the reduced retrieval context.  This is
        # used only to find the original source document for contextualization.
        "chunks": [c for c in row.get("chunks", []) if c.get("id") in kept]
        + [
            {
                "id": c["id"],
                "text": c["text"],
                "is_gold": False,
                "source": "backfill_distractor",
                "document_id": c["document_id"],
                "document_chunk_id": c["document_chunk_id"],
            }
            for c in replacements
        ],
    }


def prepare_rows_with_unanswerable(
    rows: list[dict],
    unanswerable_ratio: float,
    partial_ratio: float = 0.0,
    backfill: bool = True,
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
    partial_count = 0
    if need_new > 0 and answerable:
        pool = build_backfill_pool(rows) if backfill else None
        if backfill and not any(pool.values()):
            print("  (no distractor in the converted rows: unanswerable rows are not backfilled)")
        need_new = min(need_new, len(answerable))
        indices = random.sample(range(len(answerable)), need_new)
        for i in indices:
            u = None
            if random.random() < partial_ratio:
                u = create_unanswerable_row(answerable[i], mode="partial", backfill_pool=pool)
                if u is not None:
                    partial_count += 1
            if u is None:
                u = create_unanswerable_row(answerable[i], backfill_pool=pool)
            if u is not None:
                new_unanswerable.append(u)

    if partial_count:
        print(f"  ({partial_count} partial (table-text) unanswerable rows)")
    if new_unanswerable:
        backfilled = sum(len(u["_backfilled_titles"]) for u in new_unanswerable)
        print(f"  ({backfilled} chunks backfilled into {len(new_unanswerable)} unanswerable rows)")
    all_rows = answerable + existing_unanswerable + new_unanswerable
    random.shuffle(all_rows)
    return all_rows


# ---- Contextual retrieval over the original TAT-QA reports -----------------
SourceDocuments = dict[str, list[dict]]
SourceChunkLookup = dict[str, tuple[str, str]]
CONTEXTUALIZER_CACHE_VERSION = 1

_ctx_cache: dict[str, dict[str, str]] = {}
_ctx_inflight: dict[str, asyncio.Task] = {}
_ctx_cache_lock: asyncio.Lock | None = None  # created lazily inside the event loop
_ctx_cache_path: Path | None = None
_ctx_persisted: set[str] = set()


def load_contextualization_sources(
    source_path: str,
) -> tuple[SourceDocuments, SourceChunkLookup]:
    """Load original reports and index every chunk by its stable content hash."""
    docs, parse_warnings = load_tatqa_docs(Path(source_path))
    for warning in parse_warnings:
        log(f"[warn] {warning}")

    documents: SourceDocuments = {}
    chunk_lookup: SourceChunkLookup = {}
    for doc in docs:
        doc_id, chunks, _ = tatqa_document_chunks(doc)
        documents[doc_id] = chunks
        for chunk in chunks:
            # setdefault mirrors the converter's global content deduplication:
            # identical chunks retain the provenance of their first occurrence.
            chunk_lookup.setdefault(
                content_hash(chunk["text"]), (doc_id, chunk["id"])
            )
    return documents, chunk_lookup


def _ctx_cache_key(document_id: str, chunks: list[dict], language: str = "en") -> str:
    """Content/config-addressed key: stale prompts or models cannot reuse results."""
    payload = json.dumps(
        {
            "document_id": document_id,
            "chunks": chunks,
            "algorithm_version": CONTEXTUALIZER_CACHE_VERSION,
            "language": language,
            "model": LLM_MODEL,
            "prompt": CONTEXTUALIZER_SYSTEM_PROMPT,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_contextualization_cache(path: Path) -> None:
    """Restore source-document summaries written by an earlier Airflow attempt."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                key = record["key"]
                summaries = record["summaries"]
                if not isinstance(key, str) or not isinstance(summaries, dict):
                    raise TypeError("invalid cache record")
            except (json.JSONDecodeError, KeyError, TypeError):
                log(f"[warn] ignoring malformed contextualization cache line {line_number}")
                continue
            _ctx_cache[key] = {
                str(title): str(summary) for title, summary in summaries.items()
            }
            _ctx_persisted.add(key)


def _persist_contextualization_cache(key: str, document_id: str, summaries: dict[str, str]) -> None:
    """Append one complete source report result to the restart-safe JSONL cache."""
    if _ctx_cache_path is None or key in _ctx_persisted:
        return
    # Do not make a transient total failure permanent. Per-chunk failures have
    # already exhausted their retries and may remain empty, as in OpenRAG.
    if not any(summary.strip() for summary in summaries.values()):
        return
    record = {"key": key, "document_id": document_id, "summaries": summaries}
    with _ctx_cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    _ctx_persisted.add(key)


async def _get_or_compute_summaries(
    session: aiohttp.ClientSession,
    document_id: str,
    chunks: list[dict],
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    language: str = "en",
    max_retries: int = 3,
    request_timeout: int = 120,
    debug_http: bool = False,
) -> dict[str, str]:
    """Contextualize one source report once, including concurrent callers."""
    global _ctx_cache_lock
    if _ctx_cache_lock is None:
        _ctx_cache_lock = asyncio.Lock()

    key = _ctx_cache_key(document_id, chunks, language)
    async with _ctx_cache_lock:
        if key in _ctx_cache:
            return _ctx_cache[key]
        task = _ctx_inflight.get(key)
        if task is None:
            llm_chunks = [
                {"title": chunk["id"], "text": chunk["text"]}
                for chunk in chunks
            ]
            task = asyncio.create_task(contextualize_chunks(
                session,
                llm_chunks,
                semaphore,
                api_key=api_key,
                language=language,
                filename=f"TAT-QA financial report {document_id}",
                max_retries=min(max_retries, 2),
                timeout_seconds=request_timeout,
                debug_http=debug_http,
                # Match the production chunker: one independent request per
                # chunk with first_chunks + two preceding chunks + current.
                batched=False,
            ))
            _ctx_inflight[key] = task

    try:
        summaries = await task
    except Exception as exc:
        # A document-level failure degrades to raw chunks; individual request
        # failures are already represented by an empty summary in utils.py.
        log(f"[warn] contextualization failed for document {document_id}: {exc}")
        summaries = {}

    async with _ctx_cache_lock:
        if summaries:
            _ctx_cache[key] = summaries
            _persist_contextualization_cache(key, document_id, summaries)
        if _ctx_inflight.get(key) is task:
            del _ctx_inflight[key]
    return summaries


def _retrieved_chunk_provenance(
    row: dict,
    title: str,
    text: str,
    source_lookup: SourceChunkLookup,
) -> tuple[str, str] | None:
    """Resolve a displayed retrieval chunk back to its original report chunk."""
    for chunk in row.get("chunks", []):
        if chunk.get("id") != title:
            continue
        document_id = str(chunk.get("document_id", "")).strip()
        document_chunk_id = str(chunk.get("document_chunk_id", "")).strip()
        if document_id and document_chunk_id:
            return document_id, document_chunk_id
        break

    # Compatibility for converted JSONL files created before provenance fields
    # were added. Exact content hashes are stable across conversion runs.
    return source_lookup.get(content_hash(text))


async def _contextualize_retrieved_chunks(
    session: aiohttp.ClientSession,
    row: dict,
    source_documents: SourceDocuments,
    source_lookup: SourceChunkLookup,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    max_retries: int,
    request_timeout: int,
    debug_http: bool,
) -> dict[str, str]:
    """Contextualize retrieved chunks against each chunk's real source report."""
    retrieved: list[tuple[str, str, str]] = []
    needed_documents: dict[str, list[dict]] = {}
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        text = " ".join(sents).strip() if isinstance(sents, list) else str(sents).strip()
        provenance = _retrieved_chunk_provenance(row, title, text, source_lookup)
        if provenance is None:
            continue
        document_id, source_chunk_id = provenance
        source_chunks = source_documents.get(document_id)
        if source_chunks is None:
            continue
        retrieved.append((title, document_id, source_chunk_id))
        needed_documents[document_id] = source_chunks

    if not needed_documents:
        return {}

    document_ids = list(needed_documents)
    results = await asyncio.gather(*[
        _get_or_compute_summaries(
            session,
            document_id,
            needed_documents[document_id],
            semaphore,
            api_key=api_key,
            max_retries=max_retries,
            request_timeout=request_timeout,
            debug_http=debug_http,
        )
        for document_id in document_ids
    ])
    by_document = dict(zip(document_ids, results))
    summaries = {
        displayed_title: by_document.get(document_id, {}).get(source_chunk_id, "")
        for displayed_title, document_id, source_chunk_id in retrieved
    }
    # Complete document failure must fall back to raw chunks rather than emit a
    # context marker with no useful text for every chunk.
    return summaries if any(summary.strip() for summary in summaries.values()) else {}


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
    source_documents: SourceDocuments | None = None,
    source_lookup: SourceChunkLookup | None = None,
    contextualize: bool = False,
) -> dict | None:
    is_unanswerable = row.get("_is_unanswerable", False)

    # Optionally compute contextual summaries. Unanswerable rows are
    # contextualised too: skipping them would make the presence of a [CONTEXT]
    # block a perfect predictor of "answerable". Every retrieved chunk is mapped
    # back to its own complete source report; distractors are never presented as
    # if they were sections of the question's gold report.
    summaries: dict[str, str] | None = None
    if contextualize:
        if source_documents is None or source_lookup is None:
            raise ValueError("Original TAT-QA source documents are required for contextualization")
        summaries = await _contextualize_retrieved_chunks(
            session,
            row,
            source_documents,
            source_lookup,
            semaphore,
            api_key=LLM_API_KEY,
            max_retries=max_retries,
            request_timeout=request_timeout,
            debug_http=debug_http,
        )

    context_str, chunk_stats = format_context_chunks(
        context=row["context"],
        relevant_titles=set(row.get("supporting_facts", {"title": [], "sent_id": []}).get("title", [])),
        shuffle=shuffle,
        max_remove_background_ratio=max_remove_background_ratio,
        min_chunks=2,
        protected_background_titles={"Table", *row.get("_protected_background_titles", [])},
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
        language="en",
        style=DEFAULT_CITATION_STYLE,
        end_of_completion=end_of_completion,
        refusal_instruction=_generic_refusal_instruction("en"),
    )
    reasoning = await call_llm(
        session,
        row["question"],
        context_str,
        semaphore,
        api_key=LLM_API_KEY,
        system_prompt=system_prompt,
        context_in_system=True,
        max_retries=max_retries,
        timeout_seconds=request_timeout,
        max_tokens=max_tokens,
        debug_http=debug_http,
    )
    if reasoning is None:
        log(f"Row failed: id={row.get('id')}")
        return None
    # Whether the model actually refused (informational; the filter enforces it).
    grounded_refusal = bool(is_unanswerable and detect_refusal(reasoning))

    title_mapping = chunk_stats.pop("title_mapping", {})
    sf_titles = list(dict.fromkeys(
        title_mapping.get(t, t)
        for t in row.get("supporting_facts", {}).get("title", [])
    ))
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
        "grounded_refusal": grounded_refusal,
        "citation_end_of_completion": end_of_completion,
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
    contextualize_ratio: float = 0.0,
    unanswerable_partial_ratio: float = 0.0,
    unanswerable_backfill: bool = True,
    source_documents_path: str | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        log(f"Random seed: {seed}")
    if not 0.0 <= contextualize_ratio <= 1.0:
        raise ValueError("--contextualize-ratio must be between 0.0 and 1.0")

    output_prefix = output_prefix.removesuffix(".jsonl")
    output_file = f"{output_prefix}.jsonl"
    checkpoint_file = f"{output_prefix}_checkpoint.json"

    # A run may be invoked more than once in the same interpreter (notably in
    # tests); asyncio locks/tasks must never leak across event loops.
    global _ctx_cache_lock, _ctx_cache_path
    _ctx_cache.clear()
    _ctx_inflight.clear()
    _ctx_persisted.clear()
    _ctx_cache_lock = None
    _ctx_cache_path = None

    source_documents: SourceDocuments | None = None
    source_lookup: SourceChunkLookup | None = None
    if contextualize_ratio > 0:
        if not source_documents_path:
            raise ValueError(
                "--source-documents is required when --contextualize-ratio is greater than 0"
            )
        source_documents, source_lookup = load_contextualization_sources(
            source_documents_path
        )
        _ctx_cache_path = Path(f"{output_prefix}_contextualization_cache.jsonl")
        _load_contextualization_cache(_ctx_cache_path)
        log(f"Contextualization source: {len(source_documents)} original TAT-QA reports")
        if _ctx_cache:
            log(f"Contextualization cache: {len(_ctx_cache)} source reports restored")

    rows = jsonl_read(dataset_path, limit=limit)
    rows = prepare_rows_with_unanswerable(
        rows,
        unanswerable_ratio=unanswerable_ratio,
        partial_ratio=unanswerable_partial_ratio,
        backfill=unanswerable_backfill,
    )

    processed = load_checkpoint_ids(checkpoint_file)
    rows = [r for r in rows if r["id"] not in processed]

    log(f"Rows to process: {len(rows)}")
    if not rows:
        if Path(output_file).is_file() and Path(output_file).stat().st_size > 0:
            log("Nothing to process; existing non-empty output is complete.")
            return
        raise RuntimeError(
            "No rows remain to process, but the output JSONL is missing or empty. "
            "The checkpoint and output are inconsistent."
        )

    # Unanswerable rows now also call the LLM (to generate query-grounded refusals).
    needs_llm = bool(rows)
    if needs_llm and not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required to process rows (set in .env file or environment)")

    if contextualize_ratio > 0:
        log(f"Contextualize ratio: {contextualize_ratio:.0%}")

    # Pre-assign which rows get contextualised (deterministic with seed)
    ctx_flags = [random.random() < contextualize_ratio for _ in rows]

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
                batch_flags = ctx_flags[start:start + batch_size]
                tasks = [
                    process_row(
                        session, row, semaphore,
                        shuffle=shuffle,
                        max_remove_background_ratio=max_remove_background_ratio,
                        max_retries=max_retries,
                        request_timeout=request_timeout,
                        max_tokens=max_tokens,
                        debug_http=debug_http,
                        source_documents=source_documents,
                        source_lookup=source_lookup,
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
    if metrics.failed:
        raise RuntimeError(
            f"{metrics.failed} row(s) failed generation; successful rows were checkpointed "
            "and the task should be retried."
        )


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
    parser.add_argument(
        "--source-documents",
        default=None,
        help="Original TAT-QA JSON used to contextualize chunks against their full source reports",
    )
    parser.add_argument("--output", default="tatqa_augmented", help="Output prefix (default: tatqa_augmented)")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows before augmentation")
    parser.add_argument("--concurrency", type=int, default=LLM_MAX_CONCURRENT, help="Async concurrency")
    parser.add_argument("--unanswerable-ratio", type=float, default=0.1, help="Target unanswerable ratio")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--unanswerable-partial-ratio", type=float, default=0.0,
                        help="Fraction of created unanswerable rows that keep partial gold evidence "
                             "(table-text questions: drop either the table or the gold paragraphs). "
                             "Falls back to drop_gold when not applicable (default: 0.0)")
    parser.add_argument("--no-unanswerable-backfill", action="store_true",
                        help="Do not replace the dropped gold chunks with distractors from other "
                             "reports. Leaves unanswerable rows shorter than their answerable twin, "
                             "which makes the context length predict the refusal (default: backfill)")
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
    parser.add_argument(
        "--contextualize-ratio",
        type=float,
        default=0.0,
        help="Fraction of rows to format with OpenRAG contextual summaries (0.0–1.0)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    run_async(
        run(
            dataset_path=args.dataset,
            output_prefix=args.output,
            limit=args.limit,
            concurrency=args.concurrency,
            unanswerable_ratio=args.unanswerable_ratio,
            unanswerable_partial_ratio=args.unanswerable_partial_ratio,
            unanswerable_backfill=not args.no_unanswerable_backfill,
            shuffle=not args.no_shuffle,
            max_remove_background_ratio=args.max_remove_background_ratio,
            seed=args.seed,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            max_tokens=args.max_tokens,
            debug_http=args.debug_http,
            contextualize_ratio=args.contextualize_ratio,
            source_documents_path=args.source_documents,
        )
    )


if __name__ == "__main__":
    main()
