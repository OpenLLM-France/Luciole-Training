#!/usr/bin/env python3
"""
Convert TATQA data into augmentation format compatible with the existing pipeline.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from benchmark_format import augment_row_to_benchmark
from distractor_generation import (
    DenseRetriever,
    MixedDistractorSelector,
    build_bm25_index,
    content_hash as _ctx_hash,
    parse_mix_spec,
    select_bm25_distractors,
    verify_hard_candidates,
)
from utils import load_env_key


def _stringify_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|").strip()


def table_to_markdown(table_rows: list[list[object]]) -> str:
    if not table_rows:
        return ""
    headers = [_stringify_cell(cell) for cell in table_rows[0]]
    if not headers:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in table_rows[1:]:
        cells = [_stringify_cell(c) for c in row]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + body)


def _extract_docs_from_broken_array(text: str) -> list[dict]:
    docs = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        m = re.search(r"\{", text[idx:])
        if not m:
            break
        start = idx + m.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "table" in obj and "questions" in obj:
                docs.append(obj)
            idx = start + end
        except json.JSONDecodeError:
            idx = start + 1
    return docs


def load_tatqa_docs(input_path: Path) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    text = input_path.read_text(encoding="utf-8")

    # 1) Try full JSON first.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            docs = [d for d in data if isinstance(d, dict)]
            return docs, warnings
        if isinstance(data, dict):
            return [data], warnings
    except json.JSONDecodeError as e:
        warnings.append(f"Full JSON parse failed: {e}")

    # 2) Try JSONL line by line.
    docs = []
    jsonl_ok = 0
    jsonl_bad = 0
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                docs.append(obj)
                jsonl_ok += 1
        except json.JSONDecodeError:
            jsonl_bad += 1
    if jsonl_ok > 0 and jsonl_bad == 0:
        return docs, warnings
    if jsonl_ok > 0:
        warnings.append(f"JSONL parse partial: {jsonl_ok} valid lines, {jsonl_bad} invalid lines.")

    # 3) Fallback for truncated pretty-printed array.
    recovered = _extract_docs_from_broken_array(text)
    if recovered:
        warnings.append(
            f"Recovered {len(recovered)} docs from partially malformed/truncated JSON array."
        )
        return recovered, warnings

    raise ValueError(f"Could not parse input file: {input_path}")


def _format_number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def normalize_answer(question: dict) -> tuple[str, object, list[str] | None]:
    answer_type = question.get("answer_type", "span")
    answer_raw = question.get("answer")
    answer_spans: list[str] | None = None

    if answer_type == "multi-span":
        if isinstance(answer_raw, list):
            answer_spans = [str(x) for x in answer_raw]
            answer = ", ".join(answer_spans)
        else:
            answer_spans = [str(answer_raw)]
            answer = str(answer_raw)
        return answer, answer_raw, answer_spans

    if answer_type in {"arithmetic", "count"}:
        if isinstance(answer_raw, list) and answer_raw:
            first = answer_raw[0]
            answer = _format_number(first)
        else:
            answer = _format_number(answer_raw)
        return answer, answer_raw, answer_spans

    # Default span
    if isinstance(answer_raw, list):
        answer = str(answer_raw[0]) if answer_raw else ""
    else:
        answer = str(answer_raw)
    return answer, answer_raw, answer_spans


MIN_PARAGRAPH_CHARS = 50


def _merge_small_paragraphs(
    paragraphs: list[dict],
) -> tuple[list[dict], dict[int, str]]:
    """Merge paragraphs whose text is shorter than MIN_PARAGRAPH_CHARS into the
    next paragraph (or the previous one if they are last).

    Returns the merged paragraph list and a mapping from each original paragraph
    order number to the title of the merged chunk it ended up in, so that
    supporting_facts references can be updated.
    """
    # Build list of (order, text) keeping original order numbers
    items: list[tuple[int, str]] = []
    for p in paragraphs:
        order = p.get("order", 0)
        text = str(p.get("text", "")).strip()
        items.append((order, text))

    # Forward pass: merge small items into the next one
    merged: list[tuple[list[int], str]] = []  # (list_of_orders, combined_text)
    pending_orders: list[int] = []
    pending_texts: list[str] = []

    for order, text in items:
        pending_orders.append(order)
        pending_texts.append(text)
        if len(text) >= MIN_PARAGRAPH_CHARS:
            # Flush: combine all pending into one chunk
            combined = "\n".join(pending_texts)
            merged.append((list(pending_orders), combined))
            pending_orders = []
            pending_texts = []

    # If trailing small paragraphs remain, attach to last merged chunk
    if pending_orders:
        if merged:
            prev_orders, prev_text = merged[-1]
            combined = prev_text + "\n" + "\n".join(pending_texts)
            merged[-1] = (prev_orders + pending_orders, combined)
        else:
            # All paragraphs are tiny — keep them as a single chunk
            combined = "\n".join(pending_texts)
            merged.append((list(pending_orders), combined))

    # Build final paragraphs and order-to-title mapping
    merged_paragraphs: list[dict] = []
    order_to_title: dict[int, str] = {}
    for idx, (orders, text) in enumerate(merged, start=1):
        title = f"Paragraph {idx}"
        merged_paragraphs.append({"order": idx, "text": text})
        for orig_order in orders:
            order_to_title[orig_order] = title

    return merged_paragraphs, order_to_title


def tatqa_document_chunks(doc: dict) -> tuple[str, list[dict], dict[int, str]]:
    """Return the stable chunking of one original TAT-QA report.

    The same function is used for conversion, distractor provenance and later
    contextual retrieval.  Keeping this as a single source of truth is
    important: a retrieved chunk must be contextualized against the exact
    source document/chunk from which it was indexed, not against the mixed
    question context assembled after retrieval.
    """
    table = doc.get("table", {})
    doc_id = str(table.get("uid", "table"))
    chunks = [{
        "id": "Table",
        "text": table_to_markdown(table.get("table", [])),
        "type": "table",
    }]

    paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    merged_paragraphs, order_to_title = _merge_small_paragraphs(paragraphs)
    for paragraph in merged_paragraphs:
        chunks.append({
            "id": f"Paragraph {paragraph.get('order', '?')}",
            "text": str(paragraph.get("text", "")),
            "type": "paragraph",
        })
    return doc_id, chunks, order_to_title


def build_distractor_pool(docs: list[dict]) -> list[dict]:
    """Collect tables and post-merge paragraphs from every doc for BM25
    distractor selection.

    Each entry stores its source doc id so we can exclude same-doc chunks at
    selection time. Duplicate texts are deduplicated by content hash.
    Including tables makes the pool realistic for RAG scenarios where a
    retriever may return a table from an unrelated document.
    """
    pool: list[dict] = []
    seen_hashes: set[str] = set()
    for doc in docs:
        doc_id, chunks, _ = tatqa_document_chunks(doc)
        for chunk in chunks:
            text = chunk["text"].strip()
            if len(text) < MIN_PARAGRAPH_CHARS:
                continue
            h = _ctx_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pool.append({
                "doc_id": doc_id,
                "chunk_id": chunk["id"],
                "text": text,
                "hash": h,
                "type": chunk["type"],
            })
    return pool


def select_cross_doc_distractors(
    *,
    question: str,
    current_doc_id: str,
    pool: list[dict],
    bm25,
    num_distractors: int,
    start_rank: int = 1,
    answer_strings: list[str] | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Pick top-BM25 paragraphs from docs other than current_doc_id.

    If ``answer_strings`` is provided, paragraphs that contain any of these
    strings (normalized substring match) are skipped to avoid answer leakage.
    Strings shorter than ``LEAK_MIN_LENGTH`` characters are ignored to prevent
    over-filtering on numeric or single-token answers.
    """
    return select_bm25_distractors(
        question=question,
        pool=pool,
        bm25=bm25,
        num_distractors=num_distractors,
        exclude_doc_ids={current_doc_id},
        start_rank=start_rank,
        answer_strings=answer_strings,
        stats=stats,
    )


def attach_distractors(
    row: dict,
    distractors: list[dict],
    distractor_stats: dict | None = None,
) -> None:
    """Append cross-doc distractors to a converted row (context + chunks).

    Distractor titles use a type-prefix + content-hash so they (a) stay unique
    across docs, (b) don't collide with gold titles in title_mapping, and
    (c) "Paragraph/Table dist_*" get renumbered by format_context_chunks.
    """
    for d in distractors:
        prefix = "Table" if d.get("type") == "table" else "Paragraph"
        title = f"{prefix} dist_{d['hash']}"
        row["context"]["title"].append(title)
        row["context"]["sentences"].append([d["text"]])
        difficulty = d.get("difficulty")
        row["chunks"].append({
            "id": title,
            "text": d["text"],
            "is_gold": False,
            "source": f"{difficulty}_distractor" if difficulty else "bm25_distractor",
            "document_id": d.get("doc_id", ""),
            "document_chunk_id": d.get("chunk_id", ""),
        })
        if difficulty:
            # Arrow-friendly fixed-key structs (see convert_multilingual_qa).
            row.setdefault("_distractor_meta", []).append(
                {"title": title, "difficulty": difficulty}
            )
    if distractor_stats is not None:
        distractor_stats["added"] = distractor_stats.get("added", 0) + len(distractors)
        table_count = sum(1 for d in distractors if d.get("type") == "table")
        distractor_stats["tables_added"] = (
            distractor_stats.get("tables_added", 0) + table_count
        )


def convert_doc(
    doc: dict,
    *,
    question_limit: int | None = None,
    pool: list[dict] | None = None,
    bm25=None,
    num_distractors: int = 0,
    distractor_start_rank: int = 1,
    distractor_stats: dict | None = None,
    selector: MixedDistractorSelector | None = None,
) -> list[dict]:
    out = []
    table_uid, source_chunks, order_to_title = tatqa_document_chunks(doc)
    base_titles = [chunk["id"] for chunk in source_chunks]
    base_sentences = [[chunk["text"]] for chunk in source_chunks]

    questions = doc.get("questions", [])
    if question_limit is not None:
        questions = questions[:question_limit]

    for q in questions:
        q_uid = q.get("uid", "unknown")
        row_id = f"{table_uid}_{q_uid}"
        answer_from = q.get("answer_from", "")
        rel_paragraphs = q.get("rel_paragraphs", []) or []

        sf_titles: list[str] = []
        if answer_from in {"table", "table-text"}:
            sf_titles.append("Table")
        if answer_from in {"text", "table-text"}:
            for order in rel_paragraphs:
                # rel_paragraphs values may be str or int depending on the source
                order_key = int(order) if isinstance(order, str) else order
                mapped_title = order_to_title.get(order_key, f"Paragraph {order}")
                sf_titles.append(mapped_title)
        # Deduplicate preserving order
        sf_titles = list(dict.fromkeys(sf_titles))

        answer_norm, answer_raw, answer_spans = normalize_answer(q)

        # Build leak-check strings only for textual answer types. Arithmetic
        # and count answers are derived numbers — their value may legitimately
        # appear in unrelated paragraphs and would over-filter the pool.
        answer_type = q.get("answer_type", "span")
        leak_strings: list[str] = []
        if answer_type in {"span", "multi-span"}:
            if answer_spans:
                leak_strings = [s for s in answer_spans if s and str(s).strip()]
            elif isinstance(answer_raw, list):
                leak_strings = [str(s) for s in answer_raw if s]
            elif answer_raw:
                leak_strings = [str(answer_raw)]

        # Structured chunk export for the benchmark. Each chunk carries an
        # explicit is_gold flag (against rel_paragraphs / answer_from) and a
        # source label distinguishing same-doc background from added
        # cross-doc distractors. Order matches context["title"]. Distractors
        # are appended by attach_distractors, immediately on the legacy path,
        # or after the LLM verification pass on the mixed-difficulty path.
        sf_set = set(sf_titles)
        chunks: list[dict] = []
        for title, sents in zip(base_titles, base_sentences):
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            chunks.append({
                "id": title,
                "text": text,
                "is_gold": title in sf_set,
                "source": "original",
                "document_id": table_uid,
                "document_chunk_id": title,
            })

        row = {
            "id": row_id,
            "question": q.get("question", ""),
            "answer": answer_norm,
            "answer_raw": answer_raw,
            "answer_type": q.get("answer_type", ""),
            "answer_from": answer_from,
            "derivation": q.get("derivation", ""),
            "scale": q.get("scale", ""),
            "req_comparison": bool(q.get("req_comparison", False)),
            "document_id": table_uid,
            "context": {"title": list(base_titles), "sentences": list(base_sentences)},
            "supporting_facts": {"title": sf_titles, "sent_id": [0] * len(sf_titles)},
            "chunks": chunks,
            # Compatibility with existing reporting scripts
            "type": q.get("answer_type", "unknown"),
            "level": "tatqa",
        }

        if selector is not None:
            evidence_parts = [
                " ".join(sents)
                for title, sents in zip(base_titles, base_sentences)
                if title in sf_set
            ]
            selector.prepare(
                row_id,
                question=q.get("question", ""),
                evidence=" ".join(evidence_parts),
                exclude_doc_ids={table_uid},
                answer_strings=leak_strings or None,
            )
            row["_pending_distractors"] = row_id
        elif pool and bm25 and num_distractors > 0:
            distractors = select_cross_doc_distractors(
                question=q.get("question", ""),
                current_doc_id=table_uid,
                pool=pool,
                bm25=bm25,
                num_distractors=num_distractors,
                start_rank=distractor_start_rank,
                answer_strings=leak_strings or None,
                stats=distractor_stats,
            )
            if distractor_stats is not None and len(distractors) < num_distractors:
                distractor_stats["short_rows"] = (
                    distractor_stats.get("short_rows", 0) + 1
                )
            attach_distractors(row, distractors, distractor_stats)

        if answer_spans is not None:
            row["answer_spans"] = answer_spans
        out.append(row)
    return out


def convert(
    input_file: str,
    output_file: str,
    *,
    num_distractors: int = 0,
    distractor_start_rank: int = 1,
    output_format: str = "augment",
    include_id: bool = True,
    distractor_mix: str | None = None,
    verify_hard: bool = False,
    evidence_query: bool = False,
    dense_model: str | None = None,
    verify_concurrency: int = 10,
    verify_checkpoint: str | None = None,
    seed: int | None = None,
    limit: int | None = None,
) -> tuple[int, int, int]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1")

    docs, warnings = load_tatqa_docs(Path(input_file))

    pool: list[dict] = []
    bm25 = None
    if num_distractors > 0 or distractor_mix:
        pool = build_distractor_pool(docs)
        if pool:
            n_tables = sum(1 for p in pool if p["type"] == "table")
            n_paras = len(pool) - n_tables
            print(f"BM25 pool: {len(pool)} unique chunks ({n_tables} tables + {n_paras} paragraphs)")
            bm25 = build_bm25_index(pool)
        else:
            print("[warn] distractor pool is empty, no distractors will be added")

    selector: MixedDistractorSelector | None = None
    if distractor_mix and bm25 is not None:
        mix = parse_mix_spec(distractor_mix)
        num_distractors = sum(mix.values())
        dense = None
        if dense_model:
            print(f"Loading dense retriever {dense_model} and encoding pool...")
            dense = DenseRetriever(dense_model)
            dense.index_pool(pool)
        selector = MixedDistractorSelector(
            pool=pool,
            bm25=bm25,
            mix=mix,
            dense=dense,
            use_evidence_query=evidence_query,
            verify_hard=verify_hard,
            seed=seed,
        )
        print(
            f"Mixed distractor selection: {mix} "
            f"(verify_hard={verify_hard}, evidence_query={evidence_query}, "
            f"dense={dense_model or 'off'})"
        )

    distractor_stats: dict[str, int] = {"added": 0, "leak_filtered": 0, "short_rows": 0}
    rows_out = []
    remaining = limit
    for doc in docs:
        if remaining == 0:
            break
        converted = convert_doc(
            doc,
            question_limit=remaining,
            pool=pool,
            bm25=bm25,
            num_distractors=num_distractors,
            distractor_start_rank=distractor_start_rank,
            distractor_stats=distractor_stats,
            selector=selector,
        )
        rows_out.extend(converted)
        if remaining is not None:
            remaining -= len(converted)

    if selector is not None:
        if verify_hard:
            # Default checkpoint sits next to the output so a same-run_name retry
            # resumes the verify pass instead of restarting it from scratch.
            ckpt = verify_checkpoint
            if ckpt is None:
                ckpt = str(Path(output_file).with_suffix("")) + "_verify_checkpoint.jsonl"
            asyncio.run(verify_hard_candidates(
                selector,
                api_key=load_env_key("LLM_API_KEY") or "",
                concurrency=verify_concurrency,
                checkpoint_path=ckpt,
            ))
        for row in rows_out:
            pending_key = row.pop("_pending_distractors", None)
            if pending_key is not None:
                attach_distractors(row, selector.finalize(pending_key), distractor_stats)

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows_out:
            if output_format == "benchmark":
                row = augment_row_to_benchmark(row, include_id=include_id)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for w in warnings:
        print(f"[warn] {w}")
    print(f"Docs parsed: {len(docs)}")
    print(f"Questions converted: {len(rows_out)}")
    if selector is not None:
        print("Mixed distractor stats:")
        for name in sorted(selector.stats):
            print(f"  {name}: {selector.stats[name]}")
    elif num_distractors > 0:
        added = distractor_stats["added"]
        leaked = distractor_stats["leak_filtered"]
        short = distractor_stats["short_rows"]
        considered = added + leaked
        leak_pct = (100.0 * leaked / considered) if considered else 0.0
        tables_added = distractor_stats.get("tables_added", 0)
        print(
            f"Cross-doc BM25 distractors per question: up to {num_distractors} "
            f"(start rank {distractor_start_rank})"
        )
        print(f"  added:          {added} ({tables_added} tables, {added - tables_added} paragraphs)")
        print(f"  leak-filtered:  {leaked} ({leak_pct:.1f}% of considered)")
        print(f"  short rows:     {short} (got fewer than {num_distractors} distractors)")
    return len(docs), len(rows_out), len(warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TATQA to augmentation or benchmark JSONL")
    parser.add_argument("--input", required=True, help="Path to TATQA input (JSON/JSONL)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=0,
        help="Number of cross-doc BM25 distractor paragraphs per question (default: 0 = none)",
    )
    parser.add_argument(
        "--distractor-start-rank",
        type=int,
        default=1,
        help="1-based BM25 rank to start selecting cross-doc distractors from (default: 1)",
    )
    parser.add_argument(
        "--output-format",
        choices=["augment", "benchmark"],
        default="augment",
        help="Output schema: augmentation format or prompt-agnostic benchmark format",
    )
    parser.add_argument(
        "--no-id",
        action="store_true",
        help="Do not include the id column in benchmark output",
    )
    parser.add_argument(
        "--distractor-mix", type=str, default=None,
        help="Mixed-difficulty selection, e.g. 'hard=2,medium=5,random=2'. "
             "Overrides --num-distractors/--distractor-start-rank.",
    )
    parser.add_argument(
        "--verify-hard", action="store_true",
        help="LLM-verify hard candidates (one call per question); rejected "
             "candidates are backfilled from medium.",
    )
    parser.add_argument(
        "--evidence-query", action="store_true",
        help="Also rank candidates by similarity to the gold table/paragraphs "
             "(RRF fusion).",
    )
    parser.add_argument(
        "--dense-model", type=str, default=None,
        help="Optional sentence-transformers model for hybrid dense+BM25 ranking",
    )
    parser.add_argument(
        "--verify-concurrency", type=int, default=10,
        help="Concurrent LLM verification calls (default: 10)",
    )
    parser.add_argument(
        "--verify-checkpoint", type=str, default=None,
        help="JSONL file caching --verify-hard verdicts so a killed run resumes "
             "instead of restarting the whole verify pass "
             "(default: <output>_verify_checkpoint.jsonl).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the random bucket")
    parser.add_argument("--limit", type=int, default=None, help="Limit converted questions before LLM verification")
    args = parser.parse_args()
    if args.distractor_start_rank < 1:
        raise ValueError("--distractor-start-rank must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    convert(
        args.input,
        args.output,
        num_distractors=args.num_distractors,
        distractor_start_rank=args.distractor_start_rank,
        output_format=args.output_format,
        include_id=not args.no_id,
        distractor_mix=args.distractor_mix,
        verify_hard=args.verify_hard,
        evidence_query=args.evidence_query,
        dense_model=args.dense_model,
        verify_concurrency=args.verify_concurrency,
        verify_checkpoint=args.verify_checkpoint,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
