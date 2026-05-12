#!/usr/bin/env python3
"""Helpers for exporting prompt-agnostic RAG benchmark rows."""

from collections import defaultdict
from typing import Any


def _chunk_text(sentences: Any) -> str:
    """Normalize one context entry to a single document string."""
    if isinstance(sentences, list):
        return " ".join(str(sentence).strip() for sentence in sentences).strip()
    return str(sentences).strip()


def supporting_indices(
    titles: list[str],
    supporting_titles: list[str],
) -> list[int]:
    """Map supporting fact titles to zero-based indices in ``titles``."""
    indices_by_title: dict[str, list[int]] = defaultdict(list)
    for idx, title in enumerate(titles):
        indices_by_title[str(title)].append(idx)

    indices: list[int] = []
    seen: set[int] = set()
    for title in supporting_titles:
        for idx in indices_by_title.get(str(title), []):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)

    return indices


def augment_row_to_benchmark(row: dict, *, include_id: bool = True) -> dict:
    """Convert an augmentation-format row to the benchmark schema.

    Output schema:
    - ``query``: question string
    - ``retrieved_documents``: list of retrieved chunk texts
    - ``titles``: list of chunk titles aligned with ``retrieved_documents``
    - ``supporting_index``: zero-based indices into ``retrieved_documents``
    - ``answer``: expected answer; empty string with empty supports means unanswerable
    """
    context = row["context"]
    titles = [str(title) for title in context["title"]]
    documents = [_chunk_text(sentences) for sentences in context["sentences"]]
    supporting_titles = row.get("supporting_facts", {}).get("title", [])

    benchmark_row = {
        "query": row["question"],
        "retrieved_documents": documents,
        "titles": titles,
        "supporting_index": supporting_indices(titles, supporting_titles),
        "answer": row.get("answer", ""),
    }

    if include_id and "id" in row:
        return {"id": row["id"], **benchmark_row}
    return benchmark_row
