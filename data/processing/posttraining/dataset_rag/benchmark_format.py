#!/usr/bin/env python3
"""Helpers for exporting prompt-agnostic RAG benchmark rows."""

import random
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


def shuffle_and_renumber(
    titles: list[str],
    documents: list[str],
    supporting_titles: list[str],
    seed_key: str,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle chunks per row and renumber "Paragraph*"/"Table*" titles 1..N.

    Raw converter titles leak provenance ("Table" = gold doc, "Paragraph dist_<hash>"
    = cross-doc distractor, source chunks always first), and the benchmark prompt
    shows titles verbatim. Renumbering after a per-id deterministic shuffle makes
    titles and order carry no signal. Other titles are kept as-is.
    """
    order = list(range(len(titles)))
    random.Random(f"benchmark_order:{seed_key}").shuffle(order)
    counters = {"Paragraph": 0, "Table": 0}
    mapping: dict[str, str] = {}
    new_titles: list[str] = []
    for idx in order:
        old = titles[idx]
        prefix = next((p for p in counters if old.startswith(p)), None)
        if prefix is None:
            new = old
        else:
            counters[prefix] += 1
            new = f"{prefix} {counters[prefix]}"
        mapping.setdefault(old, new)
        new_titles.append(new)
    new_documents = [documents[idx] for idx in order]
    new_supporting = [mapping.get(str(t), str(t)) for t in supporting_titles]
    return new_titles, new_documents, new_supporting


def augment_row_to_benchmark(
    row: dict,
    *,
    include_id: bool = True,
    renumber_titles: bool = False,
) -> dict:
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
    if renumber_titles:
        titles, documents, supporting_titles = shuffle_and_renumber(
            titles, documents, supporting_titles, str(row.get("id", row["question"]))
        )

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
