#!/usr/bin/env python3
"""Export HotpotQA (distractor setting) as JSONL with a structured ``chunks``
field, matching the format produced by
convert_tatqa_to_augment_format.py and convert_multilingual_qa.py.

Can also emit the prompt-agnostic benchmark schema:
``query``, ``retrieved_documents``, ``titles``, ``supporting_index``, ``answer``.

All 10 paragraphs per question come from the original HotpotQA dataset, so every
chunk gets ``source: "original"``.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from benchmark_format import augment_row_to_benchmark

BASE_DIR = Path(__file__).parent


def _cache_dir(split: str) -> Path:
    return BASE_DIR / f"hotpotqa_cache_{split}"


def load_hotpotqa(split: str, dataset_cache: str | None = None) -> Dataset:
    if dataset_cache:
        cache_dir = Path(dataset_cache)
        print(f"Loading from dataset cache: {cache_dir}")
        return Dataset.load_from_disk(str(cache_dir))

    cache_dir = _cache_dir(split)
    if cache_dir.exists():
        print(f"Loading from cache: {cache_dir}")
        return Dataset.load_from_disk(str(cache_dir))
    print(f"Downloading HotpotQA {split} from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds.save_to_disk(str(cache_dir))
    print(f"Cached {len(ds)} rows to {cache_dir}")
    return ds


def build_chunks(context: dict, supporting_facts: dict) -> list[dict]:
    sf_titles = set(supporting_facts.get("title", []))
    chunks: list[dict] = []
    for title, sents in zip(context["title"], context["sentences"]):
        text = " ".join(sents) if isinstance(sents, list) else str(sents)
        chunks.append({
            "id": title,
            "text": text,
            "is_gold": title in sf_titles,
            "source": "original",
        })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HotpotQA as augmentation or prompt-agnostic benchmark JSONL",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
        help="HotpotQA split to export",
    )
    parser.add_argument(
        "--dataset-cache",
        default=None,
        help="Optional local Dataset.load_from_disk cache to export instead of HuggingFace HotpotQA",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows")
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
    args = parser.parse_args()

    ds = load_hotpotqa(args.split, dataset_cache=args.dataset_cache)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    output_path = Path(args.output)
    count = 0
    total_chunks = 0
    gold_chunks = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            chunks = build_chunks(row["context"], row["supporting_facts"])
            out = {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type", ""),
                "level": row.get("level", ""),
                "context": {"title": row["context"]["title"], "sentences": row["context"]["sentences"]},
                "supporting_facts": {"title": row["supporting_facts"]["title"], "sent_id": row["supporting_facts"]["sent_id"]},
                "chunks": chunks,
            }
            if args.output_format == "benchmark":
                out = augment_row_to_benchmark(out, include_id=not args.no_id)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
            for c in chunks:
                total_chunks += 1
                if c["is_gold"]:
                    gold_chunks += 1

    print(f"Wrote {count} rows to {output_path}")
    gold_pct = (100.0 * gold_chunks / total_chunks) if total_chunks else 0.0
    print(f"Total chunks: {total_chunks}, gold: {gold_chunks} ({gold_pct:.1f}%)")


if __name__ == "__main__":
    main()
