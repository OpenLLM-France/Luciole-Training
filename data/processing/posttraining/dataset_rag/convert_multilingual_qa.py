#!/usr/bin/env python3
"""
Convert multilingual QA datasets into the augmentation format expected by
augment_hotpotqa.py.

Each dataset is downloaded from HuggingFace, distractor passages are added
via BM25 ranking, and rows are output in HotpotQA-compatible JSONL.
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

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

# ---------------------------------------------------------------------------
# Dataset configuration registry
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[tuple[str, str], dict] = {
    # French -----------------------------------------------------------
    ("piaf", "fr"): {
        "hf_path": "piaf",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": True,
        "has_title": True,
    },
    ("frenchqa", "fr"): {
        "hf_path": "CATIE-AQ/frenchQA",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": False,
        "has_title": False,  # title field = source name, not article title
        "has_unanswerable": True,  # pragnakalp/squad_v2_french_translated includes impossible questions
        # Keep only training-compatible sources and deduplicate v2 variants.
        "source_filter": {
            "field": "title",
            "allowed": [
                "pragnakalp/squad_v2_french_translated",
                "piaf",
                "lincoln/newsquadfr",
            ],
        },
    },
    ("frenchqa-piaf", "fr"): {
        "hf_path": "CATIE-AQ/frenchQA",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": False,
        "has_title": False,
        "source_filter": {
            "field": "title",
            "allowed": ["piaf"],
        },
    },
    ("newsquadfr", "fr"): {
        "hf_path": "CATIE-AQ/frenchQA",
        "hf_name": None,
        "split": "validation",
        "trust_remote_code": False,
        "has_title": False,  # title field = source name, not article title
        "source_filter": {
            "field": "title",
            "allowed": ["lincoln/newsquadfr"],
        },
    },
    ("newsquadfr-train", "fr"): {
        "hf_path": "CATIE-AQ/frenchQA",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": False,
        "has_title": False,
        "source_filter": {
            "field": "title",
            "allowed": ["lincoln/newsquadfr"],
        },
    },
    ("newsquadfr-test", "fr"): {
        "hf_path": "lincoln/newsquadfr",
        "hf_name": None,
        "split": "test",
        "trust_remote_code": False,
        "has_title": True,
    },
    ("newsquadfr-v2-train", "fr"): {
        "hf_path": "CATIE-AQ/frenchQA",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": False,
        "has_title": False,
        "has_unanswerable": True,
        "source_filter": {
            "field": "title",
            "allowed": ["lincoln/newsquadfr_v2"],
        },
    },
    ("squad2-fr-local-train", "fr"): {
        "hf_path": "local:squad_v2_french_translated_train.json",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": False,
        "has_title": True,
        "has_unanswerable": True,
    },
    ("squad2-fr-local-eval", "fr"): {
        "hf_path": "local:squad_v2_french_translated_eval.json",
        "hf_name": None,
        "split": "validation",
        "trust_remote_code": False,
        "has_title": True,
        "has_unanswerable": True,
    },
    # ============================================================
    # Evaluation-only datasets (not used for training)
    # ============================================================
    # English — SQuAD 2.0 (50% unanswerable → refusal evaluation)
    ("squad2", "en"): {
        "hf_path": "rajpurkar/squad_v2",
        "hf_name": None,
        "split": "validation",
        "trust_remote_code": False,
        "has_title": True,
        "has_unanswerable": True,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
HF_DATASETS_HOME = Path.home() / ".cache" / "huggingface" / "datasets"


def _make_chunk_title(base_title: str, context_text: str) -> str:
    """Make chunk titles unique so citation-based evaluation can disambiguate passages."""
    clean_title = base_title.strip() or "Untitled"
    return f"{clean_title} [{_ctx_hash(context_text)}]"


def _make_title(row: dict, *, has_title: bool = True) -> str:
    """Generate a chunk title from passage text, independent of source article title."""
    context_text = row["context"].strip().lstrip("\ufeff")
    words = context_text.split()[:8]
    title = " ".join(words)
    if len(title) > 80:
        title = title[:77] + "..."
    return _make_chunk_title(title, context_text)


def _load_dataset(cfg: dict, cache_dir: Path) -> Dataset:
    """Load a dataset, caching to disk."""
    if str(cfg["hf_path"]).startswith("local:"):
        return _load_local_squad_dataset(Path(str(cfg["hf_path"]).removeprefix("local:")))

    if cache_dir.exists():
        print(f"Loading from cache: {cache_dir}")
        return Dataset.load_from_disk(str(cache_dir))

    print(f"Downloading {cfg['hf_path']} ({cfg.get('hf_name', '')}) ...")
    kwargs = {
        "path": cfg["hf_path"],
        "split": cfg["split"],
        "trust_remote_code": cfg["trust_remote_code"],
    }
    if cfg.get("hf_name"):
        kwargs["name"] = cfg["hf_name"]
    try:
        ds = load_dataset(**kwargs)
    except ValueError as exc:
        # SQuAD 2.0 can fail to load on older/newer `datasets` versions because the
        # cached feature schema uses `_type=List`. Fall back to the local Arrow cache.
        if cfg["hf_path"] == "rajpurkar/squad_v2" and "Feature type 'List' not found" in str(exc):
            ds = _load_squad2_from_arrow_cache(cfg)
        else:
            raise
    ds.save_to_disk(str(cache_dir))
    print(f"Cached {len(ds)} rows to {cache_dir}")
    return ds


def _load_local_squad_dataset(path: Path) -> Dataset:
    path = path if path.is_absolute() else BASE_DIR / path
    print(f"Loading local SQuAD JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    for article in payload.get("data", []):
        title = str(article.get("title", "")).strip()
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", "")).strip()
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", []) or []
                rows.append({
                    "id": qa.get("id", _ctx_hash(f"{title}::{context}::{qa.get('question', '')}")),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": {"text": [a.get("text", "") for a in answers if a.get("text", "")]},
                    "is_impossible": bool(qa.get("is_impossible", False)),
                })
    return Dataset.from_list(rows)


def _load_dataset_multi_split(cfg: dict, splits: list[str]) -> Dataset:
    datasets_per_split: list[Dataset] = []
    for split in splits:
        split_cfg = dict(cfg)
        split_cfg["split"] = split
        parts = [split_cfg["hf_path"].replace("/", "_")]
        if split_cfg.get("hf_name"):
            parts.append(split_cfg["hf_name"])
        parts.append(split_cfg["split"])
        cache_name = "_".join(parts)
        cache_dir = BASE_DIR / f"{cache_name}_cache"
        datasets_per_split.append(_load_dataset(split_cfg, cache_dir))
    return concatenate_datasets(datasets_per_split)


def _apply_dataset_filters(ds: Dataset, cfg: dict) -> Dataset:
    source_filter = cfg.get("source_filter")
    if source_filter:
        field = source_filter["field"]
        allowed = set(source_filter["allowed"])
        pre_len = len(ds)
        ds = ds.filter(lambda row: row.get(field, "") in allowed)
        print(f"Source filter ({field} in {sorted(allowed)}): {pre_len} → {len(ds)} rows")

    return ds


def _load_squad2_from_arrow_cache(cfg: dict) -> Dataset:
    split = cfg["split"]
    candidates = sorted(
        HF_DATASETS_HOME.glob("rajpurkar___squad_v2/squad_v2/*/*/squad_v2-*.arrow")
    )
    arrow_path = next((p for p in candidates if p.name == f"squad_v2-{split}.arrow"), None)
    if arrow_path is None:
        raise FileNotFoundError(
            f"SQuAD 2.0 Arrow cache for split '{split}' not found under {HF_DATASETS_HOME}"
        )

    import pyarrow as pa
    import pyarrow.ipc as ipc

    print(f"Falling back to Arrow cache: {arrow_path}")
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.RecordBatchStreamReader(source).read_all()
    return Dataset.from_list(table.to_pylist())


# ---------------------------------------------------------------------------
# Passage pool & BM25 distractors
# ---------------------------------------------------------------------------

def build_passage_pool(dataset: Dataset, *, has_title: bool = True) -> list[dict]:
    """Collect all unique passages from the dataset."""
    seen: set[str] = set()
    pool: list[dict] = []
    for row in dataset:
        title = _make_title(row, has_title=has_title)
        context_text = row["context"]

        h = _ctx_hash(context_text)
        if h in seen:
            continue
        seen.add(h)
        pool.append({
            "title": title,
            "text": context_text,
            "hash": h,
        })
    return pool


def select_distractors(
    question: str,
    true_hash: str,
    pool: list[dict],
    bm25,
    num_distractors: int = 9,
    start_rank: int = 1,
    *,
    answer_strings: list[str] | None = None,
    stats: dict | None = None,
) -> list[dict]:
    """Select topically-similar distractor passages using BM25.

    If ``answer_strings`` is provided, passages that contain any of these
    strings (normalized substring match) are skipped to avoid answer leakage.
    Strings shorter than ``LEAK_MIN_LENGTH`` characters are ignored.
    """
    return select_bm25_distractors(
        question=question,
        pool=pool,
        bm25=bm25,
        num_distractors=num_distractors,
        exclude_hashes={true_hash},
        start_rank=start_rank,
        answer_strings=answer_strings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Structured chunk export
# ---------------------------------------------------------------------------

def _build_chunks(
    titles: list[str],
    sentences: list[list[str]],
    *,
    gold_index: int = 0,
    is_unanswerable: bool = False,
    difficulties: dict[str, str] | None = None,
) -> list[dict]:
    """Build a structured chunks list for the benchmark export.

    In the multilingual converter the gold passage is always at position
    ``gold_index`` (0 by default) and everything else is a BM25 distractor.
    For unanswerable questions no chunk is marked gold. When mixed-difficulty
    selection is active, ``difficulties`` maps distractor title -> bucket and
    the source label becomes e.g. ``hard_distractor``.
    """
    chunks: list[dict] = []
    for i, (title, sents) in enumerate(zip(titles, sentences)):
        text = " ".join(sents) if isinstance(sents, list) else str(sents)
        is_original = i == gold_index
        if is_original:
            source = "original"
        elif difficulties and title in difficulties:
            source = f"{difficulties[title]}_distractor"
        else:
            source = "bm25_distractor"
        chunks.append({
            "id": title,
            "text": text,
            "is_gold": is_original and not is_unanswerable,
            "source": source,
        })
    return chunks


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def _row_selection_inputs(
    row: dict,
    *,
    has_unanswerable: bool,
    has_title: bool,
) -> tuple[str, str, str, list[str] | None, bool]:
    """Shared per-row inputs for distractor selection: (gold_title, gold_text,
    gold_hash, leak_strings, is_unanswerable)."""
    answers_text = row["answers"]["text"]
    is_unanswerable = has_unanswerable and not answers_text
    gold_title = _make_title(row, has_title=has_title)
    gold_text = row["context"]
    gold_hash = _ctx_hash(gold_text)
    leak_strings: list[str] | None = None
    if not is_unanswerable and answers_text:
        leak_strings = [a for a in answers_text if a]
    return gold_title, gold_text, gold_hash, leak_strings, is_unanswerable


def prepare_row_selection(
    row: dict,
    key,
    selector: MixedDistractorSelector,
    *,
    has_unanswerable: bool,
    has_title: bool,
) -> None:
    """Phase 1 of mixed selection: rank and stage candidates for one row."""
    _, gold_text, gold_hash, leak_strings, _ = _row_selection_inputs(
        row, has_unanswerable=has_unanswerable, has_title=has_title
    )
    selector.prepare(
        key,
        question=row["question"],
        evidence=gold_text,
        exclude_hashes={gold_hash},
        answer_strings=leak_strings,
    )


def convert_row(
    row: dict,
    dataset_name: str,
    language: str,
    pool: list[dict],
    bm25,
    num_distractors: int,
    *,
    distractor_start_rank: int = 1,
    has_unanswerable: bool = False,
    has_title: bool = True,
    distractor_stats: dict | None = None,
    preselected_distractors: list[dict] | None = None,
) -> dict:
    """Convert one extractive QA row to HotpotQA-compatible format.

    When ``preselected_distractors`` is given (mixed-difficulty path), BM25
    selection is skipped and the provided tagged candidates are used as-is.
    """
    answers_text = row["answers"]["text"]
    gold_title, gold_text, gold_hash, leak_strings, is_unanswerable = _row_selection_inputs(
        row, has_unanswerable=has_unanswerable, has_title=has_title
    )
    answer_text = "" if is_unanswerable else (answers_text[0] if answers_text else "")

    if preselected_distractors is not None:
        distractors = preselected_distractors
    else:
        distractors = select_distractors(
            row["question"],
            gold_hash,
            pool,
            bm25,
            num_distractors,
            start_rank=distractor_start_rank,
            answer_strings=leak_strings,
            stats=distractor_stats,
        )
        if distractor_stats is not None:
            distractor_stats["added"] = distractor_stats.get("added", 0) + len(distractors)
            if len(distractors) < num_distractors:
                distractor_stats["short_rows"] = distractor_stats.get("short_rows", 0) + 1

    titles = [gold_title] + [d["title"] for d in distractors]
    sentences = [[gold_text]] + [[d["text"]] for d in distractors]

    difficulties = {
        d["title"]: d["difficulty"] for d in distractors if d.get("difficulty")
    } or None
    chunks = _build_chunks(
        titles,
        sentences,
        gold_index=0,
        is_unanswerable=is_unanswerable,
        difficulties=difficulties,
    )

    original_id = row.get("id", _ctx_hash(row["question"]))
    result = {
        "id": f"{dataset_name}_{language}_{original_id}",
        "question": row["question"],
        "answer": answer_text,
        "type": "extractive",
        "level": dataset_name,
        "context": {"title": titles, "sentences": sentences},
        "chunks": chunks,
    }
    if difficulties:
        # Arrow-friendly fixed-key structs (a title-keyed dict would make the
        # datasets schema explode when this JSONL is loaded by augmentation).
        result["_distractor_meta"] = [
            {"title": t, "difficulty": d} for t, d in difficulties.items()
        ]

    if is_unanswerable:
        result["supporting_facts"] = {"title": [], "sent_id": []}
        result["_is_unanswerable"] = True
    else:
        result["supporting_facts"] = {"title": [gold_title], "sent_id": [0]}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert multilingual QA datasets to augmentation or benchmark format"
    )
    parser.add_argument(
        "--dataset", required=True,
        help=(
            "Dataset name (piaf, frenchqa, frenchqa-piaf, newsquadfr, "
            "newsquadfr-train, newsquadfr-test, newsquadfr-v2-train, "
            "squad2-fr-local-train, squad2-fr-local-eval, squad2)"
        ),
    )
    parser.add_argument(
        "--language", required=True,
        help="Language code (en, fr)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--num-distractors", type=int, default=9,
        help="Number of BM25 distractor passages per question (default: 9)",
    )
    parser.add_argument(
        "--distractor-start-rank", type=int, default=10,
        help="1-based BM25 rank to start selecting distractors from (default: 10)",
    )
    parser.add_argument(
        "--distractor-mix", type=str, default=None,
        help="Mixed-difficulty selection, e.g. 'hard=2,medium=5,random=2'. "
             "Overrides --num-distractors/--distractor-start-rank.",
    )
    parser.add_argument(
        "--verify-hard", action="store_true",
        help="LLM-verify hard candidates (one call per row) instead of relying "
             "on rank skipping; rejected candidates are backfilled from medium.",
    )
    parser.add_argument(
        "--evidence-query", action="store_true",
        help="Also rank candidates by similarity to the gold passage (RRF "
             "fusion), yielding evidence-similar distractors.",
    )
    parser.add_argument(
        "--dense-model", type=str, default=None,
        help="Optional sentence-transformers model for hybrid dense+BM25 "
             "ranking (e.g. intfloat/multilingual-e5-base).",
    )
    parser.add_argument(
        "--verify-concurrency", type=int, default=10,
        help="Concurrent LLM verification calls (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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

    key = (args.dataset, args.language)
    if key not in DATASET_CONFIGS:
        available = sorted(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown (dataset, language) pair: {key}. "
            f"Available: {available}"
        )

    if args.seed is not None:
        random.seed(args.seed)
    if args.distractor_start_rank < 1:
        raise ValueError("--distractor-start-rank must be >= 1")

    cfg = DATASET_CONFIGS[key]
    parts = [cfg['hf_path'].replace('/', '_')]
    if cfg.get('hf_name'):
        parts.append(cfg['hf_name'])
    parts.append(cfg['split'])
    cache_name = "_".join(parts)
    cache_dir = BASE_DIR / f"{cache_name}_cache"
    ds = _load_dataset(cfg, cache_dir)

    ds = _apply_dataset_filters(ds, cfg)

    pool_source = cfg.get("pool_source")
    if pool_source:
        if "splits" in pool_source:
            pool_ds = _load_dataset_multi_split(pool_source, pool_source["splits"])
            print(
                f"Building passage pool from source splits {pool_source['splits']} "
                f"({len(pool_ds)} rows before filtering/deduplication)..."
            )
        else:
            pool_parts = [pool_source["hf_path"].replace("/", "_")]
            if pool_source.get("hf_name"):
                pool_parts.append(pool_source["hf_name"])
            pool_parts.append(pool_source["split"])
            pool_cache_dir = BASE_DIR / ("_".join(pool_parts) + "_cache")
            pool_ds = _load_dataset(pool_source, pool_cache_dir)
            print(
                f"Building passage pool from dedicated source "
                f"({len(pool_ds)} rows before filtering/deduplication)..."
            )
        pool_ds = _apply_dataset_filters(pool_ds, pool_source)
        pool_has_title = pool_source.get("has_title", cfg.get("has_title", True))
    else:
        pool_ds = ds
        print(f"Building passage pool from current {cfg['split']} split ({len(pool_ds)} rows)...")
        pool_has_title = cfg.get("has_title", True)

    pool = build_passage_pool(pool_ds, has_title=pool_has_title)
    print(f"Unique passages: {len(pool)}")

    effective_distractors = min(args.num_distractors, len(pool) - 1)
    if effective_distractors < args.num_distractors:
        print(
            f"Warning: only {len(pool)} unique passages, "
            f"reducing distractors from {args.num_distractors} to {effective_distractors}"
        )

    print("Building BM25 index...")
    bm25 = build_bm25_index(pool)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    has_unanswerable = cfg.get("has_unanswerable", False)
    has_title = cfg.get("has_title", True)

    # Mixed-difficulty path: stage selections for every row, run the LLM
    # verification pass once over all rows, then convert with the finalized
    # tagged distractors.
    selector: MixedDistractorSelector | None = None
    if args.distractor_mix:
        mix = parse_mix_spec(args.distractor_mix)
        effective_distractors = sum(mix.values())
        dense = None
        if args.dense_model:
            print(f"Loading dense retriever {args.dense_model} and encoding pool...")
            dense = DenseRetriever(args.dense_model)
            dense.index_pool(pool)
        selector = MixedDistractorSelector(
            pool=pool,
            bm25=bm25,
            mix=mix,
            dense=dense,
            use_evidence_query=args.evidence_query,
            verify_hard=args.verify_hard,
            seed=args.seed,
        )
        print(
            f"Mixed distractor selection: {mix} "
            f"(verify_hard={args.verify_hard}, evidence_query={args.evidence_query}, "
            f"dense={args.dense_model or 'off'})"
        )
        for i, row in enumerate(ds):
            prepare_row_selection(
                row, i, selector,
                has_unanswerable=has_unanswerable, has_title=has_title,
            )
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(ds)} rows staged")
        if args.verify_hard:
            asyncio.run(verify_hard_candidates(
                selector,
                api_key=load_env_key("LLM_API_KEY") or "",
                concurrency=args.verify_concurrency,
            ))

    print(
        f"Converting {len(ds)} rows with {effective_distractors} distractors each "
        f"to {args.output_format} format..."
    )
    if has_unanswerable:
        print("  (dataset includes unanswerable questions)")
    output_path = Path(args.output)
    count = 0
    unanswerable_count = 0
    distractor_stats: dict[str, int] = {"added": 0, "leak_filtered": 0, "short_rows": 0}
    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            converted = convert_row(
                row, args.dataset, args.language,
                pool, bm25, effective_distractors,
                distractor_start_rank=args.distractor_start_rank,
                has_unanswerable=has_unanswerable,
                has_title=has_title,
                distractor_stats=distractor_stats,
                preselected_distractors=selector.finalize(i) if selector else None,
            )
            is_unanswerable = bool(converted.get("_is_unanswerable"))
            if args.output_format == "benchmark":
                converted = augment_row_to_benchmark(
                    converted,
                    include_id=not args.no_id,
                )
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
            if is_unanswerable:
                unanswerable_count += 1
            if count % 1000 == 0:
                print(f"  {count}/{len(ds)} rows converted")

    print(f"Done: {count} rows written to {output_path}")
    if unanswerable_count:
        print(f"  ({unanswerable_count} unanswerable rows)")
    if selector is not None:
        print("Mixed distractor stats:")
        for name in sorted(selector.stats):
            print(f"  {name}: {selector.stats[name]}")
    elif effective_distractors > 0:
        added = distractor_stats["added"]
        leaked = distractor_stats["leak_filtered"]
        short = distractor_stats["short_rows"]
        considered = added + leaked
        leak_pct = (100.0 * leaked / considered) if considered else 0.0
        print(f"BM25 distractor stats:")
        print(f"  added:          {added}")
        print(f"  leak-filtered:  {leaked} ({leak_pct:.1f}% of considered)")
        print(f"  short rows:     {short} (got fewer than {effective_distractors} distractors)")


if __name__ == "__main__":
    main()
