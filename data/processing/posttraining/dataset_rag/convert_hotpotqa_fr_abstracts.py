"""Convert Mvanypersele/hotpotqa-fr-abstracts to the input format of augment_hotpotqa.py.

The release uses the raw HotpotQA JSON layout (``_id``, ``context`` as
``[[title, [sentences]], ...]``, ``supporting_facts`` as ``[[title, sent_id], ...]``)
while augment_hotpotqa.py expects the HF ``hotpotqa/hotpot_qa`` layout (``id``,
``context = {"title": [...], "sentences": [[...]]}``, ``supporting_facts =
{"title": [...], "sent_id": [...]}``). The output JSONL is loaded directly by
``augment_hotpotqa.py --dataset <file>.jsonl`` (no arrow cache, so no datasets
3.x/4.x cache mismatch).

Rows whose documents would not fit the training sequence budget are dropped
before any LLM call: context tokens + ``--reserve-tokens`` (OpenRAG system prompt
~320 tokens, chunk headers, user question and completion) must stay within
``--max-seq-length``. ``--max-seq-length 0`` disables the filter.

Usage:
    python convert_hotpotqa_fr_abstracts.py --config filtered_qwen_v2 --split train \\
        --output hotpotqa_fr_abstracts_train.jsonl
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from utils import log

REPO_ID = "Mvanypersele/hotpotqa-fr-abstracts"
# Config name -> folder under data/ (the default config is published as abstract_v1).
CONFIG_DIRS = {"default": "abstract_v1"}


def convert_row(row: dict) -> dict:
    return {
        "id": row["_id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_facts": {
            "title": [title for title, _ in row["supporting_facts"]],
            "sent_id": [sent_id for _, sent_id in row["supporting_facts"]],
        },
        "context": {
            "title": [title for title, _ in row["context"]],
            "sentences": [sentences for _, sentences in row["context"]],
        },
    }


def context_tokens(row: dict, tokenizer) -> int:
    text = "\n\n".join(
        f"{title}\n{' '.join(sentences)}"
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    )
    return len(tokenizer(text).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="filtered_qwen_v2",
                        help="Release config (default, filtered, filtered_qwen, filtered_qwen_v2)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--revision", default=None,
                        help="Dataset commit to read (default: latest; the resolved sha is logged in the stats file)")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                        help="Training sequence budget in tokens; 0 disables the length filter")
    parser.add_argument("--reserve-tokens", type=int, default=1536,
                        help="Tokens kept for system prompt, chunk headers, question and completion")
    parser.add_argument("--tokenizer", default="OpenLLM-France/Luciole-1B-SFT-1.1",
                        help="Tokenizer used to count context tokens (private repo: needs HF_TOKEN)")
    args = parser.parse_args()

    revision = args.revision or HfApi().dataset_info(REPO_ID).sha
    folder = CONFIG_DIRS.get(args.config, args.config)
    src = hf_hub_download(REPO_ID, f"data/{folder}/{args.split}.jsonl", repo_type="dataset", revision=revision)
    log(f"Source: {REPO_ID}@{revision} data/{folder}/{args.split}.jsonl")

    tokenizer = None
    if args.max_seq_length > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    budget = args.max_seq_length - args.reserve_tokens

    stats = {"source": REPO_ID, "revision": revision, "config": args.config, "split": args.split,
             "max_seq_length": args.max_seq_length, "reserve_tokens": args.reserve_tokens,
             "tokenizer": args.tokenizer if tokenizer else None,
             "read": 0, "written": 0, "dropped_too_long": 0, "max_context_tokens": 0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fin, open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            stats["read"] += 1
            row = convert_row(json.loads(line))
            if tokenizer is not None:
                n = context_tokens(row, tokenizer)
                stats["max_context_tokens"] = max(stats["max_context_tokens"], n)
                if n > budget:
                    stats["dropped_too_long"] += 1
                    continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1

    stats_path = output.with_name(output.stem + "_convert_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote {stats['written']}/{stats['read']} rows to {output} "
        f"({stats['dropped_too_long']} dropped over {budget} context tokens); stats: {stats_path}")


if __name__ == "__main__":
    main()
