#!/usr/bin/env python
"""Deduplicate the four xlam SFT sources of OpenLLM-France/tool_data into one dataset.

Reads data/sft/{xlam_thinking, xlam_irrelevance_thinking, xlam_oaiformat, xlam_irrelevance}/ of
OpenLLM-France/tool_data (pinned revision), keeps one random row per (user input, tools) across the
four sources, and writes the result to data/sft/xlam_dedup/ of OpenLLM-France/tool_data, with:
- ``chat_template_kwargs``: {"enable_thinking": true} for the rows of the *_thinking sources, false
  otherwise. NeMo-RL's OpenAIFormatV2Dataset reads this per-row column and gives it priority over
  the file-level chat_template_kwargs of the config;
- ``subset``: the source folder of the row (e.g. "xlam_thinking").

Overlap of the sources:
- xlam_thinking and xlam_oaiformat are the same xLAM rows (same id, same tools up to their order),
  with and without thinking: all the 40,432 user inputs of xlam_thinking are in xlam_oaiformat;
- xlam_irrelevance(_thinking) are xLAM rows whose needed tool was removed: same user input as the
  xlam_oaiformat / xlam_thinking row of the same id, the tool list without that tool (never the same
  tools, so never deduplicated against them); 439 rows are in both irrelevance sources;
- inside a source, ~3% of the rows repeat a user input under another id, mostly with other
  distractor tools (e.g. "Is the year 2000 a leap year?" 4 times in xlam_thinking).
The key (``dedup_key``) is the user messages + the tools (sorted), so the thinking / non-thinking
copies and the exact repeats are removed, the irrelevance rows and the repeats with other distractor
tools are kept. At REVISION: 112,116 -> 69,467 rows (split xlam_dedup of the sft config of tool_data).

Usage (datatrove env):
    python xlam_dedup.py --debug --local       # 100 rows per source, OpenLLM-France/tool_data_debug
    python xlam_dedup.py --local --commit "Add xlam_dedup"
"""

import hashlib
import json
import random

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter

# tool_data revision of the sources: main on 2026-11-09, after the removal of the rows naming another
# model (079da094; xlam_irrelevance_thinking ids 39299, 42225, 8792)
REVISION = "2190d86dabbc21d509229de10ebd206f212905b5"
# (data/sft/ folder, enable_thinking)
XLAM_SOURCES = [
    ("xlam_thinking", True),
    ("xlam_irrelevance_thinking", True),
    ("xlam_oaiformat", False),
    ("xlam_irrelevance", False),
]
OUT_NAME = "xlam_dedup"


def dedup_key(doc) -> str:
    """Deduplication key of a row: its user messages and its tool list (each tool serialized with
    sorted keys, the list sorted: the same tools in another order give the same key)."""
    user = "\n".join(m.get("content") or "" for m in doc.metadata["messages"] if m["role"] == "user")
    tools = doc.metadata.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools) or []
    return json.dumps([user, sorted(json.dumps(t, sort_keys=True) for t in tools)])


def dedup_user_input(sources: dict, seed: int):
    """Deduplicate across several sources: ``sources`` maps a folder to its rows; of each group of
    rows with the same user input and the same tools (``dedup_key``, in any of the sources), one
    random row is kept (all rows shuffled with ``seed``, first one kept). Returns {folder: kept rows,
    in their original order} and {folder: number of dropped rows}. Materializes the sources (xlam:
    ~112k rows)."""
    rows = [(folder, i) for folder, docs in sources.items() for i in range(len(docs))]
    random.Random(seed).shuffle(rows)
    seen, keep = set(), {folder: [] for folder in sources}
    for folder, i in rows:
        key = dedup_key(sources[folder][i])
        if key not in seen:
            seen.add(key)
            keep[folder].append(i)
    kept = {folder: [sources[folder][i] for i in sorted(keep[folder])] for folder in sources}
    dropped = {folder: len(sources[folder]) - len(kept[folder]) for folder in sources}
    return kept, dropped


class XlamDedup(PipelineStep):
    """Read the xlam sources (one task: dedup needs them all), deduplicate them with
    ``dedup_user_input`` and yield the kept rows with ``metadata["chat_template_kwargs"]``
    and ``metadata["subset"]``."""

    name = "🧹 xlam dedup"
    type = "📖 - READER"

    def __init__(self, sources, revision: str, seed: int, limit: int = -1):
        super().__init__()
        self.sources = sources
        self.revision = revision
        self.seed = seed
        self.limit = limit  # per source; set by create_executor in --debug mode

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from utils import instruct_adapter

        docs = {}
        for folder, _ in self.sources:
            reader = ParquetReader(
                f"hf://datasets/OpenLLM-France/tool_data@{self.revision}/data/sft/{folder}",
                glob_pattern="*.parquet",
                adapter=instruct_adapter,
                limit=self.limit,
            )
            docs[folder] = list(reader(None, 0, 1))
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:dedup:xlam".encode()).digest()[:4], "little")
        kept, dropped = dedup_user_input(docs, seed=seed)
        for folder, thinking in self.sources:
            self.stat_update(f"read/{folder}", value=len(docs[folder]))
            self.stat_update(f"duplicate/{folder}", value=dropped[folder])
            self.stat_update(f"kept/{folder}", value=len(kept[folder]))
            for doc in kept[folder]:
                doc.metadata["chat_template_kwargs"] = {"enable_thinking": thinking}
                doc.metadata["subset"] = folder
                yield doc


def xlam_dedup_hub_adapter(self, document):
    """``hub_adapter`` plus ``chat_template_kwargs`` (struct {enable_thinking: bool}) and ``subset``."""
    from utils import hub_adapter

    row = hub_adapter(self, document)
    row["chat_template_kwargs"] = {"enable_thinking": bool(document.metadata["chat_template_kwargs"]["enable_thinking"])}
    row["subset"] = document.metadata["subset"]
    return row


if __name__ == "__main__":
    from utils import create_executor, create_parser, parse_args

    parser = create_parser()
    parser.add_argument("--seed", type=int, default=42, help="Seed of the random choice among duplicates.")
    parser.add_argument("--revision", default=REVISION, help="Revision of OpenLLM-France/tool_data to read.")
    args = parse_args(parser)
    DATA_PATH = f"{args.data_path}/{OUT_NAME}"

    pipeline = [
        XlamDedup(XLAM_SOURCES, revision=args.revision, seed=args.seed),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/hf",
            output_filename=f"data/sft/{OUT_NAME}/${{rank}}.parquet",
            adapter=xlam_dedup_hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            commit_message=args.commit,
        ),
    ]
    executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{DATA_PATH}/logs",
        job_name=OUT_NAME,
        tasks=1,  # the dedup needs every source in one task
        time="02:00:00",
        qos="qos_cpu-t3",
        skip_completed=not args.force,
    )
    executor.run()
