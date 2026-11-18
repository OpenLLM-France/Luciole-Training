from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from utils import (
    hub_adapter,
    instruct_adapter,
    NemoRLFormat,
    ChatTokenCount,
)

class ParquetReaderFromRow(ParquetReader):
    """ParquetReader that starts reading at ``start_row``, whole row groups at a time.

    datatrove's own ``skip`` is applied by the executor, after ``read_file`` has
    already pulled every row off the wire and built a Document for it, so it saves
    no reading time. Here the row groups that end before ``start_row`` are never
    fetched. ``li`` (the document index the adapter falls back on for the id) keeps
    counting from the real row number, so ids do not shift when ``start_row`` does.
    """

    def __init__(self, *args, start_row: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_row = start_row

    def read_file(self, filepath: str):
        import pyarrow.parquet as pq

        with self.data_folder.open(filepath, "rb") as f:
            with pq.ParquetFile(f) as pqf:
                # First row group that can hold start_row; row groups are only
                # skipped whole, so reading starts at or before it.
                first_rg, li = 0, 0
                while first_rg < pqf.num_row_groups:
                    n_rows = pqf.metadata.row_group(first_rg).num_rows
                    if li + n_rows > self.start_row:
                        break
                    li += n_rows
                    first_rg += 1

                columns = [self.text_key, self.id_key] if not self.read_metadata else None
                for batch in pqf.iter_batches(
                    batch_size=self.batch_size,
                    row_groups=range(first_rg, pqf.num_row_groups),
                    columns=columns,
                ):
                    documents = []
                    with self.track_time("batch"):
                        for line in batch.to_pylist():
                            # Row groups are skipped whole, so the first one read
                            # still holds the rows before start_row: drop those.
                            if li < self.start_row:
                                li += 1
                                continue
                            document = self.get_document_from_dict(line, filepath, li)
                            li += 1
                            if not document:
                                continue
                            documents.append(document)
                    yield from documents


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "nvidia/Nemotron-Math-v2")
        tools = doc.metadata.get("tools", [])
        if len(tools) == 0:
            continue
        random.shuffle(tools)
        yield doc


# Rows of data/sft/nemotron_math_v2_tools/think_inline on the hub (written by this
# script before its think_as_tool option was removed), after the too_long filter,
# counted on 2026-11-18: with / without a "tool" turn.
N_WITH_TOOL_TURN = 268_923
N_WITHOUT_TOOL_TURN = 412_980


def subsample_no_tool(
    data,
    rank: int = 0,
    world_size: int = 1,
    no_tool_frac: float | None = None,
    seed: int = 42,
):
    """Keep every conversation with a ``tool`` turn and a random part of the others,
    so that those without one make up about ``no_tool_frac`` of the output.

    Nemotron-Math-v2's tool rows often answer without calling the tool (60.6% of
    them). The keep probability of a row without a tool turn is
    ``no_tool_frac / (1 - no_tool_frac) * N_WITH_TOOL_TURN / N_WITHOUT_TOOL_TURN``;
    it is drawn from a hash of the id, so the same rows are kept on every run.
    """
    import hashlib

    if no_tool_frac is None:
        yield from data
        return
    keep_prob = min(
        1.0, no_tool_frac / (1 - no_tool_frac) * N_WITH_TOOL_TURN / N_WITHOUT_TOOL_TURN
    )
    for doc in data:
        if any(m.get("role") == "tool" for m in doc.metadata["messages"]):
            yield doc
            continue
        h = hashlib.md5(f"{seed}:{doc.id}".encode()).digest()
        if int.from_bytes(h[:8], "big") / 2**64 < keep_prob:
            yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--no_tool_frac",
        type=float,
        default=None,
        help="If set (e.g. 0.1), subsample the conversations without a tool turn "
        "so they are about this fraction of the output (all the ones with a tool "
        "turn are kept). Written to a separate folder (nemotron_math_v2_tools_notool<pct>).",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    # Names the output folder, so a subsampled run never overwrites the full one.
    NAME = "nemotron_math_v2_tools"
    if args.no_tool_frac is not None:
        NAME += f"_notool{round(100 * args.no_tool_frac)}"


    pipeline = [
        ParquetReaderFromRow(
            "hf://datasets/nvidia/Nemotron-Math-v2",
            glob_pattern="data/low.parquet",
            # The file is sorted with the tool-using rows last: the first
            # 1,036,256 of its 1,718,159 rows have an empty `tools` list and are
            # dropped by format_messages below. Start at the first of the rows
            # that do have tools instead of reading them. Re-check this offset if
            # nvidia re-uploads the file -- starting too late silently drops data.
            start_row=1_036_256,
            adapter=instruct_adapter,
        ),
        format_messages,
        NemoRLFormat(),
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/{NAME}/too_long"),
        ),
        partial(subsample_no_tool, no_tool_frac=args.no_tool_frac),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/{NAME}",
            output_filename=f"data/sft/{NAME}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            commit_message=args.commit,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{NAME}/logs",
        job_name=f"{NAME}",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

