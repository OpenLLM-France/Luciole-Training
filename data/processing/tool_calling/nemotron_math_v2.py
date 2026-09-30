from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    hub_adapter,
    apply_chat_template,
    instruct_adapter,
    add_system_prompt,
    NemoRLFormat,
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


THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Use this tool to think and perform reasoning before calling an API. It will not obtain new information or change the database, but just append the thought to the log.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The step by step thought process.",
                }
            },
            "required": ["thought"],
        },
    },
}


def thinking_as_tool_call(
    data,
    rank: int = 0,
    world_size: int = 1,
    thinking_as_tool: bool = False,
):
    """Turn each assistant message's reasoning trace into a call to ``think``.

    Nemotron-Math-v2 carries the reasoning in ``reasoning_content`` alongside
    the answer, which NemoRLFormat later inlines as a <think> block in the
    same assistant turn. When ``thinking_as_tool`` is set, that reasoning is
    pulled out into its own assistant turn instead: a call to the ``think``
    tool with the reasoning as its ``thought`` argument, an (empty) tool
    response, then the original message with its reasoning stripped.
    """
    import json
    import uuid

    for doc in data:
        if not thinking_as_tool:
            yield doc
            continue

        tools = doc.metadata.get("tools", [])
        if not any((t.get("function") or t).get("name") == "think" for t in tools):
            tools = tools + [THINK_TOOL]
        doc.metadata["tools"] = tools

        new_messages = []
        for message in doc.metadata["messages"]:
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if message["role"] != "assistant" or not reasoning:
                new_messages.append(message)
                continue

            call_id = f"call_{uuid.uuid4().hex[:24]}"
            new_messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "think",
                                "arguments": json.dumps({"thought": reasoning}),
                            },
                        }
                    ],
                }
            )
            new_messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": ""}
            )
            new_messages.append({**message, "reasoning_content": "", "reasoning": ""})

        doc.metadata["messages"] = new_messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--thinking_as_tool",
        action="store_true",
        help="Pull each assistant turn's reasoning out into a call to the `think` "
        "tool instead of leaving it inline in a <think> block. Written to a "
        "separate folder, so both variants can coexist.",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    # Names the output folder, so the two variants never overwrite each other.
    VARIANT = "think_as_tool" if args.thinking_as_tool else "think_inline"

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

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
        partial(thinking_as_tool_call, thinking_as_tool=args.thinking_as_tool),
        partial(add_system_prompt, tokenizer=tokenizer),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/nemotron_math_v2_tools/{VARIANT}",
            output_filename=f"data/nemotron_math_v2_tools/{VARIANT}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/nemotron_math_v2_tools/{VARIANT}/logs",
        job_name=f"nemotron_math_v2_tools_{VARIANT}",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

