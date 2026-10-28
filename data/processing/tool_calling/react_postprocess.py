#!/usr/bin/env python

from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    FilterChinese,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)

def id_deduplication(data, rank: int = 0, world_size: int = 1):
    dedup_ids = set()
    for doc in data:
        id = doc.id
        if id in dedup_ids:
            continue
        dedup_ids.add(id)
        yield doc

def format_messages(data, rank: int = 0, world_size: int = 1):
    import json
    import re

    for doc in data:
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "OpenLLM-France/ReAct")
        messages = doc.metadata.pop("messages")
        tools = json.loads(doc.metadata.pop("tools"))

        # The stored system message already has the tool block rendered by the
        # chat template ("# Tools\n\n..."); keep only the prompt before it, since
        # the template re-renders the tools from the `tools` column.
        if messages[0]["role"] == "system":
            system_content = re.split(
                r"(?:\n\n)?# Tools\n\nYou may call one or more functions",
                messages[0]["content"],
                maxsplit=1,
            )[0]
            messages[0] = {**messages[0], "content": system_content}

        # Last turn is kept as is (submit_answer call included), but only
        # single submit_answer calls are valid endings.
        last_content = messages[-1]["content"]
        if "<tool_call>" in last_content:
            tool_calls = [
                json.loads(match.group(1).strip())
                for match in re.finditer(
                    r"<tool_call>(.*?)</tool_call>", last_content, flags=re.DOTALL
                )
            ]
            if len(tool_calls) != 1 or tool_calls[0]["name"] != "submit_answer":
                continue

        doc.metadata["messages"] = messages
        doc.metadata["tools"] = tools
        yield doc

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the per-episode draw."
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Process the traces generated with thinking on (source folders "
             "<dataset>/<tool_type>/thinking/<lang>) instead of non_thinking",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    split = "thinking" if args.thinking else "non_thinking"

    pipeline = [
        ParquetReader(
            "hf://datasets/OpenLLM-France/ReAct/data/",
            glob_pattern=f"*/*/{split}/*/*.parquet",
            adapter=instruct_adapter,
        ),
        id_deduplication,
        format_messages,
        NemoRLFormat(),
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": args.thinking, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/react_postprocess/{split}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/react_postprocess/{split}",
            output_filename=f"data/react_postprocess/{split}/${{dataset_name}}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
            commit_message=args.commit,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/react_postprocess/{split}/logs",
        job_name=f"react_postprocess_{split}",
        tasks=1,
        time="01:00:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
