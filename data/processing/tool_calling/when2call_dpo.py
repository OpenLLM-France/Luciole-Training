from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    instruct_adapter,
    add_system_prompt,
    format_tool_calls,
    normalize_tool_schema,
)
from when2call import drop_optional_marker


def format_tools(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        tools = doc.metadata.get("tools", [])
        tools = [normalize_tool_schema(json.loads(tool)) for tool in tools]
        tools = [drop_optional_marker(tool) for tool in tools]
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)
        # Left as a list of dicts: add_system_prompt bakes them into the system
        # message and json.dumps them for a load_dataset-friendly column.
        doc.metadata["tools"] = tools
        yield doc


def build_pairs(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    """Fork the prompt into a chosen and a rejected conversation.

    ``messages`` already carries the system message with the tool schemas baked
    in, courtesy of add_system_prompt.
    """
    import json
    import re

    def extract_toolcall(message):
        content = message["content"]
        pattern = "<TOOLCALL>(.*)</TOOLCALL>"
        # extract the pattern and remove from content
        match = re.search(pattern, content, re.DOTALL)
        if match:
            tool_calls = json.loads(match.group(1))
            content = content.replace(match.group(0), "").strip()
        else:
            tool_calls = []
        message["content"] = format_tool_calls(tool_calls, content)
        return message

    for doc in data:
        prompt = doc.metadata["messages"]
        doc.metadata["chosen"] = prompt + [
            extract_toolcall(doc.metadata["chosen_response"])
        ]
        doc.metadata["rejected"] = prompt + [
            extract_toolcall(doc.metadata["rejected_response"])
        ]
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        HuggingFaceDatasetReader(
            "nvidia/When2Call",
            {"name": "train_pref", "split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        format_tools,
        partial(add_system_prompt, tokenizer=tokenizer),
        build_pairs,
        JsonlWriter(
            f"{DATA_PATH}/when2call_dpo/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/when2call_dpo/logs",
        job_name="when2call_dpo",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
