from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    instruct_adapter,
    from_tools_to_system,
    format_tool_calls,
)


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import json
    import random
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
        # Tool
        tools = doc.metadata.get("tools", [])
        tools = [json.loads(tool) for tool in tools]
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)

        # tool call
        system_prompt = from_tools_to_system("", tools, tokenizer)
        doc.metadata["tools"] = json.dumps(tools)
        chosen_response = extract_toolcall(doc.metadata["chosen_response"])
        rejected_response = extract_toolcall(doc.metadata["rejected_response"])

        doc.metadata["chosen"] = (
            [{"role": "system", "content": system_prompt}]
            + doc.metadata["messages"]
            + [chosen_response]
        )

        doc.metadata["rejected"] = (
            [{"role": "system", "content": system_prompt}]
            + doc.metadata["messages"]
            + [rejected_response]
        )
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
        partial(format_messages, tokenizer=tokenizer),
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
