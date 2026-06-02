from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
)


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        tools = doc.metadata.get("tools", [])
        tools = [json.loads(tool) for tool in tools]
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)
        doc.metadata["tools"] = tools

        if any(
            "<tool_calls>" in message["content"] for message in doc.metadata["messages"]
        ):
            raise ValueError("Tool calls should not be in the messages")
        yield doc


def annotate_refusal(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        doc.metadata["refusal"] = "missing_argument"
        for word in ["sorry", "apologies", "apologize"]:
            if word in doc.metadata["messages"][-1]["content"].lower():
                doc.metadata["refusal"] = "apologies"
                break
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        HuggingFaceDatasetReader(
            "nvidia/When2Call",
            {"name": "train_sft", "split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        annotate_refusal,
        format_messages,
        # partial(replace_tool_name, rename_names=True, rename_params=False),
        partial(add_system_prompt, tokenizer=tokenizer),
        partial(apply_chat_template, tokenizer=tokenizer),
        check_last_message,
        JsonlWriter(
            f"{DATA_PATH}/when2call_oaiformat/data",
            output_filename="${refusal}/${rank}.jsonl",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/when2call_oaiformat/logs",
        job_name="when2call_oaiformat",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
