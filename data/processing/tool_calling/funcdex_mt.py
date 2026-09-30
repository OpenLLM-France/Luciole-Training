from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
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

def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        # Process tools
        tools = doc.metadata.get("tools", "[]")
        tools = json.loads(tools)
        random.shuffle(tools)
        doc.metadata["tools"] = tools
        # Process messages
        messages = []
        for message in doc.metadata.pop("conversations"):
            if message["tool_calls"] is not None:
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        doc.metadata["messages"] = messages
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
            "prem-research/Funcdex-MT-Function-Calling",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer, system_key="system"),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/funcdex_mt",
            output_filename="data/funcdex_mt/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/funcdex_mt/logs",
        job_name="funcdex_mt",
        tasks=1,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

