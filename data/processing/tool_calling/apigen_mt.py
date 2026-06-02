from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import apply_chat_template, instruct_adapter, add_system_prompt

def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    def _format_messages(messages):
        role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "observation": "tool",
            "function_call": "assistant",
        }

        return [
            {"role": role_map.get(turn["from"]), "content": turn["value"]}
            for turn in messages
        ]
    
    for doc in data:
        # Process tools
        tools = doc.metadata.get("tools", "[]")
        tools = json.loads(tools)
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)
        doc.metadata["tools"] = tools
        # Process messages
        messages = doc.metadata.pop("conversations")
        doc.metadata["messages"] = _format_messages(messages)
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
            "Salesforce/APIGen-MT-5k",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer, system_key="system"),
        partial(apply_chat_template, tokenizer=tokenizer),
        JsonlWriter(
            f"{DATA_PATH}/apigen_mt/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/apigen_mt/logs",
        job_name="apigen_mt",
        tasks=1,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

