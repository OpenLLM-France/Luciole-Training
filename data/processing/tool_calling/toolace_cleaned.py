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

    for doc in data:
        tools = doc.metadata.get("tools", "[]")
        tools = json.loads(tools)
        random.shuffle(tools)
        doc.metadata["tools"] = tools
        messages = json.loads(doc.metadata.pop("conversations"))
        cleaned_messages = []
        for i, message in enumerate(messages):
            if message["content"] is None:
                message.pop("content")
            
            if message["role"] == "assistant" and cleaned_messages[-1]["role"] == "assistant":
                if "content" in message or "content" in cleaned_messages:
                    raise ValueError("content must be empty")
                cleaned_messages[-1]["tool_calls"].extend(message["tool_calls"])
                continue   

            if message["role"] == "tool":
                for tool_response in message["content"]:
                    cleaned_messages.append({"role": "tool", "content": json.dumps(tool_response)})
            else:
                cleaned_messages.append(message)
        doc.metadata["messages"] = cleaned_messages
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
            "tryumanshow/ToolACE-Qwen-cleaned",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer),
        partial(apply_chat_template, tokenizer=tokenizer),
        JsonlWriter(
            f"{DATA_PATH}/toolace_cleaned/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/toolace_cleaned/logs",
        job_name="toolace_cleaned",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

