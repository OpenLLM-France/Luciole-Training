from utils import create_parser, parse_args, create_executor, hub_adapter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import instruct_adapter, NemoRLFormat

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
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)
        doc.metadata["tools"] = tools
        messages = json.loads(doc.metadata.pop("conversations"))
        cleaned_messages = []
        for i, message in enumerate(messages):
            if message["content"] is None:
                message.pop("content")

            if (
                message["role"] == "assistant"
                and cleaned_messages[-1]["role"] == "assistant"
            ):
                if "content" in message or "content" in cleaned_messages:
                    raise ValueError("content must be empty")
                cleaned_messages[-1]["tool_calls"].extend(message["tool_calls"])
                continue

            if message["role"] == "tool":
                for tool_response in message["content"]:
                    cleaned_messages.append(
                        {"role": "tool", "content": json.dumps(tool_response)}
                    )
            else:
                cleaned_messages.append(message)
        doc.metadata["messages"] = cleaned_messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path


    pipeline = [
        HuggingFaceDatasetReader(
            "tryumanshow/ToolACE-Qwen-cleaned",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        NemoRLFormat(),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/toolace_cleaned",
            output_filename="data/toolace_cleaned/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/toolace_cleaned/logs",
        job_name="toolace_cleaned",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
