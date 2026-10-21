from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    NemoRLFormat,
    ChatTokenCount,
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
        # Keep the dataset's own system prompt as-is (tools are not baked in)
        system = doc.metadata.pop("system", None)
        if system:
            messages = [{"role": "system", "content": system}] + messages
        doc.metadata["messages"] = messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "prem-research/Funcdex-MT-Function-Calling",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        NemoRLFormat(),
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/funcdex_mt/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/funcdex_mt",
            output_filename="data/funcdex_mt/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/funcdex_mt/logs",
        job_name="funcdex_mt",
        tasks=1,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()

