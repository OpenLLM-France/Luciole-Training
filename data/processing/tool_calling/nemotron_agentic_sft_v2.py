from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
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
    import random

    for doc in data:
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "nvidia/Nemotron-SFT-Agentic-v2")
        messages = doc.metadata["messages"]
        tools = doc.metadata.get("tools", [])
        random.shuffle(tools)

        # Clean tool response
        for message in messages:
            if message["role"] == "tool":
                message["content"] = message["content"].strip()

        doc.metadata["messages"] = messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path


    for split_name in ["interactive_agent", "search", "tool_calling"]:
        pipeline = [
            JsonlReader(
                "hf://datasets/nvidia/Nemotron-SFT-Agentic-v2/data/",
                glob_pattern=f"{split_name}.jsonl",
                adapter=instruct_adapter,
            ),
            format_messages,
            NemoRLFormat(),
            ChatTokenCount(
                "OpenLLM-France/tokenizer_128k_dual",
                tools_key="tools",
                chat_template_kwargs={"enable_thinking": True, "training_mode": True},
                max_tokens=2**16,
                exclusion_writer=JsonlWriter(f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/too_long"),
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
                local_working_dir=f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}",
                output_filename=f"data/nemotron_agentic_sft_v2/{split_name}/${{rank}}.parquet",
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
            logging_dir=f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/logs",
            job_name=f"agentic_v2_{split_name}",
            tasks=1,
            time="00:30:00",
            # partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
