from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    NemoRLFormat,
    ChatTokenCount,
    ToolTypeCheck,
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


SPLITS = ["interactive_agent", "search", "tool_calling"]

# Fraction of a split kept (SamplerFilter, before the slow token count). tool_calling (707k rows) is used at weight
# 0.01454 in the sft_mix: a 2% sample is enough (sft_mix.py subsamples it at 0.01454 / 0.02).
SAMPLE_RATES = {"tool_calling": 0.02}


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS, help="Splits to process.")
    args = parse_args(parser)
    DATA_PATH = args.data_path


    for split_name in args.splits:
        pipeline = [
            JsonlReader(
                "hf://datasets/nvidia/Nemotron-SFT-Agentic-v2/data/",
                glob_pattern=f"{split_name}.jsonl",
                adapter=instruct_adapter,
            ),
            *([SamplerFilter(rate=SAMPLE_RATES[split_name], seed=42)] if split_name in SAMPLE_RATES else []),
            format_messages,
            # Drop rows whose tool definitions / tool-call arguments have invalid JSON-Schema types
            # (error -> retry self-corrections are kept); dropped rows are saved for inspection.
            ToolTypeCheck(
                exclusion_writer=JsonlWriter(f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/bad_types"),
            ),
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
                output_filename=f"data/sft/nemotron_agentic_sft_v2/{split_name}/${{rank}}.parquet",
                adapter=hub_adapter,
                schema=None,
                private=True,
                cleanup=False,
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
