from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import SamplerFilter
from utils import instruct_adapter

def nemo_rl_format(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import warnings
    ALLOWED_KEYS = {"role", "content"}
    warned_keys = set()

    def format_tool_calls(tool_calls: list, content: str = "") -> str:
        parts = []

        for i, tool_call in enumerate(tool_calls):
            if (i == 0 and content) or (i > 0):
                parts.append("\n")

            if isinstance(tool_call, dict) and "function" in tool_call:
                tool_call = tool_call["function"]

            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            parts.append(
                f'<tool_call>\n{{"name": "{tool_call["name"]}", "arguments": {arguments}}}\n</tool_call>'
            )

        return content + "".join(parts)

    for doc in data:
        messages = doc.metadata["messages"]
        for message in messages:
            if "tool_calls" in message and message["tool_calls"] is not None:
                content = message.get("content") or ""
                message["content"] = format_tool_calls(message.pop("tool_calls"), content)

            if "reasoning_content" in message and message["reasoning_content"] is not None:
                content = message.get("content") or ""
                message["content"] = (
                    "<think>\n"
                    + message.pop("reasoning_content").strip("\n")
                    + "\n</think>\n\n"
                    + content.lstrip("\n")
                )

            # Keys must be in role, content
            extra_keys = set(message.keys()) - ALLOWED_KEYS
            new_keys = extra_keys - warned_keys
            if new_keys:
                warned_keys.update(new_keys)
                warnings.warn(
                    f"Message contains unexpected keys: {new_keys}. "
                    f"Allowed keys are: {ALLOWED_KEYS}. "
                    f"Message role: {message.get('role')!r}"
                )

        doc.metadata = {"messages": messages}
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "input_path",
        type=str
    )
    parser.add_argument(
        "output_path",
        type=str
    )
    parser.add_argument(
        "--global_pattern",
        type=str
    )
    parser.add_argument(
        "--rate",
        default=1.,
        type=float
    )
    args = parse_args(parser)

    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.global_pattern,
            adapter=instruct_adapter,
        ),
        SamplerFilter(rate=args.rate, seed=42),
        nemo_rl_format,
        JsonlWriter(
            f"{args.output_path}/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_path}/logs",
        job_name=f"nemo_format",
        tasks=10,
        time="00:30:00",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )

    main_processing_executor.run()

# python downsample_dataset.py  
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/pleais_rag/openrag_thinking 
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/pleais_rag/openrag_thinking_downsample 
# --glob "data/en/*.jsonl.gz" --rate 0.125

# python downsample_dataset.py \
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/nemotron_agentic_sft_v2 \
# /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/nemotron_agentic_sft_v2_downsample \
# --glob "*/data/*.jsonl.gz" --rate 0.1 