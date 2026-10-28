"""Reformat the tool-calling DPO subsets of Luciole-PostTraining-Dataset-1.1.

In the published subsets (dpo_instruct: when2call, xlam-*; dpo_thinking:
smolagents-*, nemotron-corrupt-argument), the tool schemas are baked into the
system prompt. This script moves the tool definitions into a separate ``tools``
column (JSON string), to be rendered by the chat template at training time.
The part of the system prompt before the "# Tools" section is kept as system
message, unless it is just the default Luciole identity (then it is dropped).
"""

from functools import partial

from utils import (
    create_parser,
    parse_args,
    create_executor,
    instruct_adapter,
    hub_dpo_adapter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter

SOURCE = "OpenLLM-France/Luciole-PostTraining-Dataset-1.1"

SUBSETS = [
    ("dpo_instruct", "when2call"),
    ("dpo_instruct", "xlam-corrupt-arg-llm"),
    ("dpo_instruct", "xlam-corrupt-json"),
    ("dpo_instruct", "xlam-remove-argument"),
    ("dpo_instruct", "xlam-remove-tool-call"),
    ("dpo_thinking", "smolagents-corrupt-arg-llm"),
    ("dpo_thinking", "smolagents-corrupt-json"),
    ("dpo_thinking", "smolagents-remove-argument"),
    ("dpo_thinking", "nemotron-corrupt-argument"),
]


def set_dataset(
    data,
    rank: int = 0,
    world_size: int = 1,
    name: str = "",
):
    """Fill the ``dataset`` column (JsonlReader does not set it)."""
    for doc in data:
        doc.metadata["dataset"] = name
        yield doc


def extract_tools_from_system(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    """Move the tools of the chosen/rejected system message to ``tools``.

    The system prompt before the "# Tools" section is kept, except the default
    Luciole identity which is dropped (along with the system message).
    """
    import json

    IDENTITY = (
        "You are a helpful AI assistant named Luciole, "
        "trained by LINAGORA and OpenLLM France."
    )
    TOOLS_PREFIX = (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
    )
    TOOLS_SUFFIX = (
        "\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )

    def parse_system(content):
        """Return (system prompt or None, tools)."""
        if TOOLS_PREFIX not in content:
            system, tools = content, []  # e.g. when2call samples with no tool
        else:
            system, block = content.split(TOOLS_PREFIX, 1)
            if not block.endswith(TOOLS_SUFFIX):
                raise ValueError(f"Unexpected tools section: {block[-300:]!r}")
            block = block[: -len(TOOLS_SUFFIX)]
            tools = [json.loads(line) for line in block.split("\n") if line.strip()]
        system = system.strip()
        if system in ("", IDENTITY):
            system = None
        return system, tools

    for doc in data:
        tools = None
        for key in ("chosen", "rejected"):
            messages = doc.metadata[key]
            if messages[0]["role"] != "system":
                raise ValueError(f"{doc.id}: {key} does not start with a system message")
            system, key_tools = parse_system(messages[0]["content"])
            if tools is not None and key_tools != tools:
                raise ValueError(f"{doc.id}: chosen and rejected have different tools")
            tools = key_tools
            doc.metadata[key] = (
                [{"role": "system", "content": system}] if system else []
            ) + messages[1:]
        doc.metadata["tools"] = json.dumps(tools, ensure_ascii=False)
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    for split, subset in SUBSETS:
        pipeline = [
            JsonlReader(
                f"hf://datasets/{SOURCE}/{split}/{subset}",
                glob_pattern="*.jsonl",
                adapter=instruct_adapter,
            ),
            partial(set_dataset, name=f"{SOURCE}/{split}/{subset}"),
            extract_tools_from_system,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/tool_data_dpo" + "_debug" * args.debug,
                local_working_dir=f"{DATA_PATH}/luciole_dpo_tools/{split}/{subset}",
                output_filename=f"data/{split}/{subset}/${{rank}}.parquet",
                adapter=hub_dpo_adapter,
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
            logging_dir=f"{DATA_PATH}/luciole_dpo_tools/{split}/{subset}/logs",
            job_name=f"luciole_dpo_tools_{subset}",
            tasks=1,
            time="00:30:00",
            # partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
