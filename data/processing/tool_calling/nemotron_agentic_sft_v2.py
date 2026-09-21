from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    apply_chat_template,
    instruct_adapter,
    add_system_prompt,
    NemoRLFormat,
)


# Artifacts of NVIDIA's generation harness that carry no schema meaning:
# `strict` is an OpenAI *API* enforcement flag, `title` a Pydantic auto-label
# restating the key name ("bidder_id" -> "Bidder Id"), `$schema` document
# metadata. The chat template json.dumps whatever it is handed straight into the
# <tools> block, so keeping them would make this the only corpus subset showing
# the model that dialect -- and BFCL's own schemas carry none of the three.
#
# Everything that constrains a value is kept, `additionalProperties` included:
# it is real JSON Schema (hermes uses it as {"type": "string"} to type a
# free-form map's values), not harness noise.
SCHEMA_NOISE_KEYS = {"strict", "title", "$schema"}


def _strip_schema_node(node):
    """Drop SCHEMA_NOISE_KEYS from one schema node, recursing structurally.

    Only schema *keywords* are dropped. The keys under `properties` are
    parameter names, not keywords, so that level is never filtered -- a tool
    with a parameter literally named "title" keeps it.
    """
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in SCHEMA_NOISE_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: _strip_schema_node(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _strip_schema_node(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            out[key] = [_strip_schema_node(v) for v in value]
        else:
            out[key] = value
    return out


def strip_schema_noise(tool):
    """Drop the harness artifacts from one tool schema."""
    if not isinstance(tool, dict):
        return tool
    wrapped = isinstance(tool.get("function"), dict)
    fn = tool["function"] if wrapped else tool
    cleaned = {k: v for k, v in fn.items() if k not in SCHEMA_NOISE_KEYS}
    if isinstance(cleaned.get("parameters"), dict):
        cleaned["parameters"] = _strip_schema_node(cleaned["parameters"])
    return {**tool, "function": cleaned} if wrapped else cleaned


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import random

    for doc in data:
        messages = doc.metadata["messages"]
        tools = doc.metadata.get("tools", [])
        random.shuffle(tools)
        doc.metadata["tools"] = [strip_schema_noise(tool) for tool in tools]

        # Clean tool response
        for message in messages:
            if message["role"] == "tool":
                message["content"] = message["content"].strip()

        # add_system_prompt consumes the leading system message (if any) and
        # rebuilds it with the tool schemas baked in.
        doc.metadata["messages"] = messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    for split_name in ["interactive_agent", "search", "tool_calling"]:
        pipeline = [
            JsonlReader(
                "hf://datasets/nvidia/Nemotron-SFT-Agentic-v2/data/",
                glob_pattern=f"{split_name}.jsonl",
                adapter=instruct_adapter,
            ),
            format_messages,
            partial(add_system_prompt, tokenizer=tokenizer),
            NemoRLFormat(),
            partial(apply_chat_template, tokenizer=tokenizer),
            JsonlWriter(
                f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/data",
                expand_metadata=True,
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
