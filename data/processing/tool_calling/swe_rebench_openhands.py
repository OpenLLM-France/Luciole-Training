from utils import create_parser, parse_args, create_executor
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.writers.disk_base import DiskWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)


def drop_nulls(obj):
    """Remove the None fields parquet adds when unioning struct schemas.

    The five OpenHands tools share a single arrow struct, so every tool schema
    comes back with the other tools' parameters set to None, and every message
    with the fields of the other roles (`tool_calls` on a tool message, ...).
    """
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [drop_nulls(v) for v in obj]
    return obj


def convert_property(prop: dict) -> dict:
    ordered = {key: prop[key] for key in ("type", "description") if key in prop}
    ordered.update({k: v for k, v in prop.items() if k not in ordered})
    return ordered


def convert_to_openai_format(tool: dict) -> dict:
    """Rebuild a tool schema with the key order the other datasets use.

    Arrow hands the schemas back alphabetically (`description` before `name`,
    `properties` before `type`); same fix as in jupyter_agent.py.
    """
    function = tool["function"]
    parameters = drop_nulls(function.get("parameters", {}))
    properties = parameters.get("properties", {})
    return {
        "type": "function",
        "function": {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": {
                "type": parameters.get("type", "object"),
                "properties": {
                    name: convert_property(prop)
                    for name, prop in properties.items()
                },
                "required": parameters.get("required", []),
            },
        },
    }


class SWERebenchFormat(BaseFilter):
    name = "SWERebenchFormat"

    def __init__(self, keep_unresolved: bool = False, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.keep_unresolved = keep_unresolved

    def filter(self, doc: Document):
        import json
        import random

        doc.metadata.setdefault("dataset", "nebius/SWE-rebench-openhands-trajectories")
        # The final diff is already in the last tool calls; we never use it.
        doc.metadata.pop("model_patch", None)

        # Unresolved trajectories (~52%) are kept in the exclusion dir: they
        # are the rejected side if we ever build DPO pairs on `instance_id`.
        if not doc.metadata["resolved"] and not self.keep_unresolved:
            return False, "unresolved"

        tools = [convert_to_openai_format(tool) for tool in doc.metadata["tools"]]
        random.shuffle(tools)
        doc.metadata["tools"] = tools

        messages = [drop_nulls(message) for message in doc.metadata.pop("trajectory")]
        for message in messages:
            if message["role"] == "tool":
                message["content"] = message["content"].strip()
            if message.get("tool_calls"):
                # Content ends with "\n\n" before the call, and
                # format_tool_calls adds its own "\n" separator.
                message["content"] = (message.get("content") or "").rstrip()
            for tool_call in message.get("tool_calls", []):
                # Stored as a JSON string: parse it, otherwise format_tool_calls
                # inlines it verbatim (see the nemotron_agentic_sft_v2 issue in
                # the README).
                try:
                    tool_call["function"]["arguments"] = json.loads(
                        tool_call["function"]["arguments"]
                    )
                except json.JSONDecodeError:
                    return False, "arguments_parsing_error"

        doc.metadata["messages"] = messages
        return True


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--keep_unresolved",
        action="store_true",
        help="Also keep the trajectories that did not resolve the issue "
             "(written under all/ instead of resolved/)",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    split = "all" if args.keep_unresolved else "resolved"
    out_dir = f"{DATA_PATH}/swe_rebench_openhands/{split}"

    pipeline = [
        ParquetReader(
            "hf://datasets/nebius/SWE-rebench-openhands-trajectories/",
            glob_pattern="trajectories.parquet",
            id_key="trajectory_id",
            # Trajectories average ~64 turns and ~220k characters: keep the
            # batches small to bound memory.
            batch_size=100,
            adapter=instruct_adapter,
        ),
        SWERebenchFormat(
            keep_unresolved=args.keep_unresolved,
            exclusion_writer=JsonlWriter(f"{out_dir}/excluded"),
        ),
        check_last_message,
        NemoRLFormat(),
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{out_dir}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=out_dir,
            output_filename=f"data/swe_rebench_openhands/{split}/${{rank}}.parquet",
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
        limit_debug=args.limit_debug,
        logging_dir=f"{out_dir}/logs",
        job_name=f"swe_rebench_openhands_{split}",
        # A single parquet file: datatrove shards by file, so one task.
        tasks=1,
        time="02:00:00",
        qos="qos_cpu-t3",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
