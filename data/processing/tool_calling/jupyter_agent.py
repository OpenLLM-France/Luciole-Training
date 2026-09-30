from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    hub_adapter,
    FilterChinese,
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
    NemoRLFormat,
)

# The single user message opens with the environment description (available
# files, installed packages) and only then states the task. We cut there to get
# a proper system/user split instead of one giant user turn.
SYSTEM_USER_SPLIT = "Answer the following question based on the provided files:"

def drop_nulls(obj):
    """Remove the None fields parquet adds when unioning struct schemas.

    The two tools (`add_and_execute_jupyter_code_cell` taking `code` and
    `final_answer` taking `answer`) share a single arrow struct, so every
    tool schema and every tool call comes back with the other tool's
    argument set to None.
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

    Arrow hands the schemas back alphabetically (`description` before
    `name`, `properties` before `type`), but the tools are json.dumps'd
    verbatim into the system prompt, so the layout the model sees here has
    to match the one it sees everywhere else -- see the produced xlam data
    and the inference-time tools in react_tools.py.
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


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import random


    warn_unexpected_first_message = True
    for doc in data:
        # The full source Kaggle notebook, up to several MB per row. It is 99%
        # of the dataset's weight and we never use it.
        # HuggingFaceDatasetReader sets this; the Parquet/Jsonl readers do not.
        doc.metadata.setdefault("dataset", "jupyter-agent/jupyter-agent-dataset")
        doc.metadata.pop("original_notebook", None)

        tools = [
            convert_to_openai_format(tool)
            for tool in doc.metadata.get("tools", [])
        ]
        random.shuffle(tools)
        # Leave tools as a list of dicts; add_system_prompt bakes them into the
        # system message and json.dumps them for a load_dataset-friendly column.
        doc.metadata["tools"] = tools

        messages = [drop_nulls(message) for message in doc.metadata["messages"]]

        first = messages[0]
        if first["role"] == "user" and SYSTEM_USER_SPLIT in first["content"]:
            preamble, task = first["content"].split(SYSTEM_USER_SPLIT, 1)
            messages = [
                {"role": "system", "content": preamble.strip()},
                {"role": "user", "content": (SYSTEM_USER_SPLIT + task).strip()},
            ] + messages[1:]
        elif warn_unexpected_first_message:
            print(
                f"Warning: Document {doc.id} does not start with the expected "
                f"environment preamble. Keeping its messages as-is."
            )
            warn_unexpected_first_message = False  # Only warn once

        doc.metadata["messages"] = messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Process the 'thinking' split (traces whose assistant turns carry "
             "<think> blocks) instead of 'non_thinking'",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    split = "thinking" if args.thinking else "non_thinking"

    pipeline = [
        ParquetReader(
            "hf://datasets/jupyter-agent/jupyter-agent-dataset/data/",
            glob_pattern=f"{split}-*.parquet",
            # Rows carry the whole source notebook, so they average ~1MB: keep
            # the batches small to bound memory.
            batch_size=50,
            adapter=instruct_adapter,
        ),
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer),
        check_last_message,
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        FilterChinese(
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/jupyter_agent/{split}/chinese_heavy"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/jupyter_agent/{split}",
            output_filename=f"data/jupyter_agent/{split}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{DATA_PATH}/jupyter_agent/{split}/logs",
        job_name=f"jupyter_agent_{split}",
        tasks=16,
        time="02:00:00",
        qos="qos_cpu-t3",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
