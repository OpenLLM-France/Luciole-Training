from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    FilterChinese,
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
)


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    def convert_to_openai_format(tool: dict) -> dict:
        """
        Converts a custom tool schema to OpenAI function calling format.

        Args:
            tool: Dictionary with name, description, and parameters fields

        Returns:
            OpenAI-compatible tool definition
        """
        properties = {}
        required = []

        for param_name, param_info in tool.get("parameters", {}).items():
            prop = {
                "type": param_info.get("type", "string")
                .split(",")[0]
                .strip(),  # handle "str, optional"
                "description": param_info.get("description", ""),
            }

            # Add default if present
            if "default" in param_info:
                prop["default"] = param_info["default"]

            properties[param_name] = prop

            # Mark as required only if not optional
            if "optional" not in param_info.get("type", ""):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    for doc in data:
        tools = doc.metadata.get("tools", None)
        tools = json.loads(tools)
        # Format argument as OpenAI
        tools = [convert_to_openai_format(tool) for tool in tools]
        if tools:
            random.shuffle(tools)
        doc.metadata["tools"] = tools

        doc.metadata["messages"] = [
            {"role": "user", "content": doc.metadata.pop("query")},
            {
                "role": "assistant",
                "tool_calls": json.loads(doc.metadata.pop("answers")),
            },
        ]
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        HuggingFaceDatasetReader(
            "Salesforce/xlam-function-calling-60k",
            {"split": "train"},
            # streaming=True,
            adapter=instruct_adapter,
        ),
        format_messages,
        # partial(replace_tool_name, rename_names=True, rename_params=False),
        partial(add_system_prompt, tokenizer=tokenizer),
        partial(apply_chat_template, tokenizer=tokenizer),
        check_last_message,
        FilterChinese(
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/xlam/chinese_heavy"),
        ),
        JsonlWriter(
            f"{DATA_PATH}/xlam_oaiformat/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/xlam_oaiformat/logs",
        job_name="xlam_oaiformat",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
