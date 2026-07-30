from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import apply_chat_template, instruct_adapter, add_system_prompt, NemoRLFormat


def convert_to_openai_format(tool: dict) -> dict:
    """Wrap a bare ToolACE tool into the shared OpenAI function-calling schema.

    ToolACE tools are bare ({name, description, parameters}) with a flat
    parameters map (param -> {description, type, default}) and no
    {"type":"object","properties","required"} envelope. Rebuild that envelope so
    the tool definition matches the other tool-calling datasets. A param is
    marked required when its default is empty ("" means no usable default);
    non-empty defaults are kept as JSON-schema `default` and left optional.
    """
    type_map = {"int": "integer", "float": "number"}
    properties = {}
    required = []
    for param_name, param_info in tool.get("parameters", {}).items():
        ptype = param_info.get("type", "string")
        # `type` is usually a str but can be a JSON-schema union list, e.g.
        # ["string", "null"] for nullable params -- normalize each element.
        if isinstance(ptype, list):
            ptype = [type_map.get(t, t) for t in ptype]
        else:
            ptype = type_map.get(ptype, ptype)
        prop = {
            "type": ptype,
            "description": param_info.get("description", ""),
        }
        default = param_info.get("default", "")
        if default == "":
            required.append(param_name)
        else:
            prop["default"] = default
        properties[param_name] = prop

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
        tools = [convert_to_openai_format(tool) for tool in tools]
        random.shuffle(tools)
        doc.metadata["tools"] = tools
        messages = json.loads(doc.metadata.pop("conversations"))
        cleaned_messages = []
        for i, message in enumerate(messages):
            if message["content"] is None:
                message.pop("content")
            
            if message["role"] == "assistant" and cleaned_messages[-1]["role"] == "assistant":
                if "content" in message or "content" in cleaned_messages:
                    raise ValueError("content must be empty")
                cleaned_messages[-1]["tool_calls"].extend(message["tool_calls"])
                continue   

            if message["role"] == "tool":
                for tool_response in message["content"]:
                    cleaned_messages.append({"role": "tool", "content": json.dumps(tool_response)})
            else:
                cleaned_messages.append(message)
        doc.metadata["messages"] = cleaned_messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        HuggingFaceDatasetReader(
            "tryumanshow/ToolACE-Qwen-cleaned",
            {"split": "train"},
            adapter=instruct_adapter,
        ),
        format_messages,
        partial(add_system_prompt, tokenizer=tokenizer),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        JsonlWriter(
            f"{DATA_PATH}/toolace_cleaned/data",
            expand_metadata=True,
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

