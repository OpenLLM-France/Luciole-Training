from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    apply_chat_template,
    instruct_adapter,
    check_last_message,
    add_system_prompt,
    NemoRLFormat,
    normalize_tool_schema,
)


# When2Call folds optionality into the type string ("str, optional") instead of
# into `required`. Strip that marker, keeping what it says by recording the
# parameter as optional in `required` rather than dropping the information.
OPTIONAL_SUFFIX = ", optional"


def split_optional(type_str):
    """Return ``(base_type, is_optional)`` for a When2Call type string.

    Only a *trailing* ", optional" is removed: splitting on every comma would
    mangle generics such as "Tuple[float, float]".
    """
    if not isinstance(type_str, str):
        return type_str, False
    base = type_str.strip()
    if base.lower().endswith(OPTIONAL_SUFFIX):
        return base[: -len(OPTIONAL_SUFFIX)].strip(), True
    return base, False


def drop_optional_marker(tool):
    """Remove ", optional" from a (bare) tool's property types.

    The source's own `required` wins where it exists; the marker only fills in a
    list that is absent, so the marker's meaning is not lost with it. The two
    never disagree in the data -- 1030 agreements, 0 contradictions over a
    481-tool sample -- and the 30 tools lacking `required` have every property
    marked optional, so they correctly derive an empty list.
    """
    parameters = tool.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    stripped = {}
    implied_required = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            stripped[name] = prop
            continue
        base, optional = split_optional(prop.get("type"))
        stripped[name] = {**prop, "type": base} if "type" in prop else dict(prop)
        if not optional:
            implied_required.append(name)

    required = parameters.get("required")
    if not isinstance(required, list):
        required = implied_required

    return {
        **tool,
        "parameters": {**parameters, "properties": stripped, "required": required},
    }


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    import json
    import random

    for doc in data:
        tools = doc.metadata.get("tools", [])
        tools = [normalize_tool_schema(json.loads(tool)) for tool in tools]
        tools = [drop_optional_marker(tool) for tool in tools]
        tools = [{"type": "function", "function": tool} for tool in tools]
        random.shuffle(tools)
        doc.metadata["tools"] = tools

        if any(
            "<tool_calls>" in message["content"] for message in doc.metadata["messages"]
        ):
            raise ValueError("Tool calls should not be in the messages")
        yield doc


def annotate_refusal(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        doc.metadata["refusal"] = "missing_argument"
        for word in ["sorry", "apologies", "apologize"]:
            if word in doc.metadata["messages"][-1]["content"].lower():
                doc.metadata["refusal"] = "apologies"
                break
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
            "nvidia/When2Call",
            {"name": "train_sft", "split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        annotate_refusal,
        format_messages,
        # partial(replace_tool_name, rename_names=True, rename_params=False),
        partial(add_system_prompt, tokenizer=tokenizer),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        check_last_message,
        JsonlWriter(
            f"{DATA_PATH}/when2call_oaiformat/data",
            output_filename="${refusal}/${rank}.jsonl",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/when2call_oaiformat/logs",
        job_name="when2call_oaiformat",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
