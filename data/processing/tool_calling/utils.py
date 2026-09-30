import os
import sys
import importlib.util

# Directly load posttraining/utils.py under a unique module name to avoid
# a circular import, since this file is also named 'utils'. It re-exports the
# pretraining helpers and defines the shared instruct/formatting utilities.

spec = importlib.util.spec_from_file_location(
    "posttraining_utils",
    os.path.join(os.path.dirname(__file__), "..", "posttraining", "utils.py"),
)
posttraining_utils = importlib.util.module_from_spec(spec)
# Register under its module name so functions pulled from it (whose
# __module__ is "posttraining_utils") can be re-imported by name -- needed for
# dill/multiprocess to unpickle them by reference in forked worker processes.
sys.modules["posttraining_utils"] = posttraining_utils
spec.loader.exec_module(posttraining_utils)

create_parser = posttraining_utils.create_parser
parse_args = posttraining_utils.parse_args
create_executor = posttraining_utils.create_executor
add_sampler_filter = posttraining_utils.add_sampler_filter
_custom_adapter_for_hf = posttraining_utils._custom_adapter_for_hf
HF_SCHEMA = posttraining_utils.HF_SCHEMA
instruct_adapter = posttraining_utils.instruct_adapter
FilterChinese = posttraining_utils.FilterChinese
apply_chat_template = posttraining_utils.apply_chat_template
format_tool_calls = posttraining_utils.format_tool_calls
nemo_rl_format_messages = posttraining_utils.nemo_rl_format_messages
NemoRLFormat = posttraining_utils.NemoRLFormat
check_last_message = posttraining_utils.check_last_message

def from_tools_to_system(system_content, tools, tokenizer):
    import re
    system_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_content}],
        tools=tools,
        tokenize=False,
    )
    system_prompt = re.search(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
    ).group(1)
    return system_prompt

def hub_adapter(self, document):
    """Row adapter for the Hub writers: emit only the five shared columns.

    The default adapter writes every metadata key as its own column
    (``expand_metadata``), which gives each dataset a different schema -- so one
    ``load_dataset`` over the repo cannot union them -- and lets an optional
    field that is ``None`` in the first row type that column ``null``, after
    which the whole batch is rejected ("Table schema does not match schema used
    to create file"). Pinning the columns here gives every dataset in the repo
    one stable schema.

    ``dataset`` is the source id set by HuggingFaceDatasetReader; readers that
    do not set it (Parquet/Jsonl) leave it empty rather than null, so the column
    stays typed.
    """
    import json

    tools = document.metadata.get("tools")
    messages = document.metadata.get("messages") or []
    return {
        "id": str(document.id),
        "dataset": str(document.metadata.get("dataset") or ""),
        "tools": tools if isinstance(tools, str) else json.dumps(tools or []),
        "messages": [
            {"role": str(m.get("role", "")), "content": str(m.get("content") or "")}
            for m in messages
        ],
        "text": document.text or "",
    }


def add_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
    system_key=None,
):
    import json
    for doc in data:
        tools = doc.metadata.get("tools", [])
        doc.metadata["tools"] = json.dumps(tools)

        messages = doc.metadata["messages"]
        if system_key is None:
            # No explicit source declared: use the leading system message (if
            # any) as the system content and drop it, so we rebuild it with the
            # tool schemas baked in instead of leaving two system messages.
            if messages and messages[0]["role"] == "system":
                system_content = messages[0]["content"]
                messages = messages[1:]
            else:
                system_content = ""
        else:
            system_content = doc.metadata.get(system_key, "")

        system_prompt = from_tools_to_system(system_content, tools, tokenizer)
        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + messages
        yield doc
