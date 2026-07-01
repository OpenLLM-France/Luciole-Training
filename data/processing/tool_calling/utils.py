import os
import importlib.util

# Directly load posttraining/utils.py under a unique module name to avoid
# a circular import, since this file is also named 'utils'. It re-exports the
# pretraining helpers and defines the shared instruct/formatting utilities.

spec = importlib.util.spec_from_file_location(
    "posttraining_utils",
    os.path.join(os.path.dirname(__file__), "..", "posttraining", "utils.py"),
)
posttraining_utils = importlib.util.module_from_spec(spec)
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


def check_last_message(data, rank: int = 0, world_size: int = 1, tokenizer=None):
    raise_last_message_warning = True
    for doc in data:
        last_idx = max(
            i
            for i, m in enumerate(doc.metadata["messages"])
            if m["role"] == "assistant"
        )
        if (
            last_idx < len(doc.metadata["messages"]) - 1
        ) and raise_last_message_warning:
            print(
                f"Warning: Document {doc.id} has messages after the last assistant message. Truncating."
            )
            print(f"Last message was: {doc.metadata['messages'][-1]}")
            raise_last_message_warning = False  # Only warn once
        doc.metadata["messages"] = doc.metadata["messages"][: last_idx + 1]
        yield doc


def replace_tool_name(
    data,
    rank: int = 0,
    world_size: int = 1,
    rename_names: bool = True,
    rename_params: bool = True,
):
    if rename_params:
        raise NotImplementedError("Parameter/argument renaming is currently disabled.")

    def random_name(m, M):
        import random
        import string

        letters = (
            list(string.ascii_uppercase)
            + list(string.ascii_lowercase)
            + ["_", "."]
            + list(map(str, range(10)))
        )
        return "".join(random.choices(letters, k=random.randint(m, M)))

    def _convert_tool_names(doc):
        tools = doc.metadata["tools"]

        for tool in tools:
            if "function" not in tool:
                raise KeyError(f"Tool missing 'function' key: {tool}")

        # Only rename a tool if its definition carries a description
        tool_name_mapping = (
            {
                tool["function"]["name"]: random_name(5, 15)
                for tool in tools
                if tool["function"].get("description")
            }
            if rename_names
            else {}
        )
        # # Per-tool parameter name mapping, keyed by ORIGINAL tool name.
        # # Only rename a param if its definition carries a description.
        # param_name_mapping = (
        #     {
        #         tool["function"]["name"]: {
        #             param: random_name(4, 10)
        #             for param, param_def in tool["function"].get("parameters", {}).items()
        #             if isinstance(param_def, dict) and param_def.get("description")
        #         }
        #         for tool in tools
        #     }
        #     if rename_params
        #     else {}
        # )

        doc.metadata["tool_name_mapping"] = tool_name_mapping
        # doc.metadata["param_name_mapping"] = param_name_mapping

        # Map tool names (param renaming temporarily disabled)
        for tool in tools:
            old_tool_name = tool["function"]["name"]
            # if rename_params:
            #     params = tool["function"].get("parameters", {})
            #     args_map = param_name_mapping.get(old_tool_name, {})
            #     tool["function"]["parameters"] = {
            #         args_map.get(old_param, old_param): param_def
            #         for old_param, param_def in params.items()
            #     }
            if rename_names and old_tool_name in tool_name_mapping:
                tool["function"]["name"] = tool_name_mapping[old_tool_name]
        doc.metadata["tools"] = tools

        # Map tool names and argument keys in messages (assistant tool_calls and system prompt tool definitions)
        messages = doc.metadata["messages"]
        for message in messages:
            # Update tool call names and argument keys in assistant messages
            if message["role"] == "assistant":
                if "<tool_call>" in (message.get("content") or ""):
                    raise ValueError(
                        f"Found '<tool_call>' tag in assistant content for doc {doc.id}; "
                        "tool calls must be parsed into the structured 'tool_calls' field upstream."
                    )
            if message["role"] == "assistant" and "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    old_tool_name = tool_call["name"]
                    # if rename_params and "arguments" in tool_call and old_tool_name in param_name_mapping:
                    #     args_map = param_name_mapping[old_tool_name]
                    #     tool_call["arguments"] = {
                    #         args_map.get(k, k): v
                    #         for k, v in tool_call["arguments"].items()
                    #     }
                    if rename_names and old_tool_name in tool_name_mapping:
                        tool_call["name"] = tool_name_mapping[old_tool_name]

            # Update tool names embedded in the system prompt (as JSON/XML string)
            if rename_names and message["role"] == "system" and "content" in message:
                content = message["content"]
                for old_name, new_name in tool_name_mapping.items():
                    content = content.replace(
                        f'"name": "{old_name}"', f'"name": "{new_name}"'
                    )
                    content = content.replace(
                        f'"name": {old_name}', f'"name": {new_name}'
                    )
                message["content"] = content

        doc.metadata["messages"] = messages
        return doc

    for doc in data:
        doc = _convert_tool_names(doc)
        yield doc


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
        
        system_prompt = from_tools_to_system(
            doc.metadata.get(system_key, ""),
            tools,
            tokenizer
        )
        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + doc.metadata["messages"]
        yield doc
