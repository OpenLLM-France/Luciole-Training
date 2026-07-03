import os
import importlib.util
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

# Directly load pretraining/utils.py under a unique module name to avoid
# a circular import, since this file is also named 'utils'.

spec = importlib.util.spec_from_file_location(
    "pretraining_utils",
    os.path.join(os.path.dirname(__file__), "..", "pretraining", "utils.py"),
)
pretraining_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretraining_utils)

create_parser = pretraining_utils.create_parser
parse_args = pretraining_utils.parse_args
create_executor = pretraining_utils.create_executor
add_sampler_filter = pretraining_utils.add_sampler_filter
_custom_adapter_for_hf = pretraining_utils._custom_adapter_for_hf
HF_SCHEMA = pretraining_utils.HF_SCHEMA


def instruct_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop(self.text_key, "<empty>"),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": (
            data.pop("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        )
        | data,  # pop metadata only if it's a dict
    }


class FilterChinese(BaseFilter):
    name = "🀄 Chinese Filter"

    def __init__(
        self, chinese_threshold: float = 0.2, exclusion_writer: DiskWriter = None
    ):
        super().__init__(exclusion_writer)
        self.chinese_threshold = chinese_threshold

    def filter(self, doc: Document) -> bool:
        def chinese_proportion(text):
            import re

            if not text:
                return 0.0
            chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
            chinese_count = len(chinese_pattern.findall(text))
            return chinese_count / len(text)

        return chinese_proportion(doc.text) < self.chinese_threshold


def apply_chat_template(data, rank: int = 0, world_size: int = 1, tokenizer=None):
    for doc in data:
        doc.text = tokenizer.apply_chat_template(
            doc.metadata["messages"], tokenize=False
        )
        yield doc


def format_tool_calls(tool_calls: list, content: str = "") -> str:
    import json
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


def nemo_rl_format_messages(messages: list) -> list:
    """Flatten a list of messages to the NeMo-RL format.

    Each message becomes a plain ``{"role", "content"}`` dict: reasoning is
    inlined inside ``<think>`` tags and tool calls are rendered into the content
    via ``format_tool_calls``.

    The reasoning trace is read from ``reasoning_content`` (used by e.g. the
    nemotron datasets) or ``reasoning`` (the key vLLM's chat API returns, so
    react_hotpot's tool-calling rollouts land here). Supporting both keeps the
    <think> block regardless of which producer generated the message.
    """
    def check_last_message(messages):
        last_idx = max(
            i
            for i, m in enumerate(messages)
            if m["role"] == "assistant"
        )
        messages = messages[: last_idx + 1]
        return messages

    new_messages = []
    for message in messages:
        content = message.get("content") or ""

        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        if reasoning_content:
            content = (
                "<think>\n"
                + reasoning_content.strip("\n")
                + "\n</think>\n\n"
                + content.lstrip("\n")
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            content = format_tool_calls(tool_calls, content)

        new_messages.append({"role": message["role"], "content": content})
    return check_last_message(new_messages)

class NemoRLFormat(BaseFilter):
    name = "🐟 Nemo RL Format"

    def __init__(
        self, message_fields="messages", exclusion_writer: DiskWriter = None
    ):
        super().__init__(exclusion_writer)
        if isinstance(message_fields, str):
            message_fields = [message_fields]
        self.message_fields = message_fields

    def filter(self, doc: Document) -> bool:
        for message_key in self.message_fields:
            if message_key not in doc.metadata:
                return False, f"missing_{message_key}"
            doc.metadata[message_key] = nemo_rl_format_messages(doc.metadata[message_key])
        return True

