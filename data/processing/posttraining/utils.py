import os
import json
import importlib.util
from functools import cached_property
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
        self,
        message_fields="messages",
        original_suffix: str | None = "_original",
        exclusion_writer: DiskWriter = None,
    ):
        """
        original_suffix: if set, the pre-conversion (nested) messages of each field
            are saved with json.dumps in ``metadata[f"{field}{original_suffix}"]``.
            Set to None to disable.
        """
        super().__init__(exclusion_writer)
        if isinstance(message_fields, str):
            message_fields = [message_fields]
        self.message_fields = message_fields
        self.original_suffix = original_suffix

    def filter(self, doc: Document) -> bool:
        for message_key in self.message_fields:
            if message_key not in doc.metadata:
                return False, f"missing_{message_key}"
            if self.original_suffix:
                doc.metadata[f"{message_key}{self.original_suffix}"] = json.dumps(
                    doc.metadata[message_key], ensure_ascii=False
                )
            doc.metadata[message_key] = nemo_rl_format_messages(doc.metadata[message_key])
        return True


TOKEN_COUNT_BUCKETS = [
    (2**12, "0-4k"),
    (2**13, "4k-8k"),
    (2**14, "8k-16k"),
    (2**15, "16k-32k"),
    (2**16, "32k-65k"),
    (2**17, "65k-131k"),
]


def token_count_label(token_count: int) -> str:
    for upper, label in TOKEN_COUNT_BUCKETS:
        if token_count < upper:
            return label
    return "131k+"


class ChatTokenCount(BaseFilter):
    """Apply the chat template and count tokens (total and assistant turns).

    Saves ``metadata["token_count"]`` (whole rendered conversation) and
    ``metadata["assistant_token_count"]`` (tokens inside the template's
    ``{% generation %}`` blocks, i.e. the trained-on tokens), plus
    ``metadata["token_count_label"]`` (see ``token_count_label``). Documents are kept
    unless one of the optional bounds is set and violated.

    chat_template_kwargs: extra arguments passed to ``tokenizer.apply_chat_template``
        (e.g. ``{"enable_thinking": False}``).
    tools_key: if set, ``metadata[tools_key]`` (list or JSON string) is passed as
        ``tools``.
    """

    name = "🔢 Chat Token Count"

    def __init__(
        self,
        tokenizer_name_or_path: str,
        messages_key: str = "messages",
        tools_key: str | None = None,
        chat_template_kwargs: dict | None = None,
        max_tokens: int | None = None,
        min_assistant_tokens: int | None = None,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.messages_key = messages_key
        self.tools_key = tools_key
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.max_tokens = max_tokens
        self.min_assistant_tokens = min_assistant_tokens

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def update_doc_stats(self, doc: Document):
        # BaseFilter.run calls this on kept documents only (dropped ones just
        # count in dropped_<reason>). Replaces the base doc_len/doc_len_tokens.
        self.stat_update("token_count", value=doc.metadata["token_count"], unit="doc")
        self.stat_update(
            "assistant_token_count",
            value=doc.metadata["assistant_token_count"],
            unit="doc",
        )
        self.stat_update(f"token_count_{doc.metadata['token_count_label']}")

    def filter(self, doc: Document) -> bool:
        import json

        if self.messages_key not in doc.metadata:
            return False, f"missing_{self.messages_key}"
        kwargs = dict(self.chat_template_kwargs)
        if self.tools_key is not None:
            tools = doc.metadata.get(self.tools_key)
            if isinstance(tools, str):
                tools = json.loads(tools)
            kwargs["tools"] = tools or None

        out = self.tokenizer.apply_chat_template(
            doc.metadata[self.messages_key],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )
        token_count = len(out["input_ids"])
        assistant_token_count = sum(out["assistant_masks"])
        doc.metadata["token_count"] = token_count
        doc.metadata["assistant_token_count"] = assistant_token_count
        doc.metadata["token_count_label"] = token_count_label(token_count)

        if self.max_tokens is not None and token_count > self.max_tokens:
            return False, "too_many_tokens"
        if (
            self.min_assistant_tokens is not None
            and assistant_token_count < self.min_assistant_tokens
        ):
            return False, "too_few_assistant_tokens"
        return True
