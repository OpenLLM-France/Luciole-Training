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

