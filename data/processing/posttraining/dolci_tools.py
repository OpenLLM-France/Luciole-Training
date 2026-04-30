from utils import create_parser, parse_args, create_executor
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.writers.disk_base import DiskWriter
from functools import partial
from transformers import AutoTokenizer
from utils import apply_chat_template, instruct_adapter


class DolciFilter(BaseFilter):
    name = "DolciFilter"

    @staticmethod
    def ast_to_openai_tool_calls(fc_str):
        import ast
        import json

        def _flatten_name(node):
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return f"{_flatten_name(node.value)}.{node.attr}"
            raise ValueError(f"Unsupported callable form: {ast.dump(node)}")

        def _iter_calls(body):
            if isinstance(body, (ast.List, ast.Tuple)):
                yield from body.elts
            else:
                yield body

        tool_calls = []
        for line in fc_str.splitlines():
            line = line.strip()
            if not line:
                continue
            tree = ast.parse(line, mode="eval")
            for elem in _iter_calls(tree.body):
                if not isinstance(elem, ast.Call):
                    raise ValueError(
                        f"Expected a call expression, got: {ast.dump(elem)}"
                    )
                arguments = {}
                for kw in elem.keywords:
                    try:
                        arguments[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        arguments[kw.arg] = ast.unparse(kw.value)
                tool_calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": _flatten_name(elem.func),
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                )
        return tool_calls

    def __init__(self, tokenizer=None, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.tokenizer = tokenizer

    def filter(self, doc: Document):
        import json
        import random
        import re

        messages = doc.metadata["messages"]
        system_turn = messages[0]
        messages = messages[1:]

        tools = json.loads(system_turn["functions"])
        random.shuffle(tools)
        doc.metadata["tools"] = tools

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_turn.get("content", "")}],
            tools=tools,
            tokenize=False,
        )
        m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", rendered, re.DOTALL)
        if m is None:
            return False, "system_block_extraction_error"
        system_prompt = m.group(1)

        def split_tool_content(content, n):
            decoder = json.JSONDecoder()
            parts, idx, length = [], 0, len(content)
            while idx < length:
                while idx < length and content[idx].isspace():
                    idx += 1
                if idx >= length:
                    break
                obj, end = decoder.raw_decode(content, idx)
                parts.append(json.dumps(obj, ensure_ascii=False))
                idx = end
            if len(parts) != n:
                raise ValueError(
                    f"tool_response split mismatch: got {len(parts)}, expected {n}"
                )
            return parts

        cleaned = []
        last_tool_call_count = 0
        for message in messages:
            if message["role"] == "environment":
                message["role"] = "tool"

            fc = message.pop("function_calls", None)
            message.pop("functions", None)
            if fc is not None and str(fc).strip() not in ("", "null", "None"):
                try:
                    message["tool_calls"] = self.ast_to_openai_tool_calls(fc)
                except Exception:
                    return False, "tool_calls_parsing_error"

            if message.get("content") is None:
                message["content"] = ""

            if message["role"] == "tool" and last_tool_call_count > 1:
                try:
                    parts = split_tool_content(message["content"], last_tool_call_count)
                except Exception:
                    return False, "tool_response_split_error"
                for part in parts:
                    cleaned.append({"role": "tool", "content": part})
            else:
                cleaned.append(message)

            if message["role"] == "assistant":
                last_tool_call_count = len(message.get("tool_calls", []))
            elif message["role"] == "tool":
                last_tool_call_count = 0

        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt},
        ] + cleaned
        return True


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    pipeline = [
        ParquetReader(
            "hf://datasets/allenai/Dolci-Instruct-SFT-Tool-Use/data/",
            glob_pattern="*.parquet",
            adapter=instruct_adapter,
        ),
        DolciFilter(
            tokenizer=tokenizer,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/dolci_tools/function_tools_parsing_error"
            ),
        ),
        partial(apply_chat_template, tokenizer=tokenizer),
        JsonlWriter(
            f"{DATA_PATH}/dolci_tools/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/dolci_tools/logs",
        job_name="dolci_tools",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
