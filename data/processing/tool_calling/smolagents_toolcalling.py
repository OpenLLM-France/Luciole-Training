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
)
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class ToolFiltering(BaseFilter):
    name = "🪚🔨🔧 Tool Calling Filtering"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool:
        import re
        import ast

        for message in doc.metadata["messages"]:
            if message["role"] == "assistant" and "<tool_call>" in message["content"]:
                message["tool_calls"] = []
                for match in re.finditer(
                    r"<tool_call>(.*?)</tool_call>", message["content"], re.DOTALL
                ):
                    raw = match.group(1).strip()
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        return False, "fail_to_parse_tool_calling"
                    message["content"] = message["content"].replace(match.group(0), "")
                    message["tool_calls"].append(parsed)
            # elif message["role"] == "tool":
            #     # remove $Obsertation:/s
        return True


def clean_tool_response(data, rank: int = 0, world_size: int = 1):
    import re

    for doc in data:

        def split_tool_responses(text):
            matches = re.findall(
                r"<tool_response>(.*?)</tool_response>", text, re.DOTALL
            )
            return [match.strip() for match in matches]

        messages = doc.metadata["messages"]
        new_messages = []
        for message in messages:
            if message["role"] == "tool" and "<tool_response>" in message["content"]:
                tool_responses = split_tool_responses(message["content"])
                # Create a new message for each tool response
                for tool_response in tool_responses:
                    new_messages.append({"role": "tool", "content": tool_response})
            else:
                new_messages.append(message)
        doc.metadata["messages"] = new_messages
        yield doc


def reformat_messages(data, rank: int = 0, world_size: int = 1, tokenizer=None):
    import json
    import re
    import random

    for doc in data:
        xml_tools = doc.metadata.pop("chat_template_kwargs")["xml_tools"][0]
        objs = xml_tools.strip().split("\n")
        # Add visit_website tool
        visit_webpage_tool = {
            "type": "function",
            "function": {
                "name": "visit_webpage",
                "description": "Retrieves and returns the content of a webpage given its URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL of the webpage to visit.",
                        }
                    },
                    "required": ["url"],
                },
            },
        }
        tools = [json.loads(o) for o in objs] + [visit_webpage_tool]
        if tools:
            random.shuffle(tools)

        doc.metadata["tools"] = tools

        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}],
            tools=tools,
            tokenize=False,
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)

        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt},
        ] + doc.metadata["messages"]
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
            "HuggingFaceTB/smoltalk2",
            {"name": "SFT", "split": "smolagents_toolcalling_traces_think"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        partial(reformat_messages, tokenizer=tokenizer),
        ToolFiltering(),
        check_last_message,
        partial(apply_chat_template, tokenizer=tokenizer),
        FilterChinese(
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/smolagents_toolcalling/chinese_heavy"
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/smolagents_toolcalling/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/smolagents_toolcalling/logs",
        job_name="smolagents_toolcalling",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
