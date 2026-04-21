from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from datasets import get_dataset_split_names
from transformers import AutoTokenizer
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from utils import FilterChinese, apply_chat_template, instruct_adapter


class ToolCallingFiltering(BaseFilter):
    name = "🪚🔨🔧 Tool Calling Filtering"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    def filter(self, doc: Document) -> bool:
        import re
        import ast
        import json

        for message in doc.metadata["messages"]:
            if message["role"] == "assistant" and "<tool_call>" in message["content"]:
                for match in re.finditer(
                    r"<tool_call>(.*?)</tool_call>", message["content"], re.DOTALL
                ):
                    raw = match.group(1).strip()
                    parsed = ast.literal_eval(raw)

                    # Validate keys
                    if set(parsed.keys()) != {"name", "arguments"}:
                        return False, "tool_calling_arguments"
                        # raise ValueError(f"Unexpected tool call keys: {set(parsed.keys())}")
                    if not isinstance(parsed["name"], str):
                        return False, "tool_calling_name_not_a_string"
                        # raise ValueError(f"Tool call 'name' must be a string, got: {type(parsed['name'])}")
                    if not isinstance(parsed["arguments"], dict):
                        return False, "tool_calling_arguments_not_a_dict"
                        # raise ValueError(f"Tool call 'arguments' must be a dict, got: {type(parsed['arguments'])}")

                    # Rebuild with guaranteed key order: name first
                    ordered = {"name": parsed["name"], "arguments": parsed["arguments"]}
                    clean_json = json.dumps(ordered)
                    message["content"] = message["content"].replace(
                        match.group(0), f"<tool_call>{clean_json}</tool_call>"
                    )
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


class ToolExtraction(BaseFilter):
    name = "🪚🔨🔧 Tool Extraction"

    def __init__(self, split_name, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.split_name = split_name

    def filter(self, doc: Document) -> bool:
        import ast
        import re

        doc.metadata["tools"] = []

        xml_tools = doc.metadata["chat_template_kwargs"]["xml_tools"]
        if len(xml_tools) == 0:
            return True
        elif len(xml_tools) > 1:
            return False, "multiple_xml_tools_entries"

        if (
            self.split_name == "hermes_function_calling_v1_no_think"
            and "no access" in xml_tools[0]
        ):
            doc.metadata["tools"] = []
            return True

        if self.split_name in [
            "hermes_function_calling_v1_no_think",
            "xlam_traces_no_think",
        ]:
            pattern = r"<tools>\s*(\[.*\])\s*</tools>"
            match = re.search(pattern, xml_tools[0], re.DOTALL)
            if match:
                try:
                    doc.metadata["tools"] = ast.literal_eval(match.group(1))
                    return True
                except Exception:
                    return False, "failed_tool_extraction"
            else:
                return False, "tools_tag_not_found"

        if self.split_name == "smolagents_toolcalling_traces_think":
            try:
                doc.metadata["tools"] = [
                    ast.literal_eval(line)
                    for line in xml_tools[0].splitlines()
                    if line.strip()
                ]
                return True
            except Exception:
                return False, "failed_tool_extraction"
        else:
            raise NotImplementedError(
                f"Don't know how to extract tools for split: {self.split_name}"
            )


def add_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import re

    _has_custom_instructions = False
    _has_xml_tools = False
    _has_python_tools = False

    for document in data:
        system_prompt = document.metadata["chat_template_kwargs"]["custom_instructions"]
        xml_tools = document.metadata["chat_template_kwargs"]["xml_tools"]
        python_tools = document.metadata["chat_template_kwargs"]["python_tools"]

        if system_prompt != "" and not _has_custom_instructions:
            _has_custom_instructions = True
            print(f"\n>>> Found custom instructions in the dataset: {system_prompt}\n")
        if len(xml_tools) and not _has_xml_tools:
            _has_xml_tools = True
            print(f"\n>>> Found XML tools in the dataset: {xml_tools}\n")
        if len(python_tools) and not _has_python_tools:
            _has_python_tools = True
            raise ValueError(
                f"\n>>> Found Python tools in the dataset: {python_tools}\n"
            )

        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tools=document.metadata.pop("tools", None),
            tokenize=False,
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)
        document.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + document.metadata["messages"]
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    splits = get_dataset_split_names("HuggingFaceTB/smoltalk2", "SFT")
    tool_splits = [
        "smolagents_toolcalling_traces_think",
        "hermes_function_calling_v1_no_think",
        "xlam_traces_no_think",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    # Add adapter that add empty text if the text field is None, to avoid skipping data
    for split in splits:
        # if split not in tool_splits:
        #     continue
        print(f"\n\n#### Processing split: {split}")

        tool_pipeline = (
            [
                ToolExtraction(
                    split_name=split,
                    exclusion_writer=JsonlWriter(
                        f"{DATA_PATH}/smoltalk2/{split}/tool_extraction_failed"
                    ),
                ),
                ToolCallingFiltering(
                    exclusion_writer=JsonlWriter(
                        f"{DATA_PATH}/smoltalk2/{split}/tool_calling_failed"
                    ),
                ),
                clean_tool_response,
            ]
            if split in tool_splits
            else []
        )

        pipeline = [
            HuggingFaceDatasetReader(
                "NousResearch/hermes-function-calling-v1",
                {"name": "SFT", "split": split},
                streaming=True,
                adapter=instruct_adapter,
            ),
            *tool_pipeline,
            partial(add_system_prompt, tokenizer=tokenizer),
            partial(apply_chat_template, tokenizer=tokenizer),
            FilterChinese(
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/smoltalk2/{split}/chinese_heavy"
                ),
            ),
            JsonlWriter(
                f"{DATA_PATH}/smoltalk2/{split}/data",
                expand_metadata=True,
            ),
        ]
        if split == "smoltalk_multilingual8_Qwen3_32B_think":
            add_sampler_filter(pipeline, 0.3)
        elif split == "OpenThoughts3_1.2M_no_think_no_think":
            add_sampler_filter(pipeline, 0.4)
        elif split == "OpenHermes_2.5_no_think":
            add_sampler_filter(pipeline, 0.5)
        elif split == "smoltalk_smollm3_smol_magpie_ultra_no_think":
            add_sampler_filter(pipeline, 0.5)
        elif split == "OpenThoughts3_1.2M_think":
            add_sampler_filter(pipeline, 0.02)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/smoltalk2/{split}/logs",
            job_name="smoltalk2",
            tasks=1,
            time="00:30:00",
            partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()

"""
smolagents_toolcalling_traces_think
hermes_function_calling_v1_no_think => tools tags!
xlam_traces_no_think => tools tag
"""
