from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from datasets import get_dataset_split_names
from transformers import AutoTokenizer
from datatrove.pipeline.filters import LambdaFilter


def custom_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop(self.text_key, "<empty>"),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": (
            data.pop("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        )
        | data,  # pop metadata only if it's a dict
    }


def filter_chinese(text):
    def chinese_proportion(text):
        import re

        if not text:
            return 0.0
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        chinese_count = len(chinese_pattern.findall(text))
        return chinese_count / len(text)

    return chinese_proportion(text) < 0.2  # Adjust the threshold as needed


def add_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import ast
    import re

    def extract_tool(tool: str) -> str:
        pattern = r"<tools>\s*(\[.*\])\s*</tools>"
        match = re.search(pattern, tool, re.DOTALL)
        if match:
            return match.group(1)
        else:
            raise ValueError(
                f"Found <tools> tags but failed to extract tools with regex in tool: {tool}"
            )

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

        # Process tools if needed
        if xml_tools:
            try:
                tools = [
                    ast.literal_eval(line)
                    for xml_tool in xml_tools
                    for line in xml_tool.splitlines()
                    if line.strip()
                ]
            except Exception:
                try:
                    tools = ast.literal_eval(extract_tool(xml_tools[0]))
                except Exception:
                    document.metadata["failed_tool_extraction"] = True
                    tools = None
                    # raise ValueError(f"Failed to extract and parse XML tools. Original xml_tools: {repr(xml_tools)}")
        else:
            tools = None

        document.metadata["tools"] = tools
        # document.metadata["messages"] = [{"role": "system", "content": system_prompt}] + document.metadata["messages"]
        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}], tools=tools, tokenize=False
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)
        document.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + document.metadata["messages"]
        yield document


def apply_chat_template(data, rank: int = 0, world_size: int = 1, tokenizer=None):
    for document in data:
        document.text = tokenizer.apply_chat_template(
            document.metadata["messages"], tokenize=False
        )
        # document.text = remove_chinese_heavy_lines(document.text)
        # document.metadata["conversation"] = messages
        yield document


def clean_tool_response(data, rank: int = 0, world_size: int = 1, tokenizer=None):
    for doc in data:
        import re

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


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    splits = get_dataset_split_names("HuggingFaceTB/smoltalk2", "SFT")
    # print(splits)

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    # Add adapter that add empty text if the text field is None, to avoid skipping data
    for split in splits:
        # if split not in ["smolagents_toolcalling_traces_think", "hermes_function_calling_v1_no_think", "xlam_traces_no_think"]:
        #     continue
        print(f"\n\n#### Processing split: {split}")
        pipeline = [
            HuggingFaceDatasetReader(
                "HuggingFaceTB/smoltalk2",
                {"name": "SFT", "split": split},
                streaming=True,
                adapter=custom_adapter,
            ),
            partial(add_system_prompt, tokenizer=tokenizer),
            LambdaFilter(
                lambda doc: "failed_tool_extraction" not in doc.metadata,
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/smoltalk2/{split}/fail_tool_extraction"
                ),
            ),
            clean_tool_response,
            partial(apply_chat_template, tokenizer=tokenizer),
            LambdaFilter(
                lambda doc: filter_chinese(doc.text),
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/smoltalk2/{split}/chinese_heavy"
                ),
            ),
            JsonlWriter(f"{DATA_PATH}/smoltalk2/{split}/data", expand_metadata=True),
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
            time="20:00:00",
            skip_completed=not args.force,
        )
        main_processing_executor.run()

"""
smolagents_toolcalling_traces_think
hermes_function_calling_v1_no_think => tools tags!
xlam_traces_no_think => tools tag
"""
