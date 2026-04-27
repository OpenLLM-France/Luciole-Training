from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import FilterChinese, apply_chat_template, instruct_adapter
from smolagents_toolcalling import clean_tool_response


def reformat_messages(data, rank: int = 0, world_size: int = 1):
    def _reformat_messages(messages):
        role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "tool": "tool",
        }

        return [
            {"role": role_map.get(turn["from"]), "content": turn["value"]}
            for turn in messages
        ]

    for doc in data:
        messages = doc.metadata.pop("conversations")
        doc.metadata["messages"] = _reformat_messages(messages)
        yield doc


def reset_system_prompt(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import re
    import json
    import random

    for document in data:
        messages = document.metadata["messages"]
        if messages[0]["role"] == "system":
            messages = messages[1:]

        tools = document.metadata.get("tools", None)
        tools = json.loads(tools)
        if tools:
            random.shuffle(tools)

        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}],
            tools=tools,
            tokenize=False,
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)
        document.metadata["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + messages
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    for subset in ["func_calling", "func_calling_singleturn", "glaive_func_calling"]:
        pipeline = [
            HuggingFaceDatasetReader(
                "NousResearch/hermes-function-calling-v1",
                {"name": subset, "split": "train"},
                streaming=True,
                adapter=instruct_adapter,
            ),
            reformat_messages,
            partial(reset_system_prompt, tokenizer=tokenizer),
            clean_tool_response,
            partial(apply_chat_template, tokenizer=tokenizer),
            FilterChinese(
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/hermes_function_calling/{subset}/chinese_heavy"
                ),
            ),
            JsonlWriter(
                f"{DATA_PATH}/hermes_function_calling/{subset}/data",
                expand_metadata=True,
            ),
        ]

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/hermes_function_calling/{subset}/logs",
            job_name="hermes_function_calling",
            tasks=1,
            time="00:30:00",
            # partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
