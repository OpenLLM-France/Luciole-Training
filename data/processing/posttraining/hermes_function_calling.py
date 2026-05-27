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
    add_system_prompt,
)
from smolagents_toolcalling import clean_tool_response


def format_messages(data, rank: int = 0, world_size: int = 1):
    import json
    import re

    def _format_messages(messages):
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

    def extract_tools_from_system_prompt(text: str) -> list[dict]:
        matches = re.findall(r"<tools>(.*?)</tools>", text, re.DOTALL)
        tools_json_str = next((m.strip() for m in matches if m.strip()), None)
        if tools_json_str:
            return json.loads(tools_json_str)
        return []

    for doc in data:
        # HF conv format
        messages = doc.metadata.pop("conversations")
        messages = _format_messages(messages)

        # Remove system prompt if exists, we will add it back later to ensure consistency
        if messages[0]["role"] == "system":
            system_prompt, messages = messages[0], messages[1:]
        else:
            raise ValueError("First message should be system prompt")
        doc.metadata["messages"] = messages

        # Format tools
        doc.metadata.pop("tools")  # Not always correct
        try:
            doc.metadata["tools"] = extract_tools_from_system_prompt(
                system_prompt["content"]
            )
        except Exception:
            continue

        # Format tool_calls
        for message in doc.metadata["messages"]:
            if message["role"] == "assistant" and "<tool_call>" in message:
                tool_calls = re.sub(
                    r"<tool_call>(.*?)</tool_call>",
                    r"\1",
                    message["content"],
                    flags=re.DOTALL,
                )
                message["tool_calls"] = json.loads(tool_calls)
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    for subset in ["func_calling", "glaive_func_calling"]:
        pipeline = [
            HuggingFaceDatasetReader(
                "NousResearch/hermes-function-calling-v1",
                {"name": subset, "split": "train"},
                streaming=True,
                adapter=instruct_adapter,
            ),
            format_messages,
            # partial(replace_tool_name, rename_names=True, rename_params=False),
            partial(add_system_prompt, tokenizer=tokenizer),
            clean_tool_response,
            check_last_message,
            partial(apply_chat_template, tokenizer=tokenizer),
            FilterChinese(
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/hermes_oai_format/{subset}/chinese_heavy"
                ),
            ),
            JsonlWriter(
                f"{DATA_PATH}/hermes_oai_format/{subset}/data",
                expand_metadata=True,
            ),
        ]

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/hermes_oai_format/{subset}/logs",
            job_name="hermes_oai_format",
            tasks=1,
            time="00:30:00",
            # partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
