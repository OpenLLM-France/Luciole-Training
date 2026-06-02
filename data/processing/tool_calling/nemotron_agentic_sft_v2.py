from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import apply_chat_template, instruct_adapter


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import re
    import random

    for doc in data:
        messages = doc.metadata["messages"]
        tools = doc.metadata.get("tools", [])
        random.shuffle(tools)

        # Clean tool response
        for message in messages:
            if message["role"] == "tool":
                message["content"] = message["content"].strip()

        # Add tool prompt
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = ""

        system_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tools=tools,
            tokenize=False,
        )
        system_prompt = re.search(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>", system_prompt, re.DOTALL
        ).group(1)

        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt},
        ] + messages
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    for split_name in ["interactive_agent", "search", "tool_calling"]:
        pipeline = [
            JsonlReader(
                "hf://datasets/nvidia/Nemotron-SFT-Agentic-v2/data/",
                glob_pattern=f"{split_name}.jsonl",
                adapter=instruct_adapter,
            ),
            partial(format_messages, tokenizer=tokenizer),
            partial(apply_chat_template, tokenizer=tokenizer),
            JsonlWriter(
                f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/data",
                expand_metadata=True,
            ),
        ]

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/nemotron_agentic_sft_v2/{split_name}/logs",
            job_name=f"agentic_v2_{split_name}",
            tasks=1,
            time="00:30:00",
            # partition="cpu_p1",
            qos="qos_cpu-dev",
            skip_completed=not args.force,
        )
        main_processing_executor.run()
