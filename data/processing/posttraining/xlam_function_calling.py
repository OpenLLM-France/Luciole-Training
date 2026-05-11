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


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
):
    import json
    import re
    import random

    for doc in data:
        tools = doc.metadata.get("tools", None)
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

        doc.metadata["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc.metadata.pop("query")},
            {
                "role": "assistant",
                "tool_calls": json.loads(doc.metadata.pop("answers")),
            },
        ]
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
            "Salesforce/xlam-function-calling-60k",
            {"split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        partial(format_messages, tokenizer=tokenizer),
        partial(apply_chat_template, tokenizer=tokenizer),
        check_last_message,
        FilterChinese(
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/xlam/chinese_heavy"),
        ),
        JsonlWriter(
            f"{DATA_PATH}/xlam/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/xlam/logs",
        job_name="xlam",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
