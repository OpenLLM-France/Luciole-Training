from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
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
    import json
    import re
    import random

    for doc in data:
        tools = doc.metadata.get("tools", [])
        tools = [json.loads(tool) for tool in tools]
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
            "nvidia/When2Call",
            {"name": "train_sft", "split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        partial(format_messages, tokenizer=tokenizer),
        partial(apply_chat_template, tokenizer=tokenizer),
        JsonlWriter(
            f"{DATA_PATH}/when2call/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/when2call/logs",
        job_name="when2call",
        tasks=1,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
