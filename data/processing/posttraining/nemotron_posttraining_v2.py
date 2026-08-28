from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import FilterChinese, apply_chat_template, instruct_adapter
from datatrove.pipeline.filters import LambdaFilter


def filter_midjourney(doc):
    for message in doc.metadata["messages"]:
        if message["role"] == "user":
            if "midjourney" in message["content"].lower():
                return False, "midjourney"
            else:
                return True


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--split_name",
        type=str,
        choices=["stem", "chat", "math", "multilingual_fr"],
        default="chat",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    split_name = args.split_name

    pipeline = [
        HuggingFaceDatasetReader(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            {"split": split_name},
            streaming=True,
            adapter=instruct_adapter,
        ),
        *(
            [
                LambdaFilter(
                    filter_midjourney,
                    exclusion_writer=JsonlWriter(
                        f"{DATA_PATH}/nemotron_posttraining_v2/{split_name}/midjourney"
                    ),
                )
            ]
            if split_name == "chat"
            else []
        ),
        partial(apply_chat_template, tokenizer=tokenizer),
        FilterChinese(
            chinese_threshold=0.05,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/nemotron_posttraining_v2/{split_name}/chinese_heavy"
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/nemotron_posttraining_v2/{split_name}/data",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/nemotron_posttraining_v2/{split_name}/logs",
        job_name=split_name,
        tasks=5,
        time="02:00:00",
        # partition="cpu_p1",
        # qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
