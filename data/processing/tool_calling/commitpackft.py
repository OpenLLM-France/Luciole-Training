from functools import partial

from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
    ChatTokenCount,
)
from editpackft import filter_and_format


def commitpackft_adapter(subset):
    # Same fields as EditPackFT, except the instruction is the commit `subject`.
    # The reader is the generic "json" one, so the dataset name is set here.
    def adapter(self, data: dict, path: str, id_in_file: int | str):
        document = instruct_adapter(self, data, path, id_in_file)
        document["metadata"]["instruction"] = document["metadata"].pop("subject")
        document["metadata"]["dataset"] = f"bigcode/commitpackft/{subset}"
        return document

    return adapter


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--subset",
        type=str,
        default="html",
        help="CommitPackFT language subset (folder name under data/ in the repo).",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    name = f"commitpackft_{args.subset}_oaiformat"

    pipeline = [
        HuggingFaceDatasetReader(
            # The repo's loading script needs trust_remote_code: read the
            # subset's jsonl directly instead.
            "json",
            {
                "data_files": f"hf://datasets/bigcode/commitpackft/data/{args.subset}/data.jsonl",
                "split": "train",
            },
            adapter=commitpackft_adapter(args.subset),
        ),
        # Rows with an empty old file are file creations (~10% for html).
        partial(filter_and_format, allow_new_files=True),
        NemoRLFormat(),
        # Drops the trailing tool result: the sample ends on the last edit.
        check_last_message,
        ChatTokenCount(
            "OpenLLM-France/tokenizer_128k_dual",
            tools_key="tools",
            chat_template_kwargs={"enable_thinking": True, "training_mode": True},
            max_tokens=2**16,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/{name}/too_long"),
        ),
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/{name}",
            output_filename=f"data/{name}/${{rank}}.parquet",
            adapter=hub_adapter,
            schema=None,
            private=True,
            cleanup=False,
            expand_metadata=True,
            commit_message=args.commit,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{name}/logs",
        job_name=name,
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
