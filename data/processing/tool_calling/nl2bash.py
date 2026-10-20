from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import (
    hub_adapter,
    instruct_adapter,
    check_last_message,
    NemoRLFormat,
)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in a Linux shell and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


def dedup_nl(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    # NL2Bash pairs the same description with several equivalent commands
    # (up to 9). Keep the first one so one prompt maps to a single target.
    seen = set()
    for doc in data:
        nl = doc.metadata["nl"].strip()
        if nl in seen:
            continue
        seen.add(nl)
        yield doc


def format_messages(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        doc.metadata["tools"] = [BASH_TOOL]
        doc.metadata["messages"] = [
            {"role": "user", "content": doc.metadata.pop("nl").strip()},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": doc.metadata.pop("bash").strip()},
                        },
                    }
                ],
            },
        ]
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "jiacheng-ye/nl2bash",
            # The repo only ships a loading script (downloading from Dropbox):
            # read the Hub's auto-converted parquet instead. The test split is
            # left out as it is the NL2Bash benchmark.
            {"revision": "refs/convert/parquet", "split": "train+validation"},
            adapter=instruct_adapter,
        ),
        dedup_nl,
        format_messages,
        NemoRLFormat(),
        check_last_message,
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-France/tool_data" + "_debug" * args.debug,
            local_working_dir=f"{DATA_PATH}/nl2bash_oaiformat",
            output_filename="data/nl2bash_oaiformat/${rank}.parquet",
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
        logging_dir=f"{DATA_PATH}/nl2bash_oaiformat/logs",
        job_name="nl2bash_oaiformat",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
