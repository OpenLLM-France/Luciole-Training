from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
)

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline


def set_input_output_format(
    data: DocumentsPipeline,
    rank: int = 0,
    world_size: int = 1,
) -> DocumentsPipeline:
    for doc in data:
        doc.metadata["input"] = ""
        doc.metadata["output"] = doc.text
        doc.text = " "
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/dolma3_longmino_mix-100B-1125",
            glob_pattern="data/**/*.jsonl.zst",
        ),
        set_input_output_format,
        JsonlWriter(f"{DATA_PATH}/dolma3_longmino/data", expand_metadata=True),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/dolma3_longmino/logs",
        job_name="longmino",
        tasks=20,
    )

    main_processing_executor.run()
