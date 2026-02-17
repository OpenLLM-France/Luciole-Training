from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
)

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    LambdaFilter,
)
from datatrove.data import DocumentsPipeline

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    def get_source(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        import re

        for doc in data:
            m = re.search(
                r"hf://datasets/allenai/dolma3_longmino_mix-100B-1125/data/([^/]+)/.*",
                doc.metadata["file_path"],
            )
            doc.metadata["source"] = m.group(1)
            yield doc

    pipeline = [
        JsonlReader(
            "hf://datasets/allenai/dolma3_longmino_mix-100B-1125",
            glob_pattern="data/**/*.jsonl.zst",
        ),
        LambdaFilter(
            lambda doc: "lc_synth" in doc.metadata["file_path"]
            or "olmocr_science" in doc.metadata["file_path"]
        ),
        get_source,
        JsonlWriter(
            f"{DATA_PATH}/dolma3_longmino/data",
            output_filename="${source}/${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/dolma3_longmino/logs",
        job_name="longmino",
        tasks=20,
        skip_completed=not args.force,
    )

    main_processing_executor.run()
