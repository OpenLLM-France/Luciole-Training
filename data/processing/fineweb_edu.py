import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

# FineWeb edu is already on JZ: $DSDIR/HuggingFace/fineweb/data
# up to CC-MAIN-2024-10
# new dumps are available

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "fineweb_edu"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet"
        ),
        JsonlWriter(f"{output_path}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )
    main_processing_executor.tasks = 100

    main_processing_executor.run()
