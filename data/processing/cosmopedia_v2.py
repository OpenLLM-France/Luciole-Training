import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LambdaFilter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "cosmopedia_v2"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2",
            glob_pattern="*.parquet",
        ),
        LambdaFilter(
            lambda doc: doc.metadata["seed_data"] not in ["ultrachat", "openhermes2.5"],
        ),
        JsonlWriter(f"{output_path}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
