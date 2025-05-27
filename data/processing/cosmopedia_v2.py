import os

from utils import create_parser, get_data_path, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "cosmopedia_v2"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2",
            glob_pattern="*.parquet",
        ),
        JsonlWriter(f"{output_path}/output"),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
