import os

from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name = "cosmopedia_v2"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2",
            glob_pattern="*.parquet",
        ),
        JsonlWriter(f"{output_path}/output"),
    ]

    main_processing_executor = create_pipeline(
        pipeline,
        dataset_name,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
    )

    main_processing_executor.run()
