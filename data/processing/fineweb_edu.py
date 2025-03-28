import os

from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

# FineWeb edu is already on JZ: $DSDIR/HuggingFace/fineweb/data
# up to CC-MAIN-2024-10
# new dumps are available

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name = "fineweb_edu"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline=[
            ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet"),
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
