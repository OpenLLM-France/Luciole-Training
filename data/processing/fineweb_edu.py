import os

from utils import *

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

# FineWeb edu is already on JZ: $DSDIR/HuggingFace/fineweb/data
# up to CC-MAIN-2024-10
# new dumps are available

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "fineweb_edu"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline=[
            ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet"),
            JsonlWriter(f"{output_path}/output"),
        ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )
    main_processing_executor.tasks = 100

    main_processing_executor.run()
