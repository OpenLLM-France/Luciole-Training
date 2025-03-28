import os

from utils import create_pipeline, create_parser, get_data_path

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    MAIN_PATH = get_data_path(args.debug, args.local)

    dataset_name = "starcoderdata"
    output_path = os.path.join(MAIN_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/bigcode/starcoderdata",
            glob_pattern="**/*.parquet",
            text_key="content",
        ),
        LambdaFilter(
            lambda doc: doc.metadata["max_stars_count"] >= 2
            if "max_stars_count" in doc.metadata
            else True,
            exclusion_writer=JsonlWriter(f"{output_path}/1_low_stars_count"),
        ),
        JsonlWriter(f"{output_path}/1_high_stars_count"),
    ]

    main_processing_executor = create_pipeline(
        pipeline,
        debug=args.debug,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
