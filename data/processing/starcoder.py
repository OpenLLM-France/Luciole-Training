import os

from utils import create_parser, get_data_path, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader, HuggingFaceDatasetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--olmo", action="store_true", help="Read from olmo 2")
    args = parser.parse_args()
    DATA_PATH = get_data_path(args)

    dataset_name = "starcoder" + ("_olmo" if args.olmo else "")
    output_path = os.path.join(DATA_PATH, dataset_name)

    if args.olmo:
        pipeline = [
            HuggingFaceDatasetReader(
                "allenai/olmo-mix-1124",
                {"name": "starcoder", "split": "train"},
                streaming=True,
            ),
            JsonlWriter(f"{output_path}/output"),
        ]
    else:
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
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
