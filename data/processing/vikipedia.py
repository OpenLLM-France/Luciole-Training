from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--path_to_parquet",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/commun/datasets/raw/Vikidia/20250615/parquet",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        ParquetReader(args.path_to_parquet),
        JsonlWriter(
            f"{DATA_PATH}/vikidia/data",
            output_filename="${language}/${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/vikidia/logs",
        job_name="vikidia",
        tasks=20,
    )

    main_processing_executor.run()
