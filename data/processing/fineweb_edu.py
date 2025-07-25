from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_robot_filter, get_dedup_filter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    ### LOAD
    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet"
        ),
        get_dedup_filter(output_path=f"{DATA_PATH}/fineweb_edu"),
        get_robot_filter(output_path=f"{DATA_PATH}/fineweb_edu"),
        JsonlWriter(f"{DATA_PATH}/fineweb_edu/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb_edu/logs",
        job_name="fw-edu",
        tasks=100,
        max_array_size=50,
    )
    main_processing_executor.run()
