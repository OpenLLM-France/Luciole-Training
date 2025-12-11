from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_web_pipeline
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
        *get_web_pipeline(
            "en",
            f"{DATA_PATH}/fineweb_edu_filtered",
            do_edu=False,
            do_pii=True,
            do_decont=False,
        ),
        JsonlWriter(f"{DATA_PATH}/fineweb_edu_filtered/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb_edu_filtered/logs",
        job_name="fw-edu",
        tasks=100,
        max_array_size=50,
        time="20:00:00",
    )
    main_processing_executor.run()
