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
            "hf://datasets/HuggingFaceFW/fineweb", glob_pattern="sample/350BT/*.parquet"
        ),
        *get_web_pipeline(
            "en",
            f"{DATA_PATH}/fineweb_sample350BT",
            do_edu=False,
            do_pii=True,
            do_decont=False,
        ),
        JsonlWriter(f"{DATA_PATH}/fineweb_sample350BT/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/fineweb_sample350BT/logs",
        job_name="fw-sample350BT",
        tasks=50,
        time="20:00:00",
    )
    main_processing_executor.run()
