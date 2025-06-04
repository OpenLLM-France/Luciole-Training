from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language to load",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    language = args.language

    pipeline = [
        HuggingFaceDatasetReader(
            "almanach/HALvest",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/halvest/{language}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/halvest/{language}/logs",
        job_name="halvest",
    )

    main_processing_executor.run()
