from utils import *

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
    args = parser.parse_args()
    language = args.language

    DATA_PATH = get_data_path(args)

    pipeline = [
        HuggingFaceDatasetReader(
            "almanach/HALvest",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/halvest/{language}/output"),
    ]
    pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/halvest/{language}/logs",
        job_name="halvest",
    )

    main_processing_executor.run()
