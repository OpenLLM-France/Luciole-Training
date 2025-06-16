from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

TASKS = 100

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fr", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path

    name = "culturax"
    output_dir = f"{DATA_PATH}/{name}/{language}"

    pipeline = [
        HuggingFaceDatasetReader(
            "uonlp/CulturaX",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{output_dir}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{output_dir}/logs",
        job_name=name,
        tasks=TASKS,
    )

    main_processing_executor.run()
