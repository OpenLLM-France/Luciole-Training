from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "pes2o"

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/olmo-mix-1124",
            {"name": "pes2o", "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/{dataset_name}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/{dataset_name}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
