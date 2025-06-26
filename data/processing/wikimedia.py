from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "OpenLLM-France/wikimedia",
            {"split": "train"},
            streaming=True,
        ),
        JsonlWriter(
            f"{DATA_PATH}/wikimedia/data",
            output_filename="${language}/${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/wikimedia/logs",
        job_name="wikimedia",
        tasks=50,
    )

    main_processing_executor.run()
