from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_robot_filter
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/dolmino-mix-1124",
            {"name": "dclm", "split": "train"},
            streaming=True,
        ),
        get_robot_filter(output_path=f"{DATA_PATH}/dclm_dolmino/removed"),
        PIIFormatter(ip_replacement="<IP_ADDRESS>"),
        JsonlWriter(f"{DATA_PATH}/dclm_dolmino/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/dclm_dolmino/logs",
        job_name="dclm_dolmino",
        tasks=100,
        max_array_size=50,
    )
