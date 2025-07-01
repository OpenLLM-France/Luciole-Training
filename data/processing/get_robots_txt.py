import os
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import WarcForRobotsReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

DUMP_TO_PROCESS = "CC-MAIN-2024-42"

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    output_path = os.path.join(DATA_PATH, "robots_txt")

    pipeline = [
        WarcForRobotsReader(
            "/lustre/fsmisc/dataset/CommonCrawl/CC-MAIN-2024-42/segments/",
            glob_pattern="*/robotstxt/*",  # we want the robotstxt files
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        JsonlWriter(
            f"{output_path}/data/",
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        tasks=5,
        partition="cpu_p1",
        logging_dir=f"{output_path}/logs",
        job_name="robots.txt",
    )

    main_processing_executor.run()
