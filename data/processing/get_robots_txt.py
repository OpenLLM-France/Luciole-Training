import os
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers.warc_for_robots import WarcForRobotsReader, RobotsMerger
from datatrove.pipeline.writers.jsonl import JsonlWriter
from slugify import slugify

DUMP_TO_PROCESS = "CC-MAIN-2024-42"
# DUMP_TO_PROCESS = ".CC-MAIN-2025-26"

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    output_path = os.path.join(DATA_PATH, "robots_txt", slugify(DUMP_TO_PROCESS))

    pipeline = [
        WarcForRobotsReader(
            f"/lustre/fsmisc/dataset/CommonCrawl/{DUMP_TO_PROCESS}/segments/",
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
        tasks=20,
        partition="cpu_p1",
        logging_dir=f"{output_path}/logs",
        job_name="robots.txt",
    )

    main_processing_executor.run()

    # Merge
    merger_executor = create_executor(
        [
            RobotsMerger(
                input_folder=f"{output_path}/data/",
                output_folder=f"{output_path}/data_merge/",
            )
        ],
        local=args.local,
        debug=args.debug,
        tasks=1,
        cpus_per_task=20,
        partition="prepost",
        logging_dir=f"{output_path}/logs_merge",
        job_name="robots.txt",
    )
    merger_executor.run()
