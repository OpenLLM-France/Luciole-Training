import os
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers.warc_for_robots import WarcForRobotsReader, RobotsMerger
from datatrove.pipeline.writers.jsonl import JsonlWriter
from slugify import slugify
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter

DUMP_TO_PROCESS = ".CC-MAIN-2025-26"

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    output_path = os.path.join(DATA_PATH, "robots_txt", slugify(DUMP_TO_PROCESS))

    if not args.push_only:
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
            tasks=50,
            qos="qos_cpu-t3",
            time="01:00:00",
            partition="cpu_p1",
            logging_dir=f"{output_path}/logs",
            job_name="robots.txt",
        )

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
            cpus_per_task=40,
            qos="qos_cpu-dev",
            time="02:00:00",
            partition="cpu_p1",
            logging_dir=f"{output_path}/logs_merge",
            job_name="robots.txt",
            depends=main_processing_executor,
        )
        merger_executor.run()

    else:
        pipeline = [
            JsonlReader(
                f"{output_path}/data_merge/",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename=f"robots_txt/{slugify(DUMP_TO_PROCESS)}"
                + "${rank}.parquet",
                cleanup=True,
                expand_metadata=True,
            ),
        ]

        hf_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{output_path}/logs_hf",
            job_name="hf_robots",
            tasks=1,
            time="20:00:00",
            skip_completed=not args.force,
        )

        hf_executor.run()
