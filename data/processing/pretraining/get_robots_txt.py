import os
from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers.warc_for_robots import WarcForRobotsReader, RobotsMerger
from datatrove.pipeline.writers.jsonl import JsonlWriter
from slugify import slugify
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--input_path",
        default="/lustre/fsmisc/dataset/CommonCrawl/.CC-MAIN-2026-25",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
    )
    args = parse_args(parser)

    # name of the dump folder itself, with or without a trailing slash on --input_path
    dump_name = os.path.basename(os.path.normpath(args.input_path))
    output_path = args.output_path or os.path.join(
        args.data_path, "robots_txt", slugify(dump_name)
    )

    if not args.push_only:
        pipeline = [
            WarcForRobotsReader(
                os.path.join(args.input_path, "segments"),
                glob_pattern="*/robotstxt/*",  # we want the robotstxt files
                default_metadata={"dump": dump_name},
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
                dataset="OpenLLM-France/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename=f"robots_txt/{slugify(dump_name)}"
                + "${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source=slugify(dump_name),
                    id_key=None,
                    language="",
                    language_key=None,
                    conversation_key=None,
                    remove_keys=[],
                ),
                cleanup=True,
                expand_metadata=False,
                schema=HF_SCHEMA,
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
