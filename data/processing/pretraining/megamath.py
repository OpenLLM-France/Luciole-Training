from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_robot_filter
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

SUBSETS = [
    "megamath-web",
    "megamath-web-pro",
    "megamath-translated-code",
    "megamath-code",
    "megamath-text-code-block",
]

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        choices=SUBSETS,
        default="megamath-web",
        help="Name of the dataset to process.",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    name = args.name

    print(f"\nProcessing {name}...")

    if not args.push_only:
        # Load data
        pipeline = [
            ParquetReader(
                f"hf://datasets/LLM360/MegaMath/{name}",
            ),
            JsonlWriter(f"{DATA_PATH}/megamath/{name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        load_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/megamath/{name}/logs",
            job_name=name,
        )

        # Filtering
        pipeline = [
            JsonlReader(f"{DATA_PATH}/megamath/{name}/data"),
            get_robot_filter(output_path=f"{DATA_PATH}/megamath_filtered/{name}"),
            JsonlWriter(f"{DATA_PATH}/megamath_filtered/{name}/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        filter_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/megamath_filtered/{name}/logs",
            job_name=name,
            depends=load_executor,
            partition="prepost",
        )

        filter_executor.run()

    elif name == "megamath-web":
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/megamath_filtered/{name}/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/megamath_filtered/{name}/data_hf",
                output_filename=f"data/{name}/" + "en/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source=name,
                    id_key=None,
                    reset_id=False,
                    language=None,
                    language_key="lang",
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
            logging_dir=f"{DATA_PATH}/megamath_filtered/{name}/logs_hf",
            job_name="hf_megamath",
            tasks=5,
        )

        hf_executor.run()
