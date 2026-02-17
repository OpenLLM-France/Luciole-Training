from utils import (
    create_parser,
    parse_args,
    create_executor,
    add_sampler_filter,
    print_builder_config,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from web_utils import get_robot_filter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        choices=[
            "finemath-3plus",
            "infiwebmath-3plus",
            "finemath-4plus",
            "infiwebmath-4plus",
        ],
        default="finemath",
        help="Subset to load",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.name is None:
        print_builder_config("HuggingFaceTB/finemath")

    name = args.name

    if not args.push_only:
        pipeline = [
            ParquetReader(
                f"hf://datasets/HuggingFaceTB/finemath/{name}",
                glob_pattern="*.parquet",
            ),
            get_robot_filter(output_path=f"{DATA_PATH}/finemath_filtered/{name}"),
            JsonlWriter(
                f"{DATA_PATH}/finemath_filtered/{name}/data",
                output_filename="score_${int_score}_rank${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        filter_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/finemath_filtered/{name}/logs",
            job_name=name,
            partition="prepost",
        )
        filter_executor.run()

    elif name in ["finemath-3plus", "infiwebmath-3plus"]:
        main_name = name.split("-")[0]
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/finemath_filtered/{name}/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/finemath_filtered/{name}/data_hf",
                output_filename=f"data/{main_name}/"
                + "en/score_${int_score}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source=main_name,
                    id_key=None,
                    language="en",
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
            logging_dir=f"{DATA_PATH}/finemath_filtered/{name}/logs_hf",
            job_name="hf_finemath",
            workers=10,
        )

        hf_executor.run()
