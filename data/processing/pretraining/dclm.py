from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_robot_filter, get_dedup_filter
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "allenai/dolmino-mix-1124",
                {"name": "dclm", "split": "train"},
                streaming=True,
            ),
            get_dedup_filter(output_path=f"{DATA_PATH}/dclm_dolmino"),
            get_robot_filter(output_path=f"{DATA_PATH}/dclm_dolmino"),
            PIIFormatter(ip_replacement="<IP_ADDRESS>"),
            JsonlWriter(f"{DATA_PATH}/dclm_dolmino/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/dclm_dolmino/logs",
            job_name="dclm_dolmino",
            tasks=100,
            max_array_size=50,
        )
        main_executor.run()

    else:
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/dclm_dolmino/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/dclm_dolmino/data_hf",
                output_filename="data/dclm_dolmino/en/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="dclm_dolmino",
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
            logging_dir=f"{DATA_PATH}/dclm_dolmino/logs_hf",
            job_name="hf_dclm",
            tasks=20,
            workers=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
