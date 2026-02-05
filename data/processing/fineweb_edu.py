from utils import create_parser, parse_args, create_executor, add_sampler_filter
from web_utils import get_web_pipeline
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb-edu",
                glob_pattern="data/*/*.parquet",
            ),
            *get_web_pipeline(
                "en",
                f"{DATA_PATH}/fineweb_edu_filtered",
                do_edu=False,
                do_pii=True,
                do_decont=False,
            ),
            JsonlWriter(f"{DATA_PATH}/fineweb_edu_filtered/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/fineweb_edu_filtered/logs",
            job_name="fw-edu",
            tasks=100,
            max_array_size=50,
            time="20:00:00",
        )
        main_processing_executor.run()

    else:
        # Push to hub
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/fineweb_edu_filtered/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/fineweb_edu_filtered/data_hf",
                output_filename="data/fineweb_edu/en/score_${int_score}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="fineweb_edu",
                    id_key=None,
                    language=None,
                    language_key=None,
                    conversation_key=None,
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
            logging_dir=f"{DATA_PATH}/fineweb_edu_filtered/logs_hf",
            job_name="hf_fw_edu",
            tasks=20,
            workers=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
