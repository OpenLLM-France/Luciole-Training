from utils import create_parser, parse_args, create_executor
from web_utils import get_web_pipeline
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
            *get_web_pipeline(
                "en",
                output_path=f"{DATA_PATH}/dclm_dolmino",
                do_dedup=True,
                do_edu=False,
                do_pii=True,
            ),
            JsonlWriter(f"{DATA_PATH}/dclm_dolmino/data"),
        ]

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
                dataset="OpenLLM-France/Luciole-Training-Dataset"
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
                    remove_keys=[
                        "Content-Length",
                        "Content-Type",
                        "WARC-Block-Digest",
                        "WARC-Concurrent-To",
                        "WARC-IP-Address",
                        "WARC-Payload-Digest",
                        "WARC-Record-ID",
                        "WARC-Target-URI",
                        "WARC-Type",
                        "WARC-Warcinfo-ID",
                        "previous_word_count",
                        "provenance",
                        "warcinfo",
                        "WARC-Truncated",
                        "dataset",
                    ],
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
