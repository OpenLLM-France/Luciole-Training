from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import get_web_pipeline, ROBOTSTXT_PATH
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial
from datatrove.data import DocumentsPipeline

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fr", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            ParquetReader(
                f"hf://datasets/uonlp/CulturaX/{language}", glob_pattern="*.parquet"
            ),
            *get_web_pipeline(
                language,
                robots_txt_path=ROBOTSTXT_PATH,
                output_path=f"{DATA_PATH}/culturax_filtered/{language}",
                do_edu=True,
                do_pii=True,
                do_decont=False,
            ),
            PrefixFormatter(
                date_keys=["timestamp"],
                date_format="%Y/%m/%d %H:%M:%S",
                prefix_pipeline={"domain": "Domain", "date": "Date"},
            ),
            JsonlWriter(
                f"{DATA_PATH}/culturax_filtered/{language}/data",
                output_filename="${source}_edu_${edu_score}_${rank}.jsonl.gz",
                max_file_size=int(2e9),
            ),
        ]

        main_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/culturax_filtered/{language}/logs",
            job_name=f"cx_{language}",
            tasks=50,
        )

        main_executor.run()

    else:

        def fix_data(
            data: DocumentsPipeline,
            rank: int = 0,
            world_size: int = 1,
            language: str = "fr",
        ):
            for doc in data:
                doc.metadata["language"] = language
                doc.metadata["edu_score_int"] = doc.metadata.pop("edu_score")
                doc.metadata["edu_score"] = doc.metadata.pop("edu_score_mean")
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/culturax_filtered/{language}/data",
            ),
            partial(fix_data, language=language),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/culturax/{language}/data_hf",
                output_filename="data/culturax/${language}/score_${edu_score_int}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="culturax",
                    id_key=None,
                    language=language,
                    language_key=None,
                    conversation_key=None,
                    remove_keys=["timestamp", "prefix"],
                    remove_prefix=True,
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
            logging_dir=f"{DATA_PATH}/culturax_filtered/{language}/logs_hf",
            job_name="hf_culturax",
            tasks=10,
        )

        hf_executor.run()
