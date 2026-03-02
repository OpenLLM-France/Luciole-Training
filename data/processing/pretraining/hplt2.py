from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import get_web_pipeline, ROBOTSTXT_PATH
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def rename_metadata(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        url = doc.metadata.pop("u")
        doc.metadata["url"] = url
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fra_Latn", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            ParquetReader(
                f"hf://datasets/HPLT/HPLT2.0_cleaned/{language}",
                glob_pattern="*.parquet",
            ),
            rename_metadata,
            *get_web_pipeline(
                language,
                robots_txt_path=ROBOTSTXT_PATH,
                output_path=f"{DATA_PATH}/hplt2_filtered/{language}",
                do_edu=True,
                do_pii=True,
                do_decont=False,
            ),
            PrefixFormatter(
                date_keys=["ts"],
                date_format="%Y/%m/%d %H:%M:%S",
                prefix_pipeline={"domain": "Domain", "date": "Date"},
            ),
            JsonlWriter(
                f"{DATA_PATH}/hplt2_filtered/{language}/data",
                output_filename="edu_${edu_score}_${rank}.jsonl.gz",
            ),
        ]

        main_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/hplt2_filtered/{language}/logs",
            job_name=f"hplt_{language}",
            tasks=50,
            time="20:00:00",
        )

        main_executor.run()

    else:

        def fix_data(
            data: DocumentsPipeline,
            rank: int = 0,
            world_size: int = 1,
            language: str = "fra_Latn",
        ):
            from web_utils import LanguageCodes

            for doc in data:
                doc.metadata["language_iso"] = LanguageCodes.fineweb_to_iso1(language)
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/hplt2_filtered/{language}/data",
            ),
            partial(fix_data, language=language),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/hplt2_filtered/{language}/data_hf",
                output_filename="data/hplt2/${language_iso}/score_${edu_score}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="hplt2",
                    id_key=None,
                    language=None,
                    language_key="language_iso",
                    conversation_key=None,
                    remove_prefix=True,
                    remove_keys=["prefix"],
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
            logging_dir=f"{DATA_PATH}/hplt2_filtered/{language}/logs_hf",
            job_name="hf_hplt2",
            tasks=10,
        )

        hf_executor.run()
