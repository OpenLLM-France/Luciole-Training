from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import get_web_pipeline, ROBOTSTXT_PATH
from datatrove.data import DocumentsPipeline


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

    pipeline = [
        ParquetReader(
            f"hf://datasets/HPLT/HPLT2.0_cleaned/{language}", glob_pattern="*.parquet"
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

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/hplt2_filtered/{language}/logs",
        job_name=f"hplt_{language}",
        tasks=50,
        time="20:00:00",
    )

    main_processing_executor.run()
