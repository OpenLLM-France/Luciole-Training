from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import get_web_pipeline

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fr", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "uonlp/CulturaX",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        *get_web_pipeline(
            language,
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

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/culturax/{language}/logs",
        job_name="culturax",
        tasks=50,
    )

    main_processing_executor.run()
