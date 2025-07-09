from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from web_utils import (
    get_edu_filters,
    get_pii_formatter,
    get_decontamination_filters,
)

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--language", type=str, default="fr", help="Language to process"
    )
    args = parse_args(parser)
    language = args.language
    DATA_PATH = args.data_path

    # Get language specific filtering and formatting
    edu_filters = get_edu_filters(language)
    pii_formatter = get_pii_formatter(language)
    decontamination_filters = get_decontamination_filters(language)

    pipeline = [
        HuggingFaceDatasetReader(
            "uonlp/CulturaX",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        *edu_filters,
        *decontamination_filters,
        *pii_formatter,
        PrefixFormatter(date_keys=["timestamp"], date_format="%Y/%m/%d %H:%M:%S"),
        JsonlWriter(
            f"{DATA_PATH}/culturax_filtered/{language}/data",
            output_filename="${source}_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    filtering_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/culturax_filtered/{language}/logs",
        job_name="culturax_filtered",
        tasks=50,
        partition="cpu_p1",
        cpus_per_task=2,  # OOM with 1...
        time="20:00:00",
    )
    filtering_executor.run()

    # ############
    # ### robots.txt
    # ############

    # pipeline = [
    #     JsonlReader(
    #         f"{DATA_PATH}/culturax_filtered/{language}/data/mC4",
    #     ),
    #     RobotsTxtFilter(
    #         robots_txt_path="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_data/full_datasets/robots_txt/data"
    #     ),
    #     JsonlWriter(
    #         f"{DATA_PATH}/culturax_filtered/{language}/robots_data",
    #         output_filename="${source}/${rank}.jsonl.gz",
    #         max_file_size=int(2e9),
    #     ),
    # ]

    # robots_executor = create_executor(
    #     pipeline,
    #     local=args.local,
    #     debug=args.debug,
    #     logging_dir=f"{DATA_PATH}/culturax_filtered/{language}/robots_logs",
    #     job_name="culturax_filtered",
    #     tasks=50,
    #     partition="cpu_p1",
    #     time="20:00:00",
    #     depends=filtering_executor,
    # )

    # robots_executor.run()
