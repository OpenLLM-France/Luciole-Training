from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
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

    ############
    # LOAD CULTURAX DATASET
    ############

    pipeline = [
        HuggingFaceDatasetReader(
            "uonlp/CulturaX",
            {"name": language, "split": "train"},
            streaming=True,
        ),
        JsonlWriter(
            f"{DATA_PATH}/culturax/{language}/data",
            output_filename="${source}/${rank}.jsonl.gz",
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

    ############
    # FILTER CULTURAX DATASET
    ############

    # Get language specific filtering and formatting
    edu_filters = get_edu_filters(language)
    pii_formatter = get_pii_formatter(language)

    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/culturax/{language}/data",
        ),
        *edu_filters,
        *pii_formatter,
        PrefixFormatter(date_keys=["timestamp"], date_format="%Y/%m/%d %H:%M:%S"),
        JsonlWriter(
            f"{DATA_PATH}/culturax_filtered/{language}/data",
            output_filename="${source}_edu_${edu_score}_${rank}.jsonl.gz",
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
        depends=main_processing_executor,
    )
    filtering_executor.run()

    ############
    # Decontaminate CULTURAX DATASET
    ############

    decontamination_filters = get_decontamination_filters(language)

    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/culturax_filtered/{language}/data",
        ),
        *decontamination_filters,
        PrefixFormatter(date_keys=["timestamp"], date_format="%Y/%m/%d %H:%M:%S"),
        JsonlWriter(
            f"{DATA_PATH}/culturax_decont/{language}/data",
            output_filename="${source}_${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    decont_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/culturax_decont/{language}/logs",
        job_name="culturax_decont",
        tasks=50,
        partition="cpu_p1",
        time="20:00:00",
        depends=filtering_executor,
    )
