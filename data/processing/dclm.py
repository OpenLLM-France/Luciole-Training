from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        HuggingFaceDatasetReader(
            "allenai/dolmino-mix-1124",
            {"name": "dclm", "split": "train"},
            streaming=True,
        ),
        JsonlWriter(f"{DATA_PATH}/dclm_dolmino/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/dclm_dolmino/logs",
        job_name="dclm_dolmino",
    )

    ################
    ## PII Cleaning
    ################

    pii_cleaning = [
        PIIFormatter(email_replacement="<<pii_email>>", ip_replacement="<<pii_ip>>"),
        # PhoneNumberPII("US"),
        # PhoneNumberPII("GB"),
    ]

    pipeline = [
        JsonlReader(f"{DATA_PATH}/dclm_dolmino/output"),
        *pii_cleaning,
        JsonlWriter(
            f"{DATA_PATH}/dclm_dolmino/output_clean_pii", max_file_size=int(2e9)
        ),
    ]

    pii_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/dclm_dolmino/logs_clean_pii",
        job_name="dclm_dolmino",
        depends=main_processing_executor,
    )
    pii_executor.run()
