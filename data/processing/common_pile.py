from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter

LANGUAGE_METADATA = [
    "upsto_filtered",
    "project_gutenberg_filtered",
    "library_of_congress_filtered",
    "pre_1929_books_filtered",
    "uk_hansard_filtered",
    "data_provenance_initiative_filtered",
]

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        choices=[
            "peS2o_filtered",
            "upsto_filtered",
            "biodiversity_heritage_library_filtered",
            "caselaw_access_project_filtered",
            "pubmed_filtered",
            "libretexts_filtered",
            "project_gutenberg_filtered",
            "doab_filtered",
            "library_of_congress_filtered",
            "pressbooks_filtered",
            "pre_1929_books_filtered",
            "regulations_filtered",
            "uk_hansard_filtered",
            "usgpo_filtered",
            "stackexchange_filtered",
            "ubuntu_irc_filtered",
            "public_domain_review_filtered",
            "foodista_filtered",
            "youtube_filtered",
            "data_provenance_initiative_filtered",
            "python_enhancement_proposals_filtered",
        ],
        help="Subset to load",
    )

    args = parse_args(parser)
    DATA_PATH = args.data_path

    name = args.name

    if name == "ubuntu_irc_filtered":
        language_filter = [
            LanguageFilter(
                label_only=True,
                keep_top_pairs_threshold=1,
            ),
            LambdaFilter(
                lambda doc: doc.metadata["language"]
                in ["en", "fr", "it", "de", "es", "ar", "pt", "nl"],
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/common_pile/{name}/removed/language"
                ),
            ),
        ]
    else:
        language_filter = [
            LanguageFilter(
                languages=["en", "fr", "it", "de", "es", "ar", "pt", "nl"],
                language_threshold=0.65,
                keep_top_pairs_threshold=1,
            ),
        ]

    pipeline = [
        HuggingFaceDatasetReader(
            f"common-pile/{name}",
            {"split": "train"},
            streaming=True,
        ),
        *language_filter,
        JsonlWriter(f"{DATA_PATH}/common_pile/{name}/data"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/common_pile/{name}/logs",
        job_name=name,
    )

    main_processing_executor.run()
