from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import LanguageFilter, LambdaFilter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

COMMON_PILE_DATASETS = [
    "peS2o_filtered",
    "biodiversity_heritage_library_filtered",
    "caselaw_access_project_filtered",
    "pubmed_filtered",
    "libretexts_filtered",
    "doab_filtered",
    "library_of_congress_filtered",
    "pressbooks_filtered",
    "pre_1929_books_filtered",
    "regulations_filtered",
    "stackexchange_filtered",
    "ubuntu_irc_filtered",
    "public_domain_review_filtered",
    "foodista_filtered",
    "youtube_filtered",
    "data_provenance_initiative_filtered",
    "python_enhancement_proposals_filtered",
    "oercommons_filtered",
    "news_filtered",
    "arxiv_papers_filtered",
    "arxiv_abstracts_filtered",
    "github_archive_filtered",
]


def additionnal_formatting(doc):
    out = {}
    channel = doc.metadata.get("channel")
    if channel:
        out["channel"] = doc.metadata.get("channel")
    title = doc.metadata.get("title")
    if title and title.lower() not in doc.text[:200].lower():
        out["title"] = doc.metadata.get("title")
    extfieldsofstudy = doc.metadata.get("s2fieldsofstudy")
    if extfieldsofstudy:
        extfieldsofstudy = ", ".join(extfieldsofstudy)
        if extfieldsofstudy:
            out["fields"] = extfieldsofstudy
    return out


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        nargs="+",
        choices=COMMON_PILE_DATASETS,
        help="Subset to load",
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if args.all:
        names = COMMON_PILE_DATASETS
    else:
        names = args.name

    for name in names:
        if not args.push_only:
            print(f"\nProcessing {name}...")
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

            if name == "public_domain_review_filtered":
                date_keys = []  # Date already in the text
            else:
                date_keys = ["created", "date", "published_time"]
            prefix_pipeline = {
                "channel": "Channel",
                "title": "Title",
                "fields": "Fields",
                "date": "Date",
            }

            pipeline = [
                HuggingFaceDatasetReader(
                    f"common-pile/{name}",
                    {"split": "train"},
                    streaming=True,
                ),
                *language_filter,
                PrefixFormatter(
                    date_keys=date_keys,
                    infer_date_format=True,
                    additionnal_formatting=additionnal_formatting,
                    prefix_pipeline=prefix_pipeline,
                ),
                JsonlWriter(f"{DATA_PATH}/common_pile/{name}/data"),
            ]
            add_sampler_filter(pipeline, args.sample_rate)

            main_processing_executor = create_executor(
                pipeline,
                local=args.local,
                debug=args.debug,
                logging_dir=f"{DATA_PATH}/common_pile/{name}/logs",
                job_name=name,
                tasks=50,
            )
            main_processing_executor.run()

        else:
            pipeline = [
                JsonlReader(
                    f"{DATA_PATH}/common_pile/{name}/data",
                ),
                HuggingFaceDatasetWriter(
                    dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                    + ("-debug" if args.debug else ""),
                    private=True,
                    local_working_dir=f"{DATA_PATH}/common_pile/{name}/data_hf",
                    output_filename=f"data/common_pile/{name}/en/" + "${rank}.parquet",
                    adapter=partial(
                        _custom_adapter_for_hf,
                        source=f"common_pile/{name}",
                        id_key=None,
                        language="en",
                        language_key=None,
                        conversation_key=None,
                        remove_keys=[],
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
                logging_dir=f"{DATA_PATH}/common_pile/{name}/logs_hf",
                job_name="hf_cp",
                tasks=5,
                time="20:00:00",
                skip_completed=not args.force,
            )

            hf_executor.run()
