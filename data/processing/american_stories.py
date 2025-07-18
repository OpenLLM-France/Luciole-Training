from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    PerplexityFilter,
    ExtremeTokenizerFilter,
    LanguageFilter,
)
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
from datatrove.pipeline.split_and_merge import SplitDocument, MergeDocument
import os

SUPPORTED_YEARS = [
    "1770",
    "1771",
    "1772",
    "1773",
    "1774",
    "1777",
    "1778",
    "1779",
    "1791",
    "1792",
    "1793",
] + [str(year) for year in range(1796, 1964 + 1)]


def chunk_list(lst, chunk_size=50):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


SUPPORTED_YEARS_GROUPED = chunk_list(SUPPORTED_YEARS)


def additionnal_formatting(doc):
    out = {}
    newspaper_name = doc.metadata.get("newspaper_name")
    if newspaper_name:
        out["newspaper_name"] = newspaper_name
    return out


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        nargs="+",
        choices=SUPPORTED_YEARS,
        help="Subset to load",
    )
    parser.add_argument(
        "--all",
        action="store_true",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=None,
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "american_stories"

    if args.all:
        years = SUPPORTED_YEARS
    elif args.group is not None:
        years = SUPPORTED_YEARS_GROUPED[args.group]
    else:
        years = args.year

    for year in years:
        output_path = os.path.join(DATA_PATH, dataset_name)
        pipeline = [
            HuggingFaceDatasetReader(
                "dell-research-harvard/AmericanStories",
                {
                    "name": "subset_years",
                    "trust_remote_code": True,
                    "year_list": [year],
                    "split": year,
                },
                text_key="article",
                streaming=True,
            ),
            SplitDocument(separator="\n"),
            ExtremeTokenizerFilter(
                tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
                max_token_per_char=0.35,
                remove_digits=True,
                mode="DOCUMENT",
                batch_size=10000,
                exclusion_writer=JsonlWriter(
                    f"{output_path}/removed/chunk_extreme_tokenizer",
                ),
            ),
            LanguageFilter(
                languages=["en", "fr", "it", "de", "es", "ar", "pt", "nl"],
                language_threshold=0.5,
                keep_top_pairs_threshold=1,
                exclusion_writer=JsonlWriter(f"{output_path}/removed/language/{year}"),
            ),
            PerplexityFilter(
                language_from_metadata=True,
                min_ppl=10,
                max_ppl=2000,
                exclusion_writer=JsonlWriter(f"{output_path}/removed/ppl/{year}"),
            ),
            MergeDocument(),
            PrefixFormatter(
                date_format="%Y-%m-%d",
                additionnal_formatting=additionnal_formatting,
                prefix_pipeline={
                    "newspaper_name": "Newspaper name",
                    "title": "Title",
                    "date": "Date",
                },
            ),
            JsonlWriter(f"{output_path}/data/{year}"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{output_path}/logs/{year}",
            job_name="as",
            tasks=1,
        )
        main_processing_executor.run()
