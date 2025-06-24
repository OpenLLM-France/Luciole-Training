from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    PerplexityFilter,
    ExtremeTokenizerFilter,
    LanguageFilter,
)

# from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.filters.prefix_formatter import PrefixFormatter
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


def additionnal_formatting(doc):
    out = {}
    newspaper_name = doc.metadata.get("newspaper_name")
    if newspaper_name:
        out["newspaper_name"] = newspaper_name
    return out


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "american_stories"
    if args.local:
        SUPPORTED_YEARS = ["1770", "1964"]

    for year in SUPPORTED_YEARS:
        year = str(year)

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
            ExtremeTokenizerFilter(
                tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional",
                min_token_per_char=0,
                max_token_per_char=0.4,
                filter_mode="CHUNKS",
                replace_span="\n\n[...]\n\n",
                removed_spans_in_metadata=False,  # FOR DEBUGGING only
                exclusion_writer=JsonlWriter(
                    f"{output_path}/removed/extreme_tokenizer/{year}"
                ),
            ),
            LanguageFilter(
                languages=["en", "fr", "it", "de", "es", "ar", "pt", "nl"],
                language_threshold=0.65,
                keep_top_pairs_threshold=1,
                exclusion_writer=JsonlWriter(f"{output_path}/removed/language/{year}"),
            ),
            PerplexityFilter(
                language_from_metadata=True,
                min_ppl=10,
                max_ppl=2000,
                exclusion_writer=JsonlWriter(f"{output_path}/removed/ppl/{year}"),
            ),
            # FastTextClassifierFilter(
            #     model_url = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin",
            #     save_labels_in_metadata=True,
            #     newline_replacement = " "
            # ),
            PrefixFormatter(
                date_format="%Y-%m-%d",
                additionnal_formatting=lambda doc: additionnal_formatting(doc),
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
            logging_dir=f"{output_path}/logs/{year}",
            job_name=dataset_name,
            tasks=1,
        )
        main_processing_executor.run()
