from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    LanguageFilter,
    ExtremeTokenizerFilter,
    PerplexityFilter,
    LambdaFilter,
)
from datatrove.pipeline.formatters.eurovoc_formatter import EurovocFormatter
from datatrove.pipeline.split_and_merge import SplitDocument, MergeDocument

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    pipeline = [
        JsonlReader(
            "hf://datasets/EuropeanParliament/Eurovoc/files",
        ),
        LambdaFilter(
            lambda doc: doc.metadata["lang"]
            in ["ara", "cat", "deu", "eng", "fra", "ita", "nld", "por", "spa", "eus"],
        ),
        EurovocFormatter(),
        SplitDocument(
            min_length=1000,
            max_length=2000,
            separator=("\n", ". ", ", ", " "),
        ),
        LanguageFilter(
            keep_top_pairs_threshold=1,
            languages=[
                "en",
                "fr",
                "it",
                "de",
                "es",
                "ar",
                "pt",
                "nl",
                "ca",
                "eu",
            ],
            language_threshold=0.5,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/eurovoc_filtered/removed/ft176"),
        ),
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
            max_token_per_char=0.38,
            normalize_digits=True,
            mode="DOCUMENT",
            batch_size=10000,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/eurovoc_filtered/removed/chunk_extreme_tokenizer",
            ),
        ),
        PerplexityFilter(
            use_ccnet=True,
            model_dataset="",
            language_from_metadata=True,
            min_ppl=10.0,
            max_ppl=2500,
            exclusion_writer=JsonlWriter(f"{DATA_PATH}/eurovoc_filtered/removed/ppl"),
        ),
        MergeDocument(
            min_character_ratio=0.5,
            min_words=50,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/eurovoc_filtered/removed/doc_filtered",
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/eurovoc_filtered/data",
            output_filename="${lang}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/eurovoc_filtered/logs",
        job_name="eurovoc_filtered",
        tasks=50,
        partition="cpu_p1",
        time="20:00:00",
        cpu_per_task=4,
    )

    filter_executor.run()
