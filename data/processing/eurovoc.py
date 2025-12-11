from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    ExtremeTokenizerFilter,
    LambdaFilter,
)
from datatrove.pipeline.formatters.eurovoc_formatter import EurovocFormatter

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
        ExtremeTokenizerFilter(
            tokenizer_name_or_path="OpenLLM-BPI/tokenizer_128k-arab-regional_v2",
            max_token_per_char=0.38,
            normalize_digits=True,
            mode="CHUNKS",
            min_length=1000,
            max_length=2000,
            separator=("\n", ". ", ", ", " "),
            replace_span="\n\n[...]\n\n",
            removed_spans_in_metadata=False,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/eurovoc_filtered_v2/removed/chunk_extreme_tokenizer",
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/eurovoc_filtered_v2/data",
            output_filename="${lang}/${rank}.jsonl.gz",
        ),
    ]

    filter_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/eurovoc_filtered_v2/logs",
        job_name="eurovoc",
        tasks=20,
        partition="prepost",
        time="20:00:00",
        cpu_per_task=4,
    )

    filter_executor.run()
