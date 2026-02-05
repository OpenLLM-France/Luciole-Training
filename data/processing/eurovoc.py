from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import (
    ExtremeTokenizerFilter,
    LambdaFilter,
)
from datatrove.pipeline.formatters.eurovoc_formatter import EurovocFormatter
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial
from datatrove.data import DocumentsPipeline

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    if not args.push_only:
        pipeline = [
            JsonlReader(
                "hf://datasets/EuropeanParliament/Eurovoc/files",
            ),
            LambdaFilter(
                lambda doc: doc.metadata["lang"]
                in [
                    "ara",
                    "cat",
                    "deu",
                    "eng",
                    "fra",
                    "ita",
                    "nld",
                    "por",
                    "spa",
                    "eus",
                ],
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

    else:

        def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
            from web_utils import LanguageCodes

            for doc in data:
                language = doc.metadata.pop("lang")
                if language == "ara":
                    doc.metadata["language"] = "ar"  # ISO1 code for Arabic
                else:
                    doc.metadata["language"] = LanguageCodes.iso3_to_iso1(
                        language, fallback=True
                    )
                yield doc

        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/eurovoc_filtered_v2/data",
            ),
            fix_data,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/eurovoc_filtered_v2/data_hf",
                output_filename="data/eurovoc/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="eurovoc",
                    id_key=None,
                    language=None,
                    language_key="language",
                    conversation_key=None,
                    remove_keys=["token_counts", "char_counts", "token_per_chars"],
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
            logging_dir=f"{DATA_PATH}/eurovoc_filtered_v2/logs_hf",
            job_name="hf_eurovoc",
            tasks=10,
            skip_completed=not args.force,
        )

        hf_executor.run()
