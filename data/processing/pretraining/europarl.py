from utils import create_parser, parse_args, create_executor

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    def get_language(data, rank: int = 0, world_size: int = 1):
        import re

        for doc in data:
            m = re.search(r"europarl_(.*?)_v", doc.id)
            language = m.group(1) if m else "unknown"
            language_parts = language.split("-")
            if len(language_parts) == 2:
                if language_parts[1] == "en":
                    language = language_parts[1] + "-" + language_parts[0]
            doc.metadata["language"] = language
            yield doc

    pipeline = [
        JsonlReader(
            f"{DATA_PATH}/europarl",
        ),
        get_language,
        HuggingFaceDatasetWriter(
            dataset="OpenLLM-BPI/Luciole-Training-Dataset"
            + ("-debug" if args.debug else ""),
            private=True,
            local_working_dir=f"{DATA_PATH}/europarl_hf/data_hf",
            output_filename="data/europarl/${language}/${rank}.parquet",
            adapter=partial(
                _custom_adapter_for_hf,
                source="europarl",
                id_key=None,
                language=None,
                language_key="language",
                conversation_key=None,
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
        logging_dir=f"{DATA_PATH}/europarl_hf/logs_hf",
        job_name="hf_europarl",
        tasks=1,
        skip_completed=not args.force,
    )

    hf_executor.run()
