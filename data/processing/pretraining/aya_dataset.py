import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter
from utils import _custom_adapter_for_hf, HF_SCHEMA

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from functools import partial


def append_input_output(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for document in data:
        answer = document.metadata["targets"]
        document.text = (document.text + "\n" + answer).strip()
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    DATA_PATH = args.data_path
    dataset_name = "aya_dataset"
    output_path = os.path.join(DATA_PATH, dataset_name)

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "CohereLabs/aya_dataset",
                {"name": "default", "split": "train"},
                streaming=True,
                text_key="inputs",
            ),
            LambdaFilter(
                lambda doc: doc.metadata["language_code"]
                in ["arb", "deu", "eng", "fra", "ita", "nld", "por", "spa", "eus"],
            ),
            append_input_output,
            JsonlWriter(
                os.path.join(output_path, "data"),
                output_filename="${language_code}/${rank}.jsonl.gz",
            ),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_processing_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=os.path.join(output_path, "logs"),
            job_name=dataset_name,
            tasks=50,
            skip_completed=not args.force,
        )
        main_processing_executor.run()

    else:

        def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
            from web_utils import LanguageCodes

            for doc in data:
                doc.metadata["language"] = LanguageCodes.iso3_to_iso1(
                    doc.metadata.pop("language_code"), fallback=True
                )
                yield doc

        pipeline = [
            JsonlReader(
                f"{output_path}/data",
            ),
            fix_data,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename="data/aya/" + "${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="aya_dataset",
                    id_key=None,
                    reset_id=True,
                    language=None,
                    language_key="language",
                    conversation_key=None,
                    remove_keys=["targets", "dataset"],
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
            logging_dir=f"{output_path}/logs_hf",
            job_name="hf_aya",
            tasks=1,
            skip_completed=not args.force,
        )

        hf_executor.run()
