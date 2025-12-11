import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import LambdaFilter


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
    )

    # main_processing_executor.run()

    ############
    # Push to Hub
    ############
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import HuggingFaceDatasetWriter
    from utils import _custom_adapter_for_hf, HF_SCHEMA
    from functools import partial

    def fix_data(data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        from web_utils import map_language_to_iso

        map_language_to_iso = {
            k.split("_")[0]: v for k, v in map_language_to_iso.items()
        }
        for doc in data:
            doc.metadata.pop("targets")
            doc.metadata["language"] = map_language_to_iso.get(
                doc.metadata.pop("language_code")
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
                language=None,
                language_key="language",
                conversation_key=None,
                remove_keys=[],
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
        depends=None if args.push_only else main_processing_executor,
    )

    hf_executor.run()
