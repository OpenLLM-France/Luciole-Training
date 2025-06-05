from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter


def process_metadata(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from slugify import slugify

    for doc in data:
        doc.metadata["open_type"] = slugify(doc.metadata["open_type"])
        doc.metadata["collection"] = slugify(doc.metadata["collection"])
        language_type = doc.metadata["language_type"]
        language = doc.metadata["language"]
        if language is not None:
            language = slugify(language)
            language_ = (
                language if language_type == "Spoken" else slugify(language_type)
            )
        else:
            language_ = "none"
        doc.metadata["language_"] = language_
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    name = "common_corpus"

    pipeline = [
        HuggingFaceDatasetReader(
            "PleIAs/common_corpus",
            {"split": "train"},
            streaming=True,
        ),
        process_metadata,
        JsonlWriter(
            f"{DATA_PATH}/common_corpus/data",
            output_filename="${open_type}/${collection}/${language_}/${rank}.jsonl.gz",
            max_file_size=int(2e9),
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=f"{DATA_PATH}/common_corpus/logs",
        job_name=name,
    )

    main_processing_executor.run()
