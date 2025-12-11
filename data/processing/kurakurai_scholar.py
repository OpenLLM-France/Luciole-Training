import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline


def process_documents(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for doc in data:
        doc.metadata["messages"] = doc.text
        doc.text = "\n".join([x["content"] for x in doc.text]).strip()
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    DATA_PATH = args.data_path
    dataset_name = "kurakurai_scholar"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        HuggingFaceDatasetReader(
            "kurakurai/scholar",
            {"name": "default", "split": "scholar_all"},
            streaming=True,
            text_key="messages",
        ),
        process_documents,
        JsonlWriter(
            os.path.join(output_path, "data"),
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=os.path.join(output_path, "logs"),
        job_name=dataset_name,
        tasks=20,
    )

    main_processing_executor.run()
