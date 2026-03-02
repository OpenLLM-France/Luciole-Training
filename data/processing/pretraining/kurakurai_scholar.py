from utils import create_parser, parse_args, create_executor, add_sampler_filter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


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

    if not args.push_only:
        pipeline = [
            HuggingFaceDatasetReader(
                "kurakurai/scholar",
                {"name": "default", "split": "scholar_all"},
                streaming=True,
                text_key="messages",
            ),
            process_documents,
            JsonlWriter(f"{DATA_PATH}/kurakurai_scholar/data"),
        ]
        add_sampler_filter(pipeline, args.sample_rate)

        main_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/kurakurai_scholar/logs",
            job_name="kurakurai_scholar",
            tasks=20,
        )
        main_executor.run()

    else:
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/kurakurai_scholar/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/kurakurai_scholar/data_hf",
                output_filename="data/scholar/fr/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="scholar",
                    id_key=None,
                    reset_id=True,
                    language="fr",
                    language_key=None,
                    conversation_key="messages",
                    remove_keys=["dataset"],
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
            logging_dir=f"{DATA_PATH}/kurakurai_scholar/logs_hf",
            job_name="hf_scholar",
            tasks=1,
            skip_completed=not args.force,
        )

        hf_executor.run()
