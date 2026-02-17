import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline


def append_input_output(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for document in data:
        answer = document.metadata["generated_solution"]
        document.text = (document.text + "\n" + answer).strip()
        yield document


if __name__ == "__main__":
    parser = create_parser()

    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "open_math_instruct"
    output_path = os.path.join(DATA_PATH, dataset_name)
    input_path = os.path.join(DATA_PATH, "open_math_instruct_hf/train_omi.jsonl")

    pipeline = [
        JsonlReader(
            input_path,  # "nvidia/OpenMathInstruct-1", #"hf://datasets/nvidia/OpenMathInstruct-1"
            text_key="question",
        ),
        append_input_output,
        JsonlWriter(f"{output_path}/output"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
    )

    main_processing_executor.run()
