import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
import re
import random

def append_input_output(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        if random.random() < 0.5:
            answer = document.metadata["A"]
            problem_name = random.choice(["Problem:", "Question:", "Prompt:", ""])
            solution_name = random.choice(["Answer:", "Solution:", "Final Answer:"])
            document.text = (problem_name + "\n"+ document.text + "\n" + solution_name + "\n" + answer).strip()
        else:
            problem_name, answer_name = random.choice(
                [("User:", "Assistant:"),
                 ("user", "assistant"),
                 ("", "")
                ])
            document.text = (problem_name + "\n"+ document.text + "\n" + answer_name + "\n" + document.metadata["A"]).strip()

        document.metadata.pop("A")
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "stackmathqa1600k"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        JsonlReader(
            data_folder="hf://datasets/math-ai/StackMathQA/data/stackmathqa1600k",
            glob_pattern="*.jsonl",
            text_key="Q",
        ),
        append_input_output,
        JsonlWriter(f"{output_path}/data"),
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
