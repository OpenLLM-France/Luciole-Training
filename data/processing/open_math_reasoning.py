import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
import re
import random

def split_thinking(output):
    think_pattern = r"<think>(.*?)</think>"
    thinking = re.search(think_pattern, output, flags=re.DOTALL)
    if thinking:
        thoughts = thinking.group(1).strip() 
        remaining_text = re.sub(think_pattern, "", output, flags=re.DOTALL).strip()
    else:
        thoughts = ""
        remaining_text = output.strip()

    return thoughts, remaining_text

def append_input_output(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        if random.random() < 0.5:
            thoughts, answer = split_thinking(document.text)
            thoughts_name = random.choice(["Problem discussion:", "Thinking:"])
            solution_name = random.choice(["Proposed solution:", "Solution:", "Final Answer:"])
            if thoughts:
                document.text = (thoughts_name + "\n" + thoughts + "\n" + solution_name + "\n" + answer).strip()
            else:
                document.text = answer
        else:
            document.text = document.text.strip()

        yield document

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "open_math_reasoning"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/nvidia/OpenMathReasoning/data", glob_pattern = "[ct]*.parquet",
            text_key="generated_solution",
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
