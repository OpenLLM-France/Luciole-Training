import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
import re
import random
from datatrove.pipeline.writers import HuggingFaceDatasetWriter
from utils import _custom_adapter_for_hf, HF_SCHEMA
from functools import partial


def split_thinking(output):
    # get text in between <think> and </think> tags, and the remaining text
    think_pattern = r"<think>(.*?)</think>"
    thoughts = re.findall(think_pattern, output, flags=re.DOTALL)[0]
    remaining_text = re.sub(think_pattern, "", output, flags=re.DOTALL).strip()
    return thoughts, remaining_text


def append_input_output(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    for document in data:
        if random.random() < 0.5:
            thoughts, answer = split_thinking(document.metadata["output"])
            problem_name = random.choice(["Problem:", "Question:", "Prompt:", ""])
            thoughts_name = random.choice(["Thoughts:", "Reasoning:", "Thinking:"])
            solution_name = random.choice(["Answer:", "Solution:", "Final Answer:"])
            document.text = (
                problem_name
                + "\n"
                + document.text
                + "\n"
                + thoughts_name
                + "\n"
                + thoughts
                + "\n"
                + solution_name
                + "\n"
                + answer
            ).strip()
        else:
            problem_name, answer_name = random.choice(
                [("User:", "Assistant:"), ("user", "assistant"), ("", "")]
            )
            document.text = (
                problem_name
                + "\n"
                + document.text
                + "\n"
                + answer_name
                + "\n"
                + document.metadata["output"]
            ).strip()

        document.metadata.pop("output")
        yield document


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "open_code_reasoning"
    output_path = os.path.join(DATA_PATH, dataset_name)

    if not args.push_only:
        pipeline = [
            ParquetReader(
                "hf://datasets/nvidia/OpenCodeReasoning/split_0",
                text_key="input",
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

    else:
        pipeline = [
            JsonlReader(
                f"{output_path}/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-BPI/Luciole-Training-Dataset"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{output_path}/data_hf",
                output_filename="data/open_code_reasoning/en/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="open_code_reasoning",
                    id_key=None,
                    reset_id=False,
                    language="en",
                    language_key=None,
                    conversation_key=None,
                    remove_keys=["solution"],
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
            job_name="hf_opencode",
            tasks=1,
        )

        hf_executor.run()
