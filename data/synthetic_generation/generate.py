from distilabel.models import TransformersLLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import LoadDataFromDicts, LoadDataFromDisk
from distilabel.steps import StepResources

import datasets
import os
import argparse

chat_template = """{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{{- '<|im_start|>assistant\n' }}
{{- '<think>\n\n</think>\n\n' }}"""

main_path = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name",
        type=str,
        default="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct",
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(main_path, "data/raw_datasets_ablation/fineweb2/data/fra_Latn/train"),
    )
    argparser.add_argument(
        "--prompt",
        type=str,
        default="prompt/en.txt",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="out",
    )
    argparser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable the thinking process."
    )
    args = argparser.parse_args()
    model_name = args.model_name
    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.prompt, 'r', encoding='utf-8') as file:
        prompt = file.read()

    # Preprocess dataset
    dataset = datasets.load_dataset(data_path, split="train").select(range(100))  # Select a subset for testing
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace('<text>', x["text"])},
        remove_columns=dataset.column_names,
    )

    # Define the pipeline
    with Pipeline(name="annotation") as pipeline:
        generation = TextGeneration(
            llm=TransformersLLM(
                model=model_name,
                generation_kwargs={"temperature": 0.8, "max_new_tokens": 512},
                chat_template = chat_template if args.disable_thinking else None,
            ),
            input_batch_size=8,
            # resources=StepResources(replicas=1, cpus=1, gpus=4)
        )

    distiset = pipeline.run(dataset=dataset, use_cache=False)

    distiset['default']['train'].to_json(os.path.join(output_dir, "data.jsonl"))