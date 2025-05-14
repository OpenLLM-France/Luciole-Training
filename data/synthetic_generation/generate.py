from distilabel.models import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import LoadDataFromDicts, LoadDataFromDisk

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
        "--prompt",
        type=str,
        default="prompt/en.txt",
    )
    argparser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable the thinking process."
    )
    args = argparser.parse_args()

    with open(args.prompt, 'r', encoding='utf-8') as file:
        prompt = file.read()

    # Preprocess dataset
    dataset = datasets.load_dataset(
        os.path.join(main_path, "data/raw_datasets/fineweb2/data/fra_Latn/train"), split='train'
    )
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace('<extrait>', x["text"])},
        remove_columns=dataset.column_names,
    )
    dataset = dataset.select(range(5))

    # Define the pipeline
    with Pipeline(name="annotation") as pipeline:
        generation = TextGeneration(
            llm=TransformersLLM(
                model="Qwen/Qwen3-0.6B",
                generation_kwargs={"temperature": 0.8, "max_new_tokens": 512},
                chat_template = chat_template if args.disable_thinking else None,
            )
        )

    distiset = pipeline.run(dataset=dataset, use_cache=False)

    distiset['default']['train'].to_pandas().to_csv("data.csv")