from distilabel.models.llms import TransformersLLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import LoadDataFromDicts, LoadDataFromDisk
from distilabel.steps import StepResources
from datetime import datetime
import random
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
        "--data_language",
        type=str,
        default="fr",
    )
    argparser.add_argument(
        "--prompt_name",
        type=str,
        default="en",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "synthetic_data"),
    )
    argparser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable the thinking process."
    )
    argparser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM."
    )
    args = argparser.parse_args()
    model_name = args.model_name
    data_language = args.data_language
    prompt_name = args.prompt_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    if not args.disable_thinking and "Qwen" in model_name:
        output_name = f"{model_name.split('/')[-1]}_{prompt_name}_think_{date}"
    else:
        output_name = f"{model_name.split('/')[-1]}_{prompt_name}_{date}"
    print(output_name)

    with open(f"prompt/{prompt_name}.txt", 'r', encoding='utf-8') as file:
        prompt = file.read()

    if data_language == "fr":
        data_path = os.path.join(main_path, "data/raw_datasets_ablation/fineweb2/data/fra_Latn/train")
    elif data_language == "en":
        data_path = "HuggingFaceFW/fineweb-edu-llama3-annotations"
    else:
        raise ValueError("Unsupported data language. Use 'fr' or 'en'.")

    # Preprocess dataset
    random.seed(42)  # Set seed for reproducibility
    dataset = datasets.load_dataset(data_path, split="train")
    random_indices = random.sample(range(len(dataset)), 100)
    dataset = dataset.select(random_indices)
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace('<text>', x["text"])},
        # remove_columns=dataset.column_names,
    )

    # Define the pipeline
    with Pipeline(name="annotation") as pipeline:
        chat_template = chat_template if args.disable_thinking else None
        if args.vllm:
            llm = vLLM(
                model = model_name,
                chat_template = chat_template,
                extra_kwargs = {}
            )
        else:
            llm = TransformersLLM(
                model = model_name,
                chat_template = chat_template,
                model_kwargs = {}
            )
        generation = TextGeneration(
            llm = llm,
            input_batch_size = 8,
            # resources=StepResources(replicas=1, cpus=1, gpus=4)
        )

    distiset = pipeline.run(dataset=dataset, use_cache=False)

    distiset.save_to_disk(
        os.path.join(output_dir, f'{data_language}_data', output_name),
        save_card=True,
        save_pipeline_config=True,
        save_pipeline_log=True
    )
