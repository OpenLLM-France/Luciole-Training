from distilabel.models.llms import TransformersLLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import StepResources
from datetime import datetime
import random
import datasets
import os
import argparse
import psutil
import os

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[{tag}] Memory usage: {mem_mb:.2f} MB")

chat_template = """{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{{- '<|im_start|>assistant\n' }}
{{- '<think>\n\n</think>\n\n' }}"""

main_path = os.getenv("OpenLLM_OUTPUT")

def to_shorthand(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}k"
    else:
        return str(n)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name",
        type=str,
        default="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B-Instruct",
        help="Model you want to use. It can be on HF or local."
    )
    argparser.add_argument(
        "--data_language",
        type=str,
        default="en",
        choices=['en', 'fra_Latn', 'spa_Latn', 'ita_Latn', 'deu_Latn'],
        help='Dataset language. "en" corresponds to fineweb-edu. Otherwise, from fineweb-2.'
    )
    argparser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help='Number of samples you want to annotate. Usually, we need 400k samples to train a fasttext classfier.'
    )
    argparser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help='Number of gpus to use. It will use tensor parallelism, then data parallelism.'
    )
    argparser.add_argument(
        "--prompt_name",
        type=str,
        default="en",
        help='Name of the prompt you want to use. Prompts are defined in "prompt/". You can add new ones. Use the <text> to insert the web page extract (first 2000 characters will be used).'
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "synthetic_data"),
        help="Output directory. Name of the dataset is generated automatically."
    )
    argparser.add_argument(
        "--use_cache",
        action="store_true",
        help="Activate if you want to use cache. The process may be stuck when activated..."
    )
    argparser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable the thinking process for qwen model."
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
    output_name = f"{model_name.split('/')[-1]}_{prompt_name}_{data_language}_{to_shorthand(args.nsamples)}_{date}"
    if (not args.disable_thinking) and ("Qwen" in model_name):
        output_name += "_think" 
    print(output_name)

    with open(f"prompt/{prompt_name}.txt", 'r', encoding='utf-8') as file:
        prompt = file.read()

    if data_language == "en":
        data_path = "HuggingFaceFW/fineweb-edu-llama3-annotations"
    elif data_language in ['fra_Latn', 'spa_Latn', 'ita_Latn', 'deu_Latn']:
        data_path = os.path.join(main_path, f"data/raw_datasets_ablation/fineweb2/data/{data_language}/train")
    else:
        raise ValueError("Unsupported data language. Use 'en' or 'fra_Latn'.")

    # Preprocess dataset
    random.seed(42)  # Set seed for reproducibility
    dataset = datasets.load_dataset(data_path, split="train")
    random_indices = random.sample(range(len(dataset)), args.nsamples)
    dataset = dataset.select(random_indices)
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace('<text>', x["text"][:2000])}, num_proc=4
    )
    dataset = dataset.select_columns(["text", "instruction"])
    print_memory_usage("After loading dataset")

    # Define the pipeline
    with Pipeline(output_name) as pipeline:
        chat_template = chat_template if args.disable_thinking else None
        if args.vllm:
            llm = vLLM(
                model = model_name,
                chat_template = chat_template,
                extra_kwargs={
                    "tensor_parallel_size": args.gpus,               # Number of GPUs per node
                    "gpu_memory_utilization": 0.95,  # GPU memory utilization
                },
                generation_kwargs={
                    "temperature": 0.5,
                    "max_new_tokens": 1024,
                },
            )
        else:
            llm = TransformersLLM(
                model = model_name,
                chat_template = chat_template,
                model_kwargs = {},
                generation_kwargs={
                    "temperature": 0.5,
                    "max_new_tokens": 1024,
                },
            )
        generation = TextGeneration(
            llm = llm,
            input_batch_size = 50,
            resources=StepResources(replicas=1, gpus=args.gpus),
        )

    distiset = pipeline.run(dataset=dataset, use_cache=args.use_cache)

    distiset.save_to_disk(
        os.path.join(output_dir, output_name),
        save_card=True,
        save_pipeline_config=True,
        save_pipeline_log=True
    )
