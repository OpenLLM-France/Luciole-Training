from distilabel.models.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from distilabel.steps import StepResources
from datetime import datetime
import random
import datasets
import os
import argparse
import psutil
import glob


def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[{tag}] Memory usage: {mem_mb:.2f} MB")


chat_template = """{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{{- '<|im_start|>assistant\n' }}
{{- '<think>\n\n</think>\n\n' }}"""

main_path = os.getenv("OpenLLM_OUTPUT", ".")


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
        "data_path",
        type=str,
        help="Dataset to use",
    )
    argparser.add_argument(
        "expe_name",
        type=str,
        help="Dataset to use",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Name of the dataset is generated automatically.",
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model you want to use. It can be on HF or local.",
    )
    argparser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Number of samples you want to generate",
    )
    argparser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of gpus to use. It will use tensor parallelism, then data parallelism.",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        default="prompt/qa_fr.txt",
        help="Path to the prompt",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling.",
    )
    argparser.add_argument(
        "--use_cache",
        action="store_true",
        help="Activate if you want to use cache. The process may be stuck when activated...",
    )
    argparser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable the thinking process for qwen model.",
    )
    args = argparser.parse_args()
    model_name = args.model_name
    expe_name = args.expe_name
    data_path = args.data_path
    prompt_path = args.prompt_path
    output_dir = args.output_dir
    if output_dir is None:
        print("You did not specify any output directory!")

    # Create name
    prompt_name = filename = os.path.splitext(os.path.basename(prompt_path))[0]
    date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    output_name = f"{expe_name}_{model_name.split('/')[-1]}_{prompt_name}_t{args.temperature}_n{to_shorthand(args.nsamples)}"  # _{date}
    if (not args.disable_thinking) and ("Qwen" in model_name):
        output_name += "_think"
    print(output_name)

    # get prompt
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    # Preprocess dataset
    random.seed(42)  # Set seed for reproducibility
    files = glob.glob(f"{data_path}/*.jsonl.gz")
    dataset = datasets.load_dataset(
        "json",
        data_files=files,
        split="train",
    )
    # Subsampling
    random_indices = random.sample(range(len(dataset)), args.nsamples)
    dataset = dataset.select(random_indices)
    # Apply prompt
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace("<text>", x["text"])},
        num_proc=4,
    )
    dataset = dataset.select_columns(["text", "instruction"])
    print_memory_usage("After loading dataset")

    # Define the pipeline
    with Pipeline(output_name) as pipeline:
        llm = vLLM(
            model=model_name,
            chat_template=chat_template if args.disable_thinking else None,
            extra_kwargs={
                "tensor_parallel_size": args.gpus,  # Number of GPUs per node
                "gpu_memory_utilization": 0.95,  # GPU memory utilization
            },
            generation_kwargs={
                "temperature": args.temperature,
                "max_new_tokens": 1024,
            },
        )
        generation = TextGeneration(
            llm=llm,
            input_batch_size=50,
            resources=StepResources(replicas=1, gpus=args.gpus),
        )

    distiset = pipeline.run(dataset=dataset, use_cache=args.use_cache)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        distiset.save_to_disk(
            os.path.join(output_dir, output_name),
            save_card=True,
            save_pipeline_config=True,
            save_pipeline_log=True,
        )
