import argparse
import json
import os

from datasets import load_dataset
from vllm import LLM, SamplingParams


def truncate_at_whitespace(x):
    text = x["text"]
    if len(text) <= 100:
        return {"input": text}
    idx = text.find(" ", 100)
    return {"input": text if idx == -1 else text[:idx]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on French Wikipedia data."
    )
    # Sampling parameters
    parser.add_argument(
        "model_id",
        type=str,
        help="HuggingFace model ID to run inference with (e.g. OpenLLM-France/Luciole-1B-Base)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (default: 1.05)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to take from the dataset (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to write JSONL output files (default: outputs/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )

    dataset = load_dataset(
        "OpenLLM-France/Luciole-Training-Dataset",
        split="train",
        streaming=True,
        data_dir="data/wikimedia/wikipedia/fr",
    )
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    dataset = dataset.take(args.num_samples)
    dataset = dataset.map(truncate_at_whitespace)

    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )

    prompts = [example["text"] for example in dataset]
    outputs = llm.generate(prompts, sampling_params)

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, args.model_id.split("/")[-1] + ".jsonl")
    with open(out_path, "w") as f_out:
        for output in outputs:
            record = {
                "prompt": output.prompt,
                "generated": output.outputs[0].text,
            }
            f_out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
