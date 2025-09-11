import argparse
import gc
import torch
from transformers import AutoModel


def count_parameters(repo_id: str):
    print(f"\n>>> Loading {repo_id} to count parameters")
    model = AutoModel.from_pretrained(
        repo_id,
        low_cpu_mem_usage=True,  # use less RAM when loading
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {repo_id}")
    print(f"  → Parameters: {num_params:,}")

    # Free memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return num_params


def main():
    parser = argparse.ArgumentParser(
        description="Count parameters in Hugging Face models"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="List of Hugging Face model repo IDs (e.g. allenai/OLMo-2-0425-1B)",
    )
    args = parser.parse_args()

    for repo_id in args.models:
        count_parameters(repo_id)


if __name__ == "__main__":
    main()
