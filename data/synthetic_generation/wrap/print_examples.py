from datasets import load_from_disk, load_dataset
import argparse
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--expe_path",
        type=str,
    )
    argparser.add_argument(
        "--from_parquet", action="store_true", help="read from parquet files."
    )
    args = argparser.parse_args()
    expe_path = args.expe_path

    if args.from_parquet:
        ds = load_dataset(
            "parquet", data_files={"train": os.path.join(expe_path, "*.parquet")}
        )["train"]
    else:
        ds = load_from_disk(expe_path)

    for i, data in enumerate(ds, 0):
        print(f"\n---\n## Example {i}")
        print(data["text"])
        print("\n---\n")
        print(data["generation"])
