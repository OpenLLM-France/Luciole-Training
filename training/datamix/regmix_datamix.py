import os
import argparse
import pandas as pd
import json
import numpy as np


def to_nb_tokens(x):
    x = x.replace("b", " * 1_000_000_000")
    x = x.replace("m", " * 1_000_000")
    try:
        return int(eval(x))
    except Exception as e:
        raise ValueError(f"Invalid value for --mode: {x} (a number of tokens)") from e


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(main_path, "data/tokenized_data/tokens_training_v2"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "ablations/regmix"),
        help="Output directory",
    )

    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args()

    for seed in range(50):
        np.random.seed(seed)

        # Load your data (only if not just --help)
        df = pd.read_csv(os.path.join(args.data_path, "stats/all_stats_merged.csv"))
        df = df[["name", "total_tokens"]]

        n_datasets = len(df)

        lambda_param = np.random.uniform(0.1, 5)
        df["lambda"] = lambda_param
        df["weight"] = np.random.dirichlet(lambda_param * df["total_tokens"])

        df["name"] = df["name"] + "_text_document"
        out = {
            "data_path": args.data_path,
            "train": df[["name", "weight"]].to_dict(orient="records"),
        }

        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/regmix_{seed}.json", "w") as f:
            json.dump(out, f, indent=4)

        df.to_csv(f"{args.output_dir}/regmix_{seed}_stats.csv")
