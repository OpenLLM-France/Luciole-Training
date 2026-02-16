import pandas as pd
import os
import subprocess
import glob

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Subsample MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=str, help="Output folder", default="test")
    parser.add_argument(
        "--repeats_path",
        type=str,
        default="chronicles/luciole_training_v2/phase3/repeats.csv",
        help="Path to the repeats CSV file",
    )
    parser.add_argument(
        "--root_data_path",
        type=str,
        default=os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), "data/tokenized_data/tokens_lucie2"),
        help="Path to the root data folder",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode, process only first 10 documents",
    )

    args = parser.parse_args()

    repeats_df = pd.read_csv(args.repeats_path)

    for index, row in repeats_df.iterrows():
        repeat = row["repeat"]
        if repeat <= 0:
            continue
        name = row["name"]
        data_path = os.path.join(args.root_data_path, f"{name}_text_document")

        if os.path.exists(os.path.join(args.output, "completion", name)):
            print("--------------------------------------")
            print(f"⏩ Skipping {name}, output files already exist.")
            print("--------------------------------------")
            continue

        pattern = os.path.join(args.output, "data", f"{name}*")
        if glob.glob(pattern):
            print("--------------------------------------")
            print(f"⚠️  Warning for {name}!")
            print("--------------------------------------")
            continue

        print("--------------------------------------")
        print(f"🚀 Split long context for: {name} with sampling {repeat}")
        print("--------------------------------------")
        result = subprocess.run(
            [
                "sbatch",
                f"--job-name=long_{name}",
                "template.slurm",
                "long_context_splitting.py",
                data_path,
                args.output,
                "--sample_rate",
                str(repeat),
                "--debug" if args.debug else "",
            ],
            capture_output=True,
            text=True,
        )

        if args.debug:
            break
