import os
import subprocess
import argparse
import glob
import re
import json
import pandas as pd
import time


def merge_stats(data_path):
    # Get only JSON filenames
    json_files = [
        file for file in os.listdir(data_path) if file.lower().endswith(".json")
    ]

    data_list = []
    for json_file in json_files:
        with open(os.path.join(data_path, json_file), "r") as f:
            stats = json.load(f)
            stats = {"name": os.path.splitext(json_file)[0], **stats}
            data_list.append(stats)

    df = pd.DataFrame(data_list).sort_values(by="name")
    return df


def wait_for_jobs(job_prefix, poll_interval=30):
    print("⏳ Waiting for Slurm jobs to finish...")
    while True:
        result = subprocess.run(
            ["squeue", "--user", os.getenv("USER"), "--name", job_prefix, "--noheader"],
            stdout=subprocess.PIPE,
            text=True,
        )
        if result.stdout.strip() == "":
            print("✅ All jobs finished.")
            break
        else:
            print("⏳ Still running...")
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "token_dir",
        type=str,
        help="Directory that contains all your tokenized datasets (.idx)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of stats even if they already exist.",
    )
    args = parser.parse_args()
    token_dir = args.token_dir

    files = glob.glob(os.path.join(token_dir, "*_text_document.idx"))
    names = [
        re.match(r"(.*?)_text_document\.idx", os.path.basename(f)).group(1)
        for f in files
    ]

    for name in names:
        if (
            not os.path.isfile(os.path.join(token_dir, "stats", f"{name}.json"))
            or args.force
        ):
            print("--------------------------------------")
            print(f"🚀 Stats for: {name}")
            print("--------------------------------------")
            subprocess.run(
                [
                    "sbatch",
                    f"--job-name=stats_{name}",
                    "extract_stats.slurm",
                    token_dir,
                    name,
                ]
            )
        else:
            print("--------------------------------------")
            print(f"⏩ Skipping {name}")
            print("--------------------------------------")

    wait_for_jobs("stats_")
    merged_df = merge_stats(os.path.join(token_dir, "stats"))
    output_csv_path = os.path.join(token_dir, "stats/all_stats_merged.csv")
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged stats saved to {output_csv_path}")
