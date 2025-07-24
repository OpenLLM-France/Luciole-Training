import os
import subprocess
import argparse
import glob
import re

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

    job_ids = []
    for name in names:
        if (
            not os.path.isfile(os.path.join(token_dir, "stats", f"{name}.json"))
            or args.force
        ):
            print("--------------------------------------")
            print(f"🚀 Stats for: {name}")
            print("--------------------------------------")
            result = subprocess.run(
                [
                    "sbatch",
                    f"--job-name=stats_{name}",
                    "template.slurm",
                    "extract_stats.py",
                    token_dir,
                    name,
                ],
                capture_output=True,
                text=True,
            )
            # Parse job ID from output: "Submitted batch job 123456"
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
            else:
                print("❌ sbatch failed:", result.stderr)
        else:
            print("--------------------------------------")
            print(f"⏩ Skipping {name}")
            print("--------------------------------------")

    if job_ids:
        dependency = ":".join(job_ids)
        subprocess.run(
            [
                "sbatch",
                "--job-name=merge_stats",
                f"--dependency=afterok:{dependency}",
                "template.slurm",
                "merge_stats.py",
            ]
        )
    else:
        print("✅ No stats jobs were submitted. Running merge job directly.")
        subprocess.run(
            [
                "sbatch",
                "--job-name=merge_stats",
                "template.slurm",
                "merge_stats.py",
            ]
        )
