import os
import subprocess
import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file that contains the datasets you want to tokenize. See for example configs/ablations_v0.yaml.",
    )
    args = parser.parse_args()
    yaml_file = args.yaml_file

    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    for entry in yaml_data:
        inputs = entry["input"]
        output = entry["output"]

        if not os.path.isfile(output + ".idx"):
            if os.path.isfile(output + ".bin"):
                print("--------------------------------------")
                print(
                    f"⚠️  Warning! Found a .bin file at {output}, but no .idx file. Either a job has failed or is still running."
                )
                print("--------------------------------------")
            else:
                print("--------------------------------------")
                print(f"🚀 Concatening: {os.path.basename(output)}")
                print("--------------------------------------")

                # Submit job using sbatch
                subprocess.run(
                    [
                        "sbatch",
                        "--job-name=concat",
                        "template.slurm",
                        "concat_tokens.py",
                        *inputs,
                        output,
                    ]
                )
        else:
            print("--------------------------------------")
            print(f"✅  Already done: {os.path.basename(output)}")
            print("--------------------------------------")
