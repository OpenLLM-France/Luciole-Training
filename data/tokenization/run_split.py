import pandas as pd
import os
import subprocess
import glob


def load_data(repeats_dir):
    phase_path = os.path.join(repeats_dir, "repeats.csv")
    assert os.path.exists(phase_path), f"File not found: {phase_path}"
    df = pd.read_csv(phase_path)
    df["modulo"] = df["repeat"] % 1
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token_dir",
        default="",
        type=str,
    )
    parser.add_argument("--repeats_dir", type=str, nargs=2, required=True)
    args = parser.parse_args()
    token_dir = args.token_dir
    repeats_dir = args.repeats_dir

    dfs = [load_data(repeats_dir) for repeats_dir in args.repeats_dir]
    df = dfs[0].merge(dfs[1], on="name", suffixes=("_phase1", "_phase2"), how="outer")
    df = df.fillna(0)

    df["remainder"] = (1 - df["modulo_phase1"] - df["modulo_phase2"]) % 1
    df = df[
        (df["modulo_phase1"] > 0) | (df["modulo_phase2"] > 0) | (df["remainder"] > 0)
    ]

    assert (df["remainder"] >= 0).all(), (
        f"This script does not support your datamix yet. "
        f"Invalid rows:\n{df[df['remainder'] < 0]}"
    )

    for _, row in df.iterrows():
        ratio = (row["modulo_phase1"], row["modulo_phase2"], row["remainder"])
        name = row["name"]

        if glob.glob(os.path.join(token_dir, name + "_text_document") + "_ratio*"):
            print()
            print("--------------------------------------")
            print(f"⏩ Skipping dataset: {name}")
            print("--------------------------------------")
        else:
            print()
            print("--------------------------------------")
            print(f"🚀 Splitting dataset: {name}")
            print(f"Ratio: {ratio[0]:.3f} - {ratio[1]:.3f} - {ratio[2]:.3f}")
            print("--------------------------------------")

            subprocess.run(
                [
                    "sbatch",
                    f"--job-name=split_{name}",
                    "template.slurm",
                    "split_tokens.py",
                    os.path.join(token_dir, name + "_text_document"),
                    os.path.join(token_dir, name + "_text_document"),
                    str(ratio[0]),
                    str(ratio[1]),
                    str(ratio[2]),
                ]
            )
