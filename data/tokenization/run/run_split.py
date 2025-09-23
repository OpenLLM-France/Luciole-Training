import pandas as pd
import os
import subprocess


def load_data(expe_path, phase="phase1"):
    phase_path = os.path.join(expe_path, phase, "repeats.csv")
    assert os.path.exists(phase_path), f"File not found: {phase_path}"
    df = pd.read_csv(phase_path)
    df["modulo"] = df["repeat"] % 1
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "token_dir",
        type=str,
    )
    parser.add_argument(
        "expe_path",
        type=str,
    )
    args = parser.parse_args()
    expe_path = args.expe_path
    token_dir = args.token_dir

    df_phase1 = load_data(expe_path, phase="phase1")
    df_phase2 = load_data(expe_path, phase="phase2")

    df = df_phase1.merge(df_phase2, on="name", suffixes=("_phase1", "_phase2"))

    df = df[(df["modulo_phase1"] > 0) & (df["modulo_phase2"] > 0)]
    df["remainder"] = 1 - df["modulo_phase1"] - df["modulo_phase2"]
    assert (df["remainder"] >= 0).all(), (
        f"This script does not support your datamix yet. "
        f"Invalid rows:\n{df[df['remainder'] < 0]}"
    )

    for _, row in df.iterrows():
        print()
        name = row["name"]
        ratio = (row["modulo_phase1"], row["modulo_phase2"])
        print(f">> {name}")
        print(f"Split: {ratio[0]:.3f} - {ratio[1]:.3f} - {row['remainder']:.3f}")

        print("--------------------------------------")
        print(f"🚀 Splitting dataset: {name}")
        print(f"-> Ratio: {ratio}")
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
            ]
        )
