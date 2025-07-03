from datasets import load_from_disk
import argparse
import os
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
from utils import extract_educational_json


def plot_label_crosstab(data):
    df = pd.DataFrame(data)
    # Cross-tabulation (contingency table)
    cross_tab = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])
    print("\nCross-tabulation:")
    print(cross_tab.to_string())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "expe_1",
        type=str,
    )
    argparser.add_argument(
        "expe_2",
        type=str,
    )
    argparser.add_argument(
        "--key",
        type=str,
        default="educational_score",
    )
    args = argparser.parse_args()
    expe_1 = args.expe_1
    expe_2 = args.expe_2
    key = args.key

    out = {}
    for expe_path in [expe_1, expe_2]:
        print(f"\nExperiment path: {expe_path}")
        ds = load_from_disk(os.path.join(expe_path, "default"))["train"]
        print("\nExample generation:")
        for i in range(3):
            print(f"Example {i}: {ds[i]['generation']}")
        ds = ds.map(lambda x: extract_educational_json(x["generation"], keys=[key]))
        out[expe_path.split("/")[-1]] = ds[key]

    plot_label_crosstab(out)
