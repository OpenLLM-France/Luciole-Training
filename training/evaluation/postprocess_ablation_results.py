import os
import argparse
from utils import read_experiment_results, read_datamix
import pandas as pd
import numpy as np

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expe_dir",
        type=str,
    )
    args = parser.parse_args()

    metric = "word_perplexity"
    os.makedirs(os.path.join(args.expe_dir, "out"), exist_ok=True)

    # READ RESULTS
    out = []
    for expe_name in os.listdir(args.expe_dir):
        if os.path.isdir(os.path.join(args.expe_dir, expe_name)) and expe_name != "out":
            expe_path = os.path.join(args.expe_dir, expe_name)

            # Read datamix
            datamix = read_datamix(expe_path)
            datamix = {"datamix:" + d["name"]: d["weight"] for d in datamix}

            # Read results
            df_results = read_experiment_results(expe_path)
            if df_results is None:
                continue
            df_results = df_results[df_results["metric"] == metric]
            df_results = df_results[np.isclose(df_results["steps"], 1000, atol=10)]

            # df_results = df_results[round(df_results["tokens"], 1) == tokens]
            results = df_results[["task", "score"]].to_dict(orient="records")
            results = {"target:" + d["task"]: d["score"] for d in results}

            out.append({**datamix, **results})

    df = pd.DataFrame(out)

    df.loc[:, df.columns.str.startswith("datamix:")] = df.loc[
        :, df.columns.str.startswith("datamix:")
    ].fillna(0)
    df = df.dropna(subset=df.columns[df.columns.str.startswith("target:")])
    df = df[sorted(df.columns)]

    df.to_csv(os.path.join(args.expe_dir, "out", "regmix_results.csv"))
