import os
import argparse
from utils import read_experiment_results, read_datamix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")


def compute_correlation_matrix(df, features, targets, method="pearson"):
    corr_matrix = pd.DataFrame(index=features, columns=targets)

    for feature in features:
        for target in targets:
            x = df[feature]
            y = df[target]

            if method == "pearson":
                corr = x.corr(y, method="pearson")
            elif method == "spearman":
                corr = x.corr(y, method="spearman")
            elif method == "kendall":
                corr = x.corr(y, method="kendall")
            elif method == "mutual_info":
                # mutual_info_regression needs 2D array for X
                scaler = MinMaxScaler()
                x_scaled = scaler.fit_transform(x.values.reshape(-1, 1))
                y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
                corr = mutual_info_regression(
                    x_scaled, y_scaled, discrete_features=False
                )[0]
            else:
                raise ValueError(f"Unsupported method: {method}")

            corr_matrix.loc[feature, target] = corr

    # Convert all values to float
    corr_matrix = corr_matrix.astype(float)
    return corr_matrix


def plot_heatmap(
    matrix, title="Correlation Heatmap", figsize=(12, 8), save_path=None, show=True
):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)

    # Rotate x-axis (column) labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Rotate y-axis (row) labels if needed (optional)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_datamix_vs_target(df, output_dir):
    """
    Save one plot per datamix column (vs all target columns) as separate images.

    Parameters:
    - df: pandas DataFrame with 'datamix:' and 'target:' columns
    - output_dir: directory where plots will be saved
    """
    datamix_col = df.columns[df.columns.str.startswith("datamix:")]
    target_col = df.columns[df.columns.str.startswith("target:")]

    os.makedirs(output_dir, exist_ok=True)

    for datamix_ref in datamix_col:
        plt.figure(figsize=(8, 6))

        for target_ref in target_col:
            plt.plot(df[datamix_ref], df[target_ref], ".", label=target_ref)

        plt.title(datamix_ref)
        plt.xlabel(datamix_ref)
        plt.ylabel("Target Values")
        # plt.xscale('log')
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            output_dir, f"{datamix_ref.replace(':', '_')}_plot.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expe_dir",
        type=str,
    )
    args = parser.parse_args()

    metric = "word_perplexity"

    # READ RESULTS
    out = []
    for expe_name in os.listdir(args.expe_dir):
        if os.path.isdir(os.path.join(args.expe_dir, expe_name)):
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
    print("Processing...")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(args.expe_dir, "regmix_results.csv"))

    df.loc[:, df.columns.str.startswith("datamix:")] = df.loc[
        :, df.columns.str.startswith("datamix:")
    ].fillna(0)
    df = df.dropna(subset=df.columns[df.columns.str.startswith("target:")])
    df = df[sorted(df.columns)]

    df.to_csv(os.path.join(args.expe_dir, "regmix_results.csv"))
    datamix_col = df.columns[df.columns.str.startswith("datamix:")]
    target_col = df.columns[df.columns.str.startswith("target:")]

    # DATAMIX
    plot_datamix_vs_target(df, args.expe_dir)

    print(f"Plots saved to {args.expe_dir}")

    # CORRELATION
    for method in ["spearman", "pearson", "kendall", "mutual_info"]:
        corr_matrix = compute_correlation_matrix(
            df, datamix_col, target_col, method=method
        )
        # print(f"\nMethod: {method}")
        # print(corr_matrix)
        plot_heatmap(
            corr_matrix,
            title=f"{method.capitalize()} Correlation Heatmap",
            show=False,
            save_path=os.path.join(args.expe_dir, f"{method}.png"),
        )
    print(f"Correlation plots saved to {args.expe_dir}")