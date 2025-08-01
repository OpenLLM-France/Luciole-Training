import os
import argparse
from utils import read_experiment_results, read_datamix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expe_dir",
        type=str,
        default=os.path.join(MAIN_PATH, "ablations/train/regmix/common-pile"),
    )
    args = parser.parse_args()

    tasks = [
        "helm|the_pile:commoncrawl|0",
        "helm|the_pile:stackexchange|0",
        "helm|the_pile:wikipedia|0",
    ]
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
            df_results = df_results[df_results["task"].isin(tasks)]
            df_results = df_results[df_results["steps"] == 951]

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

    df.to_csv(os.path.join(args.expe_dir, "regmix_results.csv"))

    # CORRELATION
    datamix_col = df.columns[df.columns.str.startswith("datamix:")]
    target_col = df.columns[df.columns.str.startswith("target:")]

    for method in ["spearman", "pearson", "kendall", "mutual_info"]:
        corr_matrix = compute_correlation_matrix(
            df, datamix_col, target_col, method=method
        )
        print(f"\nMethod: {method}")
        print(corr_matrix)
        plot_heatmap(
            corr_matrix,
            title=f"{method.capitalize()} Correlation Heatmap",
            show=False,
            save_path=os.path.join(args.expe_dir, f"{method}.png"),
        )
