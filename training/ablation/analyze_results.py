import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler


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
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{datamix_ref.replace(':', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.path, "out", "regmix_results.csv"))

    # plot datamix vs target
    plot_datamix_vs_target(
        df, output_dir=os.path.join(args.path, "out", "datamix_vs_target_plots")
    )

    # plot correlation heatmaps
    datamix_col = df.columns[df.columns.str.startswith("datamix:")]
    target_col = df.columns[df.columns.str.startswith("target:")]

    for method in ["spearman", "pearson", "kendall", "mutual_info"]:
        corr_matrix = compute_correlation_matrix(
            df, datamix_col, target_col, method=method
        )
        plot_heatmap(
            corr_matrix,
            title=f"{method.capitalize()} Correlation Heatmap",
            show=False,
            save_path=os.path.join(args.path, "out", f"correlation/{method}.png"),
        )
