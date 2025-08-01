import os
import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_weight_distributions(
    df,
    weight_prefix="weight",
    num_weights=10,
    dataset_col="name",
    output_path="weight_distributions.png",
    show_plot=False,
):
    """
    Plots and saves bar distributions of weight columns across datasets.

    Parameters:
    - df: pd.DataFrame with one row per dataset and columns like 'weight_0', ..., 'weight_49'
    - weight_prefix: prefix used for weight columns (default: "weight")
    - num_weights: number of weight columns to plot (default: 10)
    - dataset_col: name of the column containing dataset identifiers
    - output_path: file path to save the figure (default: "weight_distributions.png")
    - show_plot: whether to show the plot with plt.show() (default: False)
    """
    # Set dataset column as index if not already
    if dataset_col in df.columns:
        df = df.set_index(dataset_col)

    weight_cols = [f"{weight_prefix}_{i}" for i in range(num_weights)]

    # Add 1 extra subplot for legend
    total_plots = num_weights + 1

    fig, axes = plt.subplots(
        nrows=total_plots,
        ncols=1,
        figsize=(20, 3 * total_plots),
        constrained_layout=True,
    )
    axes = axes.flatten()

    for i, col in enumerate(weight_cols):
        ax = axes[i]
        df[col].plot(kind="bar", ax=ax)
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Weight")
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(range(len(df.index)), rotation=90)
        # ax.tick_params(axis='x', labelrotation=90)

    # Legend subplot
    legend_ax = axes[-1]
    legend_ax.axis("off")

    # Create legend handles as "{index} - {name}"
    handles = [
        plt.Line2D([0], [0], color="C0", label=f"{i} - {name}")
        for i, name in enumerate(df.index)
    ]
    legend_ax.legend(
        handles=handles, loc="center", ncol=4, fontsize="small", title="Datasets"
    )

    # Save figure
    fig.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(main_path, "data/tokenized_data/tokens_training_v2"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--start_with",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(main_path, "ablations/regmix"),
        help="Output directory",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        default=50,
    )
    parser.add_argument("--help", "-h", action="store_true")
    args = parser.parse_args()
    output_dir = os.path.join(
        args.output_dir, args.start_with if args.start_with != "" else "all"
    )

    # Load your data (only if not just --help)
    df = pd.read_csv(os.path.join(args.data_path, "stats/all_stats_merged.csv"))
    df = df[df["name"].str.startswith(args.start_with)]
    df = df[["name", "total_tokens"]]
    df["total_tokens_dist"] = df["total_tokens"] / df["total_tokens"].sum()
    df["name"] = df["name"] + "_text_document"

    for seed in range(args.max_seed):
        np.random.seed(seed)
        n_datasets = len(df)

        lambda_param = np.random.uniform(0.1, 5)
        weight = np.random.dirichlet(lambda_param * df["total_tokens_dist"])
        df[f"weight_{seed}"] = weight

        df_seed = df[["name", f"weight_{seed}"]].rename(
            columns={f"weight_{seed}": "weight"}
        )
        df_seed = df_seed[df_seed["weight"] > 0]

        out = {
            "data_path": args.data_path,
            "train": df_seed.to_dict(orient="records"),
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/regmix_{seed}.json", "w") as f:
            json.dump(out, f, indent=4)

    df.to_csv(f"{output_dir}/regmix_weights.csv")
    plot_weight_distributions(df.sort_values("total_tokens", ascending=False).head(20))
