import matplotlib.pyplot as plt
import seaborn as sb
import os
import pandas as pd
import argparse
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import math
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import json


def format_tokens(tokens):
    if tokens >= 1e12:
        return f"{tokens / 1e12:.2f} T"
    elif tokens >= 1e9:
        return f"{tokens / 1e9:.1f} B"
    elif tokens >= 1e6:
        return f"{tokens / 1e6:.1f} M"
    elif tokens >= 1e3:
        return f"{tokens / 1e3:.1f} K"
    elif tokens > 0:
        return f"{tokens:.1f}"
    else:
        return ""


def format_tokens_ticks(x, pos):
    return format_tokens(x)


def plot_horizontal_bar(
    df, column_name, output_file, num_columns=2, color_column="total_tokens"
):
    df = df.reset_index(drop=True)
    min_tokens = min(df[df["total_tokens"] > 0]["total_tokens"])
    total_tokens_all = df["total_tokens"].sum()

    total = len(df)
    rows_per_col = math.ceil(total / num_columns)
    fig_height = max(6, rows_per_col * 0.4)
    fig_width = 10 * num_columns + 2  # Extra space for colorbar/legend

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    spec = GridSpec(
        nrows=1,
        ncols=num_columns + 1,
        figure=fig,
        width_ratios=[1] * num_columns + [0.05],
    )

    axes = []
    for i in range(num_columns):
        if i == 0:
            ax = fig.add_subplot(spec[0, i])
        else:
            ax = fig.add_subplot(spec[0, i], sharex=axes[0])
        axes.append(ax)

    legend_ax = fig.add_subplot(spec[0, -1])
    legend_ax.axis("off")

    # Color setup
    if color_column == "total_tokens":
        log_tokens = np.log10(df["total_tokens"].clip(lower=1))
        norm = mcolors.Normalize(vmin=log_tokens.min(), vmax=log_tokens.max())
        cmap = sb.color_palette("rocket", as_cmap=True)
        colors = cmap(1 - norm(log_tokens))

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=legend_ax, orientation="vertical")
        cbar.set_label("log₁₀(Total Tokens)")

    elif color_column in ["dataset", "group", "subset", "language", "web_data"]:
        unique_categories = df[color_column].astype("category").cat.categories
        palette = sb.color_palette("colorblind", len(unique_categories))

        color_map = dict(zip(unique_categories, palette))
        colors = df[color_column].map(color_map)

        handles = [
            Patch(color=color_map[cat], label=str(cat)) for cat in unique_categories
        ]
        legend_ax.legend(handles=handles, title=color_column.capitalize(), loc="center")

    for i in range(num_columns):
        start = i * rows_per_col
        end = min(start + rows_per_col, total)
        sub_df = df.iloc[start:end]
        ax = axes[i]

        ax.barh(
            y=sub_df[column_name],
            width=sub_df["total_tokens"],
            color=colors[start:end],
            alpha=0.7,
        )
        ax.set_xscale("log")
        # ax.set_xlim(1e7, 1e13)
        ax.xaxis.set_major_formatter(FuncFormatter(format_tokens_ticks))
        ax.set_ylabel(column_name.capitalize() if i == 0 else "")
        ax.invert_yaxis()
        for j, (tokens, label) in enumerate(
            zip(sub_df["total_tokens"], sub_df[column_name])
        ):
            ax.text(
                max(tokens * 0.95, 2 * min_tokens),
                j,
                f"{format_tokens(tokens)} ({tokens/total_tokens_all:.1%})"
                if tokens > 0
                else "",
                va="center",
                ha="right",
                fontsize=8,
            )

        ax.set_title(f"{column_name.capitalize()}s {start + 1}-{end}")

    fig.suptitle(
        f"Tokens per {column_name}\nTotal tokens: {total_tokens_all / 1e9:.1f} B",
        fontsize=14,
    )

    fig.savefig(output_file, dpi=300)
    plt.close()


def plot_box_plot_from_summary(df, column_name, output_file):
    df = df.sort_values("Q2_tokens", ascending=False).reset_index(drop=True)

    num_items = len(df)
    fig_height = max(6, num_items * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for i, row in df.iterrows():
        y_pos = i

        # Whiskers (min to max)
        ax.plot(
            [row["min_tokens"], row["max_tokens"]],
            [y_pos, y_pos],
            color="gray",
            linewidth=1,
            zorder=1,
        )

        # IQR box (Q1 to Q3)
        ax.add_patch(
            plt.Rectangle(
                (row["Q1_tokens"], y_pos - 0.3),
                row["Q3_tokens"] - row["Q1_tokens"],
                0.6,
                edgecolor="black",
                facecolor="skyblue",
                alpha=0.7,
                zorder=2,
            )
        )

        # Median (dot)
        ax.plot(row["Q2_tokens"], y_pos, "o", color="black", zorder=3)

        # Mean (diamond)
        ax.plot(row["mean_tokens"], y_pos, "D", color="red", zorder=3)

    ax.set_yticks(range(num_items))
    ax.set_yticklabels(df[column_name])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(format_tokens_ticks))
    ax.set_xlabel("Tokens (log scale)")
    ax.set_title(f"Token distribution per {column_name}\n(log scale)")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0))
    ax.grid(True, which="major", axis="x", linestyle="--", color="lightgray", alpha=0.7)
    ax.axvline(
        4096,
        color="lightblue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="4096 tokens",
    )
    ax.axvline(
        8192,
        color="lightblue",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="8192 tokens",
    )
    line_4096 = Line2D(
        [0], [0], color="lightblue", linestyle="--", linewidth=1.5, label="4096 tokens"
    )
    line_8192 = Line2D(
        [0], [0], color="lightblue", linestyle="-", linewidth=1.5, label="8192 tokens"
    )

    # Build a comprehensive legend
    legend_elements = [
        Line2D([0], [0], color="gray", lw=1, label="Min–Max"),
        Patch(facecolor="skyblue", edgecolor="black", label="IQR (Q1–Q3)", alpha=0.7),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Median (Q2)",
            linestyle="",
            markersize=6,
        ),
        Line2D(
            [0], [0], marker="D", color="red", label="Mean", linestyle="", markersize=6
        ),
        line_4096,
        line_8192,
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def create_datamix_file(df, token_dir, output_dir):
    df = df.copy()
    df = df[df["total_tokens"] > 0]
    df.loc[:, "name"] = df["name"] + "_text_document"
    df.loc[:, "weight"] = df["total_tokens"] / df["total_tokens"].sum()
    out = {
        "data_path": token_dir,
        "total_tokens": int(df["total_tokens"].sum()),
        "train": df[["name", "weight"]].to_dict(orient="records"),
    }
    with open(f"{output_dir}/datamix.json", "w") as f:
        json.dump(out, f, indent=4)


def map_language(x):
    european = ["ar", "nl", "de", "pt", "es", "it"]
    if x in ["code", "math", "fr", "en", "aligned", "multi"]:
        return x
    elif x in european:
        return "european"
    elif x in ["ca", "regional"]:
        return "regional"
    else:
        raise ValueError(f"Unknown language: {x}")


def is_web_dataset(name):
    if "fineweb" in name:
        return "web"
    if "dclm" in name:
        return "web"
    if "culturax" in name:
        return "web"
    if "hplt2" in name:
        return "web"
    return "no_web"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot token treemaps by language and dataset."
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="chronicles/all_stats_merged.csv",
        help="Path to the all_stats.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chronicles/raw",
        help="Path to the directory containing repeats file.",
    )
    parser.add_argument(
        "--repeats_file",
        type=str,
        default=None,
        help="Path to the repeats file.",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="",
        help="Path to the token directory.",
    )
    args = parser.parse_args()
    stats_file = args.stats_file
    output_dir = args.output_dir
    repeats_file = args.repeats_file
    token_dir = args.token_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figs"), exist_ok=True)

    df = pd.read_csv(stats_file)

    if repeats_file is not None:
        repeats = pd.read_csv(repeats_file)
        df = df.merge(repeats, on="name", how="left")
        df["repeat"] = df["repeat"].fillna(0)
        df["total_tokens"] = df["total_tokens"] * df["repeat"]
        create_datamix_file(df, token_dir, output_dir)

    assert (
        not df["name"].duplicated().any()
    ), f"Duplicate names in all_stats_merged.csv. Duplicates: {df[df['name'].duplicated()]['name'].tolist()}"

    df["group"] = df.apply(
        lambda row: f"{row['dataset']}_{row['subset']}"
        if pd.notnull(row["subset"])
        else row["dataset"],
        axis=1,
    )

    # Apply the mapping
    df["language"] = df["language"].apply(map_language)
    df["web_data"] = df["language"] + "_" + df["name"].apply(is_web_dataset)
    # Groupby
    language_df = (
        df.groupby("language")["total_tokens"]
        .sum()
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )
    web_df = (
        df.groupby("web_data")
        .agg(total_tokens=("total_tokens", "sum"), language=("language", "first"))
        .reset_index()
        .sort_values("web_data", ascending=False)
    )
    dataset_df = (
        df.groupby("dataset")["total_tokens"]
        .sum()
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )
    group_df = (
        df.groupby("group")[["total_tokens"]]
        .sum()
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )
    df = df.sort_values(
        by=["dataset", "language", "total_tokens"], ascending=[True, True, False]
    )

    # Horizontal bar
    plot_horizontal_bar(
        language_df,
        "language",
        os.path.join(output_dir, "figs", "bar_language.png"),
        color_column="language",
    )
    plot_horizontal_bar(
        web_df,
        "web_data",
        os.path.join(output_dir, "figs", "bar_web.png"),
        color_column="language",
    )
    # plot_horizontal_bar(
    #     dataset_df, "dataset", os.path.join(output_dir, "bar_datasets.png")
    # )
    # plot_horizontal_bar(group_df, "group", os.path.join(output_dir, "bar_group.png"))
    plot_horizontal_bar(
        df,
        "name",
        os.path.join(output_dir, "figs", "bar_all.png"),
        color_column="language",
        num_columns=2,
    )

    # Box plot
    plot_box_plot_from_summary(
        df, "name", os.path.join(output_dir, "figs", "boxplot_all.png")
    )
