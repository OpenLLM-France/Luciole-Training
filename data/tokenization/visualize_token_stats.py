import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
import math
from matplotlib.gridspec import GridSpec


def map_language(x):
    european_arab = ["ar", "nl", "de", "pt", "es", "it"]
    if x in ["code", "math", "fr", "en", "aligned", "multi", "cot"]:
        return x
    elif x in european_arab:
        return "euro/arab"
    elif x in ["ca", "eu", "regional"]:
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


def patch_language_column(name, language):
    if ("nemotron-post" in name) and ("think" in name or "stem" in name):
        return "cot"
    if "numina" in name:
        return "math"
    if "open-thoughts" in name:
        return "cot"
    if "stack-math-qa" in name:
        return "math"
    if "open-r1" in name:
        return "cot"
    if "open-code-reasoning" in name:
        return "cot"
    return language


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


def plot_horizontal_bar(df, column_name, output_file, num_columns=2, title=None):
    df.sort_values(column_name, ascending=True, inplace=True)
    total = len(df)
    min_tokens = min(df["total_tokens"])

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

    unique_categories = [
        "fr",
        "en",
        "euro/arab",
        "regional",
        "aligned",
        "multi",
        "code",
        "math",
        "cot",
    ]
    palette = sb.color_palette("colorblind", len(unique_categories))

    color_map = dict(zip(unique_categories, palette))
    colors = df["language"].map(color_map)

    handles = [Patch(color=color_map[cat], label=str(cat)) for cat in unique_categories]
    legend_ax.legend(handles=handles, title="language".capitalize(), loc="center")

    for i in range(num_columns):
        start = i * rows_per_col
        end = min(start + rows_per_col, total)
        sub_df = df.iloc[start:end]
        ax = axes[i]

        ax.barh(
            y=sub_df[column_name],
            width=sub_df["tokens"],
            color=colors[start:end],
            alpha=0.7,
        )
        ax.set_xscale("log")
        # ax.set_xlim(1e7, 1e13)
        ax.xaxis.set_major_formatter(FuncFormatter(format_tokens_ticks))
        ax.set_ylabel(column_name.capitalize() if i == 0 else "")
        ax.invert_yaxis()
        for j, (tokens, epoch, weight) in enumerate(
            zip(sub_df["tokens"], sub_df["epochs"], sub_df["weight"])
        ):
            ax.text(
                max(min_tokens * 30, tokens),
                j,
                f"{format_tokens(tokens)} - {weight:.1%} ({epoch} epoch) ",
                va="center",
                ha="right",
                fontsize=8,
            )

    if title:
        fig.suptitle(
            title,
            fontsize=14,
        )

    fig.savefig(output_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot token treemaps by language and dataset."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory. It must contains the datamix.",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="chronicles/all_stats_merged.csv",
        help="Path to the all_stats.",
    )
    parser.add_argument("--title", type=str, default=None, help="Title of the plot.")
    args = parser.parse_args()
    stats_file = args.stats_file
    output_dir = args.output_dir

    # Load stats
    df = pd.read_csv(stats_file)
    df["language"] = df.apply(
        lambda x: patch_language_column(x["name"], x["language"]), axis=1
    )
    df["language"] = df["language"].apply(map_language)
    df["web_data"] = df["name"].apply(is_web_dataset)

    # Read datamix file
    datamix_file = os.path.join(output_dir, "datamix.json")
    if os.path.isfile(datamix_file):
        with open(datamix_file, "r") as f:
            datamix = json.load(f)
        total_tokens = datamix["total_tokens"]
        df_datamix = pd.DataFrame(datamix["train"])
        df_datamix["name"] = df_datamix["name"].str.replace(
            "_text_document", "", regex=False
        )
        df_datamix["weight"] /= df_datamix["weight"].sum()
        df_datamix["tokens"] = df_datamix["weight"] * total_tokens
    else:
        raise FileNotFoundError(
            f"No datamix.json file found in {output_dir}. Please make sure it exists and is named datamix.json."
        )

    df = df_datamix.merge(df, on="name", how="left", validate="one_to_one")
    df["epochs"] = (df["tokens"] / df["total_tokens"]).round(2)
    print(df.head(5))

    # NAME
    plot_horizontal_bar(
        df,
        "name",
        os.path.join(output_dir, "figs", "bar_all.png"),
        num_columns=2,
        title=args.title,
    )
