import matplotlib.pyplot as plt
import seaborn as sb
import os
import pandas as pd
import re
import squarify
import argparse
import sys


def format_tokens(tokens):
    if tokens >= 1e9:
        return f"{tokens / 1e9:.1f} B"
    elif tokens >= 1e6:
        return f"{tokens / 1e6:.0f} M"
    elif tokens > 0:
        return f"{tokens:.0f}"
    else:
        return ""


def plot_treemap(df, column_name, output_file):
    labels = [
        f"{name}\n{format_tokens(tokens)}"
        for name, tokens in zip(df[column_name], df["total_tokens"])
    ]

    plt.figure(figsize=(10, 6))
    squarify.plot(
        sizes=df["total_tokens"],
        label=labels,
        pad=0.05,
        alpha=0.7,
        text_kwargs={"fontsize": 6, "color": "black"},
        color=sb.color_palette("rocket", len(df)),
    )

    plt.axis("off")
    plt.title(
        f"Tokens per {column_name}\n Total tokens: {df['total_tokens'].sum() / 1e9:.1f} B"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_horizontal_bar(df, column_name, output_file):
    df = df.sort_values("total_tokens", ascending=False)

    plt.figure(figsize=(10, 6))
    colors = sb.color_palette("rocket", len(df))

    plt.barh(y=df[column_name], width=df["total_tokens"] / 1e9, color=colors, alpha=0.7)

    plt.xlabel("Total tokens (Billions)")
    plt.ylabel(column_name.capitalize())
    plt.title(
        f"Tokens per {column_name}\nTotal tokens: {df['total_tokens'].sum() / 1e9:.1f} B"
    )
    plt.gca().invert_yaxis()

    for i, (tokens, label) in enumerate(zip(df["total_tokens"], df[column_name])):
        text = format_tokens(tokens)
        if text:
            plt.text(tokens / 1e9 + 1, i, text, va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def extract_info(text):
    pattern = r"((.*?)(?:_(?:.*))*)_(.*)"
    match = re.match(pattern, text)
    if match:
        rest = match.group(1)
        first = match.group(2)
        lang = match.group(3)
        return {"language": lang, "group": first, "dataset": rest}
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot token treemaps by language and dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the token directory (it must contains stats/ ).",
    )
    args = parser.parse_args()

    stats_dir = os.path.join(args.path, "stats")
    if not os.path.isdir(stats_dir):
        print(
            f"Error: The directory '{args.path}' does not contain a 'stats/' subdirectory."
        )
        sys.exit(1)  # exit with error code

    input_file = os.path.join(stats_dir, "all_stats_merged.csv")
    if not os.path.isfile(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    input_file = os.path.join(args.path, "stats/all_stats_merged.csv")
    output_path = os.path.join(args.path, "figs")
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(input_file)
    df = pd.concat([df, df["name"].apply(extract_info).apply(pd.Series)], axis=1)

    language_df = (
        df.groupby("language")["total_tokens"]
        .sum()
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )
    dataset_df = (
        df.groupby("dataset")["total_tokens"]
        .sum()
        .reset_index()
        .sort_values("total_tokens", ascending=False)
    )
    df = df.sort_values(by=["dataset", "total_tokens"], ascending=False)

    # Treemap
    plot_treemap(
        language_df, "language", os.path.join(output_path, "treemap_language.png")
    )
    plot_treemap(
        dataset_df, "dataset", os.path.join(output_path, "treemap_datasets.png")
    )
    plot_treemap(df, "name", os.path.join(output_path, "treemap_all.png"))
    # Horizontal bar
    plot_horizontal_bar(
        language_df, "language", os.path.join(output_path, "bar_language.png")
    )
    plot_horizontal_bar(
        dataset_df, "dataset", os.path.join(output_path, "bar_datasets.png")
    )
    plot_horizontal_bar(df, "name", os.path.join(output_path, "bar_all.png"))
