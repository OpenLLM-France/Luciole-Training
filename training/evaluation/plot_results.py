import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import argparse

task_group_mapping = {
    "mmlu": [
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm"),
    ],
    "en": [
        ("helm|boolq|0", "pem"),
        ("lighteval|triviaqa|0", "qem"),
        ("lighteval|arc:easy|0", "acc"),
        ("lighteval|arc:easy|0", "acc_norm"),
        ("leaderboard|arc:challenge|0", "acc"),
        ("leaderboard|arc:challenge|0", "acc_norm"),
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|openbookqa|0", "acc_norm"),
        ("lighteval|piqa|0", "acc_norm"),
    ],
    "fr": [
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm"),
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm"),
        ("lighteval|xcodah_fra_cf|0", "acc_norm"),
        ("lighteval|xcsqa_fra_cf|0", "acc_norm"),
        ("lighteval|xnli2.0_fra_cf|0", "acc_norm"),
        ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"),
        ("lighteval|fquadv2_fra|0", "f1_fra"),
        ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"),
        ("lighteval|mintaka_fra|0", "f1_fra"),
    ],
}


def read_json_results(file):
    with open(file, "r") as file:
        data = json.load(file)
    model_name = data["config_general"]["model_name"]
    results = data["results"]
    df = pd.DataFrame.from_dict(results, orient="index").reset_index(names="task")
    df["model_name"] = model_name
    return df


def read_experiment_results(main_dir):
    main_dir = Path(main_dir)
    experiment_name = main_dir.name

    json_files = main_dir.rglob("results_*.json")  # recursively finds all .json files

    dfs = []
    for file in json_files:
        df_part = read_json_results(file)
        match = re.match(r"results_(.*)\.json", file.name)
        df_part["timestamp"] = match.group(1) if match else None
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)

    # Fix date and deduplicate
    df["timestamp"] = (
        df["timestamp"]
        .str.replace("T", " ")
        .str.replace(r"(\d{2})-(\d{2})-(\d{2}\.\d+)", r"\1:\2:\3", regex=True)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Post process some columns
    df["experiment_name"] = experiment_name
    df["step"] = df["model_name"].str.extract(r"--step_([0-9.]+)-")[0].astype(float)
    df["samples"] = (
        df["model_name"].str.extract(r"-consumed_samples_([0-9.]+)")[0].astype(float)
    )
    df["tokens"] = df["samples"] * 2048 / 10**9
    return df


def plot_task(ax, df, task, metric, xlog=False):
    df = df[df["task"] == task]
    df = df.sort_values("tokens")

    pivot_df = df.pivot(index="tokens", columns="experiment_name", values=metric)
    stderr_df = df.pivot(
        index="tokens", columns="experiment_name", values=metric + "_stderr"
    )

    for col in pivot_df.columns:
        mean = pivot_df[col].dropna()
        stderr = stderr_df[col].dropna()

        # Align indices in case they differ slightly after dropna
        common_index = mean.index.intersection(stderr.index)
        mean = mean.loc[common_index]
        stderr = stderr.loc[common_index]

        ax.plot(mean.index, mean.values, marker="+", label=col, alpha=0.8)
        ax.fill_between(mean.index, mean - stderr, mean + stderr, alpha=0.1)

    ax.set_xlabel("B tokens")
    ax.set_ylabel(metric)
    ax.set_title(task)
    if xlog:
        ax.set_xscale("log")


def plot_list_of_tasks(
    df, list_of_tasks_to_plot, output_file=None, title=None, xlog=False
):
    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    # Store handles and labels for the shared legend
    legend_handles, legend_labels = None, None

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(axes[i], df, task, metric, xlog=xlog)

        # Grab the legend handles and labels from the first plot (or any plot)
        if legend_handles is None:
            handles, labels = axes[i].get_legend_handles_labels()
            legend_handles, legend_labels = handles, labels

    # Dedicated subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        title="Experiment name",
        loc="center",
        # prop={'size': 14},
        # title_fontsize=16
    )

    # Hide any other unused subplots if any
    for j in range(len(list_of_tasks_to_plot), len(axes) - 1):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        nargs="+",
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        choices=list(task_group_mapping.keys()),
        default=["en"],
        help="List of predefined groups of tasks you want to plot. You can add groups in the mapping if you want.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )
    parser.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")
    args = parser.parse_args()
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    dfs = []
    for path in args.experiment_path:
        dfs.append(read_experiment_results(path))
    df = pd.concat(dfs)

    columns = ["task", "experiment_name", "step", "samples", "tokens"]
    df_sorted = df.sort_values("timestamp")
    # # Identify duplicates that would be removed
    # duplicates_removed = df_sorted[df_sorted.duplicated(subset=columns, keep='last')]
    # # Print removed rows
    # print(duplicates_removed)
    # # Drop duplicates
    df = df_sorted.drop_duplicates(subset=columns, keep="last")

    for g in args.group:
        output_file = (
            os.path.join(args.output_path, f"{g}.png") if args.output_path else None
        )
        plot_list_of_tasks(
            df,
            task_group_mapping[g],
            output_file=output_file,
            title=None,
            xlog=args.xlog,
        )

    if not args.output_path:
        plt.show()
