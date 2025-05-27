import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
from utils import task_group_mapping, get_task_info, read_experiment_results

def plot_task(ax, df, task, metric, xlog=False, no_std=False):
    if task not in df["task"].unique():
        print(f"Task '{task}' not found in the DataFrame.")
        return None
    df = df[df["task"] == task]

    pivot_df = df.pivot_table(index="tokens", columns="experiment_name", values=metric, sort=False)
    stderr_df = df.pivot_table(
        index="tokens", columns="experiment_name", values=metric + "_stderr", sort=False
    )

    for col in pivot_df.columns:
        mean = pivot_df[col].dropna()
        stderr = stderr_df[col].dropna()

        # Align indices in case they differ slightly after dropna
        common_index = mean.index.intersection(stderr.index)
        mean = mean.loc[common_index]
        stderr = stderr.loc[common_index]

        ax.plot(mean.index, mean.values, marker="+", label=col, alpha=0.8)
        if not no_std:
            ax.fill_between(mean.index, mean - stderr, mean + stderr, alpha=0.1)

    task_infos = get_task_info(task)
    if task_infos is not None:
        xmin, xmax = ax.get_xlim()
        ax.hlines(task_infos['random'], xmin, xmax, colors='gray', linestyles='dashed', label="random")

    ax.set_xlabel("B tokens")
    ax.set_ylabel(metric)
    ax.set_title(task)
    if xlog:
        ax.set_xscale("log")

def plot_list_of_tasks(
    df, list_of_tasks_to_plot, output_file=None, title=None, xlog=False, no_std=False
):
    list_of_tasks_to_plot = [task for task in list_of_tasks_to_plot if task[0] in set(df["task"].unique())]
    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    # Use a dictionary to avoid duplicate labels
    legend_dict = {}

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(axes[i], df, task, metric, xlog=xlog, no_std=no_std)

        handles, labels = axes[i].get_legend_handles_labels()
        for h, l in zip(handles, labels):
            legend_dict[l] = h  # If duplicate, keeps last

    # Dedicated subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        legend_dict.values(),
        legend_dict.keys(),
        title="Experiment name",
        loc="center",
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
    parser.add_argument("--no_std", action="store_true", help="Remove std")
    args = parser.parse_args()
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    dfs = []
    for path in args.experiment_path:
        dfs.append(
            read_experiment_results(path)
        )
    df = pd.concat(dfs)

    columns = ["task", "experiment_name", "step", "samples", "tokens"]

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
            no_std=args.no_std
        )

    if not args.output_path:
        plt.show()
