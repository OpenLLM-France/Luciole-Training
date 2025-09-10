import os
import math
import argparse
from utils import process_results, read_experiment_results
from agg_score import calculate_agg_score, read_info
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import pandas as pd

task_group_mapping = {
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
        ("helm|siqa|0|0", "pem"),
        ("helm|commonsenseqa|0|0", "pem"),
    ],
    "fr": [
        ("lighteval|meta_mmlu_fra_cf:_average|0", "acc_norm_pmi"),
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_pmi"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm_token"),
        ("lighteval|xcodah_fra_cf|0", "acc_norm_token"),
        ("lighteval|xcsqa_fra_cf|0", "acc_norm_pmi"),
        ("lighteval|xnli2.0_fra_cf|0", "acc_"),
        ("lighteval|fquadv2_fra|0", "f1_fra"),
        ("lighteval|mintaka_fra|0", "f1_fra"),
        ("fr", "agg"),
    ],
    "agg": [
        ("AGG_EN", "agg"),
        ("AGG_EN_GK", "agg"),
        ("AGG_EN_NLU", "agg"),
        ("AGG_EN_RES", "agg"),
        ("AGG_FR", "agg"),
        ("AGG_FR_GEN", "agg"),
        ("AGG_FR_GK", "agg"),
        ("AGG_FR_NLU", "agg"),
        ("AGG_FR_RC", "agg"),
        ("AGG_FR_RES", "agg"),
    ],
}


def assign_colors(df):
    unique_experiments = df["expe_name"].unique()
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return {name: next(colors) for name in unique_experiments}


df_info = read_info()


def plot_task(ax, df, task, metric, color_map, xlog=False, fit=False):
    df = df[(df["task"] == task) & (df["metric"] == metric)]

    # Access random
    num_classes = df_info.loc[df_info["task"] == task, "num_classes"].iloc[0]
    random = 1.0 / num_classes
    ax.axhline(y=random, color="grey", linestyle="--", label="random")

    for _, row in df.iterrows():
        color = color_map[row["expe_name"]]

        if fit:
            ax.plot(
                row["tokens"],
                row["score"],
                alpha=np.clip(1 - row["r2"], 0.2, 0.8),
                linestyle=":",
                color=color,
            )

            # Plot regression line
            xaxis = np.linspace(1, 35, 100)
            y_pred = row["intercept"] + row["slope"] * np.log(xaxis)
            ax.plot(
                xaxis,
                y_pred,
                linestyle="-",
                alpha=np.clip(row["r2"], 0.2, 0.8),
                color=color,
                label=row["expe_name"],
            )

            ax.text(
                xaxis[-1],
                y_pred[-1],
                f"$R^2$={row['r2']:.2f}",
                color=color,
                fontsize=8,
                ha="left",
                va="center",
            )
        else:
            ax.plot(
                row["tokens"],
                row["score"],
                marker="+",
                alpha=0.8,
                color=color,
                label=row["expe_name"],
            )

    ax.set_xlabel("B tokens")
    ax.set_ylabel(metric)
    ax.set_title(task)

    if xlog:
        ax.set_xscale("log")


def plot_list_of_tasks(
    df,
    list_of_tasks_to_plot,
    output_file=None,
    title=None,
    xlog=False,
    fit=False,
    max_subplot=15,
):
    list_of_tasks_to_plot = [
        task for task in list_of_tasks_to_plot if task[0] in set(df["task"].unique())
    ]
    n_tasks = len(list_of_tasks_to_plot)
    if n_tasks > max_subplot:
        print("Splitting results in different figures...")
        for i, chunk_list in enumerate(
            [
                list_of_tasks_to_plot[i : i + max_subplot]
                for i in range(0, n_tasks, max_subplot)
            ]
        ):
            if output_file:
                base, ext = os.path.splitext(output_file)
                chunk_output_file = f"{base}_part{i}{ext}"
            else:
                chunk_output_file = None
            plot_list_of_tasks(df, chunk_list, chunk_output_file, title, xlog, fit)
        return

    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    color_map = assign_colors(df)  # Global color map

    # Keep track of labels added to the legend
    legend_dict = {}

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(axes[i], df, task, metric, color_map=color_map, xlog=xlog, fit=fit)

        handles, labels = axes[i].get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_dict[label] = handle

    # Dedicated subplot for legend
    legend_ax = axes[-1]
    legend_ax.axis("off")
    # Set legend handle alpha to 1.0
    for handle in legend_dict.values():
        handle.set_alpha(1.0)

    legend_ax.legend(
        legend_dict.values(), legend_dict.keys(), title="Experiment name", loc="center"
    )

    # Hide any unused subplots
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
        "experiment_path",
        type=str,
        nargs="+",
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        choices=["all"] + list(task_group_mapping.keys()),
        default=["all"],
        help="List of predefined groups of tasks you want to plot. You can add groups in the mapping if you want.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="out/",
        help="Output path where your plot are storred",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max samples",
    )
    parser.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--fit", action="store_true", help="Fit a linear regression")
    args = parser.parse_args()
    max_samples = str(args.max_samples) if args.max_samples > 0 else "None"

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    df = pd.concat(
        [
            read_experiment_results(path, seq_length=args.seq_length)
            for path in args.experiment_path
        ]
    )
    df_agg = calculate_agg_score(df).dropna()
    df = pd.concat([df, df_agg])
    df = process_results(df)
    df = df[df["max_samples"] == max_samples]

    if df.empty:
        print("No results found for the given experiments.")
        exit(0)

    for g in args.group:
        if g == "all":
            list_of_tasks_to_plot = list(
                df[["task", "metric"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            list_of_tasks_to_plot = [
                task
                for task in list_of_tasks_to_plot
                if (task[0] != "all")
                and not ("mmlu" in task[0] and "average" not in task[0])
            ]
        else:
            list_of_tasks_to_plot = task_group_mapping[g]

        filename = f'{g}{"_xlog" if args.xlog else ""}{"_fit" if args.fit else ""}.png'

        output_file = (
            os.path.join(args.output_path, filename) if args.output_path else None
        )
        plot_list_of_tasks(
            df, list_of_tasks_to_plot, output_file, xlog=args.xlog, fit=args.fit
        )

    if not args.output_path:
        plt.show()
