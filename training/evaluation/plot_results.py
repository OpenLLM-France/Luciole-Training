import os
import math
import argparse
from utils import process_results, read_experiment_results
from agg_score import calculate_agg_score, read_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

task_group_mapping = {}


def assign_colors(df):
    unique_experiments = df["expe_name"].unique()
    if len(unique_experiments) <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(cmap.N)]
    color_map = {}
    i = -1
    previous_name = ""
    for name in unique_experiments:
        if name.split("_phase")[0] != previous_name.split("_phase")[0]:
            i += 1
        previous_name = name
        color_map[name] = colors[i % len(colors)]
    return color_map

def assign_styles(df):
    unique_experiments = df["expe_name"].unique()
    style_map = {}
    for name in unique_experiments:
        if "phase2" in name:
            style_map[name] = ":"
        else:
            style_map[name] = "-"
    return style_map


df_info = read_info()


def plot_task(
    ax, df, task, metric, color_map, style_map, xlog=False, fit=False, flops=False, max_tokens=None
):
    xaxis_column = "FLOPs" if flops else "tokens"
    df = df[(df["task"] == task) & (df["metric"] == metric)]
    # print(f"Plotting {task} - {metric} with {len(df)} lines")

    # Access random
    if task in df_info["task"].values:
        num_classes = df_info.loc[df_info["task"] == task, "num_classes"].iloc[0]
        random = 1.0 / num_classes
        ax.axhline(y=random, color="grey", linestyle="--", label="random")

    for _, row in df.iterrows():
        color = color_map[row["expe_name"]]
        linestyle = style_map[row["expe_name"]]

        if fit:
            ax.plot(
                row[xaxis_column],
                row["score"],
                alpha=np.clip(1 - row["r2"], 0.2, 0.8),
                linestyle=":",
                color=color,
            )

            # Plot regression line
            xaxis = np.linspace(min(row[xaxis_column]), max(row[xaxis_column]), 100)
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
            if len(row["score"]) == 1:
                ax.plot(
                    row[xaxis_column],
                    row["score"],
                    marker="+",
                    color=color,
                    markersize=10,  # larger than usual
                    markeredgewidth=2,
                    label=row["expe_name"],
                )
            else:
                ax.plot(
                    row[xaxis_column],
                    row["score"],
                    alpha=1,
                    color=color,
                    linestyle=linestyle,
                    label=row["expe_name"],
                )

    ax.set_xlabel("FLOPs" if flops else "B tokens")
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
    flops=False,
    max_subplot=15,
    max_tokens=None,
):
    list_of_tasks_to_plot = [
        task for task in list_of_tasks_to_plot if task[0] in set(df["task"].unique())
    ]
    n_tasks = len(list_of_tasks_to_plot)
    if not isinstance(max_subplot, int) or n_tasks > max_subplot:
        print("Splitting results in different figures...")
        for i, chunk_list in enumerate(
            [
                list_of_tasks_to_plot[i : i + max_subplot]
                for i in range(0, n_tasks, max_subplot)
            ] if isinstance(max_subplot, int) else
            [
                list_of_tasks_to_plot[sum(max_subplot[:i]):sum(max_subplot[:i+1])]
                for i in range(len(max_subplot))
            ]
        ):
            if output_file:
                base, ext = os.path.splitext(output_file)
                chunk_output_file = f"{base}_part{i}{ext}"
            else:
                chunk_output_file = None
            plot_list_of_tasks(
                df, chunk_list, chunk_output_file, title, xlog, fit, flops
            )
        return

    num_tasks = len(list_of_tasks_to_plot)
    num_plots = num_tasks + 1  # +1 for the legend

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = [axes]  # wrap single Axes in a list
    else:
        axes = axes.flatten()
    color_map = assign_colors(df)  # Global color map
    style_map = assign_styles(df)

    # Keep track of labels added to the legend
    legend_dict = {}

    for i, (task, metric) in enumerate(list_of_tasks_to_plot):
        plot_task(
            axes[i],
            df,
            task,
            metric,
            color_map=color_map,
            style_map=style_map,
            xlog=xlog,
            fit=fit,
            flops=flops,
            max_tokens=max_tokens,
        )

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
        print(f"Saved figure to {output_file}")


def plot_experiments(df, args, max_subplot=15):
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    for g in args.group:
        print(f"Processing group: {g}...")
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

        filename = f'{g}{"_max" + args.max_samples if args.max_samples != "None" else ""}{"_xlog" if args.xlog else ""}{"_fit" if args.fit else ""}{"_flops" if args.flops else ""}.png'

        output_file = (
            os.path.join(args.output_path, filename) if args.output_path else None
        )
        plot_list_of_tasks(
            df,
            list_of_tasks_to_plot,
            output_file,
            xlog=args.xlog,
            fit=args.fit,
            flops=args.flops,
            max_tokens=args.max_tokens,
            max_subplot=max_subplot,
        )

    if not args.output_path:
        plt.show()


def process_experiments(args):
    # Read and aggregate all results
    all_results = []

    for path in args.experiment_path:
        # Step 1: read experiment results
        df = read_experiment_results(path, evaluation_dir=args.evaluation_dir)

        if df is None or df.empty:
            print(f"No results found in {path}, skipping...")
            continue

        # Step 2: calculate aggregated scores if needed
        if "agg" in args.group:
            df_agg = calculate_agg_score(df).dropna()
            df = pd.concat([df, df_agg])

        # Step 3: process the results
        df = process_results(df, fit=args.fit, window=args.window)

        # Step 4: collect results
        all_results.append(df)

    # Combine all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    if final_df.empty:
        print("No results found for the given experiments.")
        exit(0)
    return final_df


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
    parser.add_argument("--max_samples", type=str, default="None")
    parser.add_argument(
        "--selection", default=False, action="store_true", help="Use only a selection of tasks"
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation",
    )
    parser.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--fit", action="store_true", help="Fit a linear regression")
    parser.add_argument(
        "--flops", action="store_true", help="Use FLOPs instead of tokens"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Use a sliding window to smooth the curves. 1 means no smoothing.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None, help="Max tokens to plot (in B)"
    )

    args = parser.parse_args()

    df = process_experiments(args)
    if args.selection:

        task_and_metric_selection = [
            ("helm|boolq:_average|0", "em_with_type_exact_match"),
            ("helm|commonsenseqa|0", "em_with_normalize_gold&normalize_pred"),
            ("helm|siqa|0", "em"),
            ("leaderboard|arc:challenge|0", "acc_with_logprob_normalization"),
            ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
            ("leaderboard|hellaswag|0", "acc"),
            ("leaderboard|winogrande|0", "acc"),
            ("lighteval|arc:easy|0", "acc_with_logprob_normalization"),
            ("lighteval|openbookqa|0", "acc_with_logprob_normalization"),
            ("lighteval|piqa|0", "acc_with_logprob_normalization"),
            ("lighteval|triviaqa|0", "em_with_strip_strings&normalize_pred"),

            ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"),
            ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"),
            ("lighteval|xcodah_fra_cf|0", "acc_norm"),
            ("lighteval|xcsqa_fra_cf|0", "acc_norm_token"),
            ("lighteval|xnli2.0_fra_cf|0", "acc_norm_token"),
            ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm"),
            ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm"),
            ("lighteval|global_mmlu_all_fra_cf:_average|0", "acc_norm"),
            ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"),

            ("lighteval|mlmm_hellaswag_deu_cf|0", "acc_norm"),
            ("lighteval|mlmm_hellaswag_spa_cf|0", "acc_norm"),
            ("lighteval|mlmm_hellaswag_ita_cf|0", "acc_norm"),
            ("lighteval|mlmm_hellaswag_ara_cf|0", "acc_norm"),

            ("lighteval|global_mmlu_all_deu_cf:_average|0", "acc_norm"),
            ("lighteval|global_mmlu_all_spa_cf:_average|0", "acc_norm"),
            ("lighteval|global_mmlu_all_ita_cf:_average|0", "acc_norm"),
            ("lighteval|global_mmlu_all_ara_cf:_average|0", "acc_norm"),

            ("lighteval|belebele_deu_Latn_cf|0", "acc_norm"),
            ("lighteval|belebele_spa_Latn_cf|0", "acc_norm"),
            ("lighteval|belebele_ita_Latn_cf|0", "acc_norm"),
            ("lighteval|belebele_arb_Arab_cf|0", "acc_norm"),

            ("lighteval|belebele_por_Latn_cf|0", "acc_norm"),
            ("lighteval|global_mmlu_all_por_cf:_average|0", "acc_norm"),
            ("lighteval|belebele_nld_Latn_cf|0", "acc_norm"),
            # ("lighteval|global_mmlu_all_nld_cf:_average|0", "acc_norm"),
        ]

        max_subplot = (11, 9, 15)

        df_new = df[(df["task"] == task_and_metric_selection[0][0]) & (df["metric"] == task_and_metric_selection[0][1])]
        for task, metric in task_and_metric_selection[1:]:
            df_new = pd.concat([df_new, df[(df["task"] == task) & (df["metric"] == metric)]])
        df = df_new
    else:
        df = df[df["max_samples"] == args.max_samples]

        max_subplot = 15

    print(df)
    plot_experiments(df, args, max_subplot=max_subplot)
