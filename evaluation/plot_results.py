import os
import math
import argparse
from utils import process_results, read_experiment_results
from agg_score import calculate_agg_score, get_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

task_group_mapping = {
    "en": [
        ("lighteval|arc:easy|0", "acc_with_logprob_normalization"),
        ("leaderboard|arc:challenge|0", "acc_with_logprob_normalization"),
        ("lighteval|openbookqa|0", "acc_with_logprob_normalization"),
        ("lighteval|triviaqa|0", "em_with_strip_strings&normalize_pred"),
        ("custom|mmlu_pro_cf|0", "acc_norm_token"),
        ("helm|commonsenseqa|0", "em_with_normalize_gold&normalize_pred"),
        ("helm|siqa|0", "em"),
        ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|piqa|0", "acc_with_logprob_normalization"),
        # ("helm|boolq:_average|0", "em_with_type_exact_match"),
    ],
    "smollm": [
        ("custom|piqa_cf|0", "acc_norm"),
        ("lighteval|piqa|0", "acc_with_logprob_normalization"),
        ("custom|hellaswag_cf|0", "acc_norm"),
        ("leaderboard|hellaswag|0", "acc"),
        ("custom|openbookqa_cf|0", "acc_norm"),
        ("lighteval|openbookqa|0", "acc_with_logprob_normalization"),
        ("custom|commonsenseqa_cf|0", "acc_norm"),
        ("helm|commonsenseqa|0", "em_with_normalize_gold&normalize_pred"),
        ("custom|boolq_cf|0", "acc_norm"),
        ("helm|boolq:_average|0", "em_with_type_exact_match"),
        ("custom|arc_cf:challenge|0", "acc_norm"),
        ("leaderboard|arc:challenge|0", "acc_with_logprob_normalization"),
        ("custom|arc_cf:easy|0", "acc_norm"),
        ("lighteval|arc:easy|0", "acc_with_logprob_normalization"),
        ("custom|winogrande_cf|0", "acc_norm"),
        ("leaderboard|winogrande|0", "acc"),
        ("custom|gsm8k|5", "extractive_match"),
        ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
    ],
    "cultural": [
        ("lighteval|global_mmlu_cs_eng_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_ca_eng_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_cs_fra_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_ca_fra_cf:_average|0", "acc_norm"),
    ],
    "idiomatic_expressions": [
        ("custom:idiomatic_expressions_fib_context:_average:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:different:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:similar:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:word_by_word:0", "acc"),
    ],
    "mmlu": [
        ("custom|mmlu_pro_cf|0", "acc_norm"),
        ("custom|mmlu_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_eng_cf:_average|0", "acc_norm"),
    ],
    "en_new": [
        ("lighteval|belebele_eng_Latn_cf|0", "acc_norm"),
        ("lighteval|global_mmlu_all_eng_cf:_average|0", "acc_norm"),
    ],
    "fr": [
        ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"),  # f1_fra ?
        ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"),  # f1_fra ?
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
        ("lighteval|global_mmlu_all_fra_cf:_average|0", "acc_norm"),
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm_token"),
        ("lighteval|xcodah_fra_cf|0", "acc_norm"),
        ("lighteval|xcsqa_fra_cf|0", "acc_norm_token"),
        ("lighteval|xnli2.0_fra_cf|0", "acc_norm_token"),
    ],
    "multilingual": [
        ("lighteval|global_mmlu_all_deu_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_spa_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_ita_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_ara_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_por_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_nld_cf:_average|0", "acc_norm"),
        # ("lighteval|belebele_deu_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_spa_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_ita_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_arb_Arab_cf|0", "acc_norm"),
        # ("lighteval|belebele_por_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_nld_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_deu_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_spa_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_ita_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_ara_cf|0", "acc_norm_token"),
    ],
    "translation": [
        # ("lighteval|flores200:fra_Latn-eng_Latn|5", "bleu"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "bleu_4"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "comet"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "metricx"),
        # ("lighteval|flores200:eng_Latn-fra_Latn|5", "bleu"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "bleu_4"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "comet"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "metricx"),
    ],
    "ruler": [
        ("custom|ruler_4096:_average|0", "ruler_match"),
        ("custom|ruler_8192:_average|0", "ruler_match"),
        ("custom|ruler_16384:_average|0", "ruler_match"),
        ("custom|ruler_32768:_average|0", "ruler_match"),
        ("custom|ruler_65536:_average|0", "ruler_match"),
        ("custom|ruler_131072:_average|0", "ruler_match"),
    ],
    "finetune": [
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm"),
        ("custom|mmlu_pro_cf|0", "acc_norm"),
        ("lighteval|gpqa:diamond|0", "gpqa_pass@k_with_k"),
        ("community|gpqa-fr|0", "acc"),
        ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
        ("lighteval|gsm_plus|0", "extractive_match"),
        ("lighteval|aime25|0", "pass@k_with_k&n"),
        ("extended|lcb:codegeneration|0", "codegen_pass@1:16"),
        ("extended|ifeval|0", "prompt_level_loose_acc"),
        ("community|ifeval-fr|0", "prompt_level_loose_acc"),
        ("extended|ifbench_test|0", "prompt_level_loose_acc"),
        ("extended|ifbench_multiturn|1", "prompt_level_loose_acc"),
        ("extended|mixeval_easy:_average|0", "judge_score_flow"),
        ("extended|mixeval_hard:_average|0", "judge_score_flow"),
    ],
}


def assign_colors(df, apply_phase_style=True):
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
        if (
            not apply_phase_style
            or name.split("_phase")[0] != previous_name.split("_phase")[0]
        ) and (name.split(" (")[0] != previous_name.split(" (")[0]):
            i += 1
        previous_name = name
        color_map[name] = colors[i % len(colors)]
    return color_map


def assign_styles(df, apply_phase_style=True):
    unique_experiments = df["expe_name"].unique()
    style_map = {}
    for name in unique_experiments:
        if apply_phase_style:
            if "phase2" in name:
                style_map[name] = ":"
            else:
                style_map[name] = "-"
        else:
            style_map[name] = "-"
    return style_map


def plot_task(
    ax,
    df,
    task,
    metric,
    color_map,
    style_map,
    xlog=False,
    fit=False,
    unit="T_tokens",
    use_dots=False,
    max_tokens=None,
    checkpoint_index=None,
):
    # print(f"Plotting {task} - {metric} with {len(df)} lines")

    xaxis_column = "FLOPs" if unit == "FLOPs" else "tokens"
    xscale = 1 / 1000.0 if unit == "T_tokens" else 1.0
    df = df[(df["task"] == task) & (df["metric"] == metric)]
    if max_tokens:

        def truncate_row(row, max_tokens):
            tokens = row["tokens"]
            # find cutoff index where tokens <= max_tokens
            cutoff = sum(t <= max_tokens for t in tokens)

            # slice lists accordingly
            row["tokens"] = tokens[:cutoff]
            row["FLOPs"] = row["FLOPs"][:cutoff]
            row["score"] = row["score"][:cutoff]
            row["stderr"] = row["stderr"][:cutoff]
            return row

        df = df.apply(truncate_row, axis=1, max_tokens=max_tokens)

    if checkpoint_index is not None:

        def select_checkpoint(row, checkpoint_index):
            try:
                row["tokens"] = [row["tokens"][checkpoint_index]]
                row["FLOPs"] = [row["FLOPs"][checkpoint_index]]
                row["score"] = [row["score"][checkpoint_index]]
                row["stderr"] = [row["stderr"][checkpoint_index]]
            except IndexError:
                raise RuntimeError(
                    f"Checkpoint index {checkpoint_index} out of range for {row['expe_name']} ({len(row['tokens'])} values for task={task}, metric={metric})"
                )
            return row

        df = df.apply(select_checkpoint, axis=1, checkpoint_index=checkpoint_index)

    # Access random
    df_info = get_info()
    task_no_fewshot = "|".join(task.split("|")[:-1])
    if task_no_fewshot in df_info["task"].values:
        num_classes = df_info.loc[
            df_info["task"] == task_no_fewshot, "num_classes"
        ].iloc[0]
        random = 1.0 / num_classes
        ax.axhline(y=random, color="grey", linestyle=":", label="random")

    use_bars = max([len(s) for s in df["score"]]) == 1

    for i, (_, row) in enumerate(df.iterrows()):
        color = color_map[row["expe_name"]]
        linestyle = style_map[row["expe_name"]]

        X = np.array(row[xaxis_column]) * xscale
        Y = np.array(row["score"])

        if use_bars:
            ax.bar(
                i,
                Y,
                color=color,
                label=row["expe_name"],
                yerr=row["stderr"] if "stderr" in row else None,
                capsize=5,
            )
        elif fit:
            ax.plot(
                X,
                Y,
                alpha=np.clip(1 - row["r2"], 0.2, 0.8),
                linestyle=":",
                color=color,
            )

            # Plot regression line
            xaxis = np.linspace(min(X), max(X), 100)
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
            if len(Y) == 1:
                if use_dots:
                    ax.plot(
                        X,
                        Y,
                        marker="+",
                        color=color,
                        markersize=10,  # larger than usual
                        markeredgewidth=2,
                        linewidth=2,
                        label=row["expe_name"],
                    )
                else:
                    # Horizontal line
                    ax.axhline(
                        y=Y,
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label=row["expe_name"]
                        + f" ({X[0]:.2f} {unit.replace('_', ' ')})",
                    )
            else:
                ax.plot(
                    X,
                    Y,
                    marker="o",
                    alpha=1,
                    color=color,
                    linestyle=linestyle,
                    label=row["expe_name"],
                )
                # # Spot the last point
                # ax.plot(
                #     X[-1],
                #     Y[-1],
                #     marker="+",
                #     color=color,
                #     markersize=10,
                #     markeredgewidth=2,
                # )

    if use_bars:
        ax.set_xticks([])
    else:
        ax.set_xlabel(unit.replace("_", " "))
        if xlog:
            ax.set_xscale("log")
    ax.set_ylabel(format_metric_for_title(metric))
    ax.set_title(format_task_for_title(task))


def format_metric_for_title(metric):
    return metric.replace("exact_match_", "em_").split("_")[0].upper()


def format_task_for_title(task):
    f = task.split("|")
    if len(f) == 3:
        task = f[1]
    if task.endswith("_cf"):
        task = task[:-3]
    task = (
        task.replace("_all_", "_")
        .replace("mmlu", "MMLU")
        .replace("arc", "ARC")
        .replace("hellaswag", "HellaSwag")
        .replace("winogrande", "Winogrande")
        .replace("gsm8k", "GSM8K")
        .replace("boolq", "BoolQ")
        .replace("commonsenseqa", "CommonsenseQA")
        .replace("belebele", "Belebele")
        .replace("siqa", "SIQA")
        .replace("openbookqa", "OpenBookQA")
        .replace("piqa", "PIQA")
        .replace("triviaqa", "TriviaQA")
        .replace("mintaka", "Mintaka")
        .replace("fquadv2", "FQuADv2")
        .replace("xcodah", "XCODAH")
        .replace("xcsqa", "XCSQA")
        .replace("xnli", "XNLI")
        .replace("mlmm", "MLMM")
        .replace("flores200", "FLORES")
        .replace("_Latn", "")
        .replace("_", " ")
        .replace(":", " ")
    )
    return task


def plot_list_of_tasks(
    df,
    list_of_tasks_to_plot,
    output_file=None,
    title=None,
    xlog=False,
    fit=False,
    unit="T_tokens",
    use_dots=False,
    apply_phase_style=True,
    max_tokens=None,
    checkpoint_index=None,
    details=False,
    dpi=300,
    max_subplot=19,
):
    if all([metric == "ruler_match" for _, metric in list_of_tasks_to_plot]):

        def full_expe_name(expe_name, tokens):
            if "B training tokens" not in expe_name:
                return f"{expe_name} ({int(tokens)}B training tokens)"
            return expe_name

        # Ruler
        df_filtered = df[df["metric"] == "ruler_match"]
        data = {}
        all_data = {}
        all_context_lengths = set()
        for task, _ in list_of_tasks_to_plot:
            # Extract the context_length from the task : 'custom|ruler_4096:_average|0' -> 4096
            context_length = int(task.split("ruler_")[1].split(":")[0])
            task_prefix = task.split(":")[0]
            subtasks = set(
                [t for t in df["task"] if t.startswith(task_prefix) and t != task]
            )
            all_context_lengths.add(context_length)
            df_task = df_filtered[df_filtered["task"] == task]
            for _, row in df_task.iterrows():
                expe_name = row["expe_name"]
                row_tokens = row["tokens"]
                row_score = row["score"]
                if checkpoint_index is not None:
                    try:
                        row_tokens = [row["tokens"][checkpoint_index]]
                        row_score = [row["score"][checkpoint_index]]
                    except IndexError:
                        raise RuntimeError(
                            f"Checkpoint index {checkpoint_index} out of range for {expe_name} ({len(row['tokens'])} values for task={task})"
                        )
                for tokens, score in zip(row_tokens, row_score):
                    expe_name_with_tokens = full_expe_name(expe_name, tokens)
                    if expe_name_with_tokens not in data:
                        data[expe_name_with_tokens] = {
                            "context_length": [],
                            "score": [],
                        }
                    data[expe_name_with_tokens]["context_length"].append(context_length)
                    data[expe_name_with_tokens]["score"].append(score)
            for subtask in subtasks:
                df_subtask = df_filtered[df_filtered["task"] == subtask]
                subtask = subtask.split(":")[1].split("|")[0]
                all_data[subtask] = all_data.get(subtask, {})
                for _, row in df_subtask.iterrows():
                    expe_name = row["expe_name"]
                    for tokens, score in zip(row["tokens"], row["score"]):
                        expe_name_with_tokens = full_expe_name(expe_name, tokens)
                        if expe_name_with_tokens not in all_data[subtask]:
                            all_data[subtask][expe_name_with_tokens] = {
                                "context_length": [],
                                "score": [],
                            }
                        all_data[subtask][expe_name_with_tokens][
                            "context_length"
                        ].append(context_length)
                        all_data[subtask][expe_name_with_tokens]["score"].append(score)

        if details:
            all_data["average"] = data
        else:
            all_data = {"average": data}

        n_subtasks = len(all_data) + 1
        cols = math.ceil(math.sqrt(n_subtasks))
        rows = math.ceil(n_subtasks / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if n_subtasks > 1 else [axes]
        legend_dict = {}
        for i, (ax, subtask) in enumerate(zip(axes, sorted(all_data.keys()))):
            data = all_data[subtask]
            for expe_name_with_tokens, values in data.items():
                sorted_indices = np.argsort(values["context_length"])
                values["context_length"] = np.array(values["context_length"])[
                    sorted_indices
                ]
                values["score"] = np.array(values["score"])[sorted_indices]
                ax.plot(
                    values["context_length"],
                    values["score"],
                    marker="+",
                    markersize=10,
                    markeredgewidth=2,
                    label=expe_name_with_tokens,
                )
            # if i >= len(all_data) - 4:
            ax.set_xlabel("Context Length")
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                sorted(all_context_lengths),
                labels=[str(cl) for cl in sorted(all_context_lengths)],
            )
            ax.set_ylabel("Ruler Match Score")
            ax.set_title(subtask)
            # ax.legend(title="Experiment name", loc="best")
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_dict[label] = handle

        for ax in axes[len(all_data) :]:
            ax.axis("off")
        ax = axes[-1]
        for handle in legend_dict.values():
            handle.set_alpha(1.0)
        ax.legend(
            legend_dict.values(),
            legend_dict.keys(),
            title="Experiment name",
            loc="center",
        )

    else:
        list_of_tasks_to_plot = [
            task
            for task in list_of_tasks_to_plot
            if task[0] in set(df["task"].unique())
        ]
        n_tasks = len(list_of_tasks_to_plot)
        if not isinstance(max_subplot, int) or n_tasks > max_subplot:
            print("Splitting results in different figures...")
            for i, chunk_list in enumerate(
                [
                    list_of_tasks_to_plot[i : i + max_subplot]
                    for i in range(0, n_tasks, max_subplot)
                ]
                if isinstance(max_subplot, int)
                else [
                    list_of_tasks_to_plot[
                        sum(max_subplot[:i]) : sum(max_subplot[: i + 1])
                    ]
                    for i in range(len(max_subplot))
                ]
            ):
                if output_file:
                    base, ext = os.path.splitext(output_file)
                    chunk_output_file = f"{base}_part{i}{ext}"
                else:
                    chunk_output_file = None
                plot_list_of_tasks(
                    df,
                    chunk_list,
                    output_file=chunk_output_file,
                    title=title,
                    xlog=xlog,
                    fit=fit,
                    unit=unit,
                    use_dots=use_dots,
                    apply_phase_style=apply_phase_style,
                    max_tokens=max_tokens,
                    checkpoint_index=checkpoint_index,
                    details=details,
                    dpi=dpi,
                    max_subplot=max_subplot,
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
        color_map = assign_colors(
            df, apply_phase_style=apply_phase_style
        )  # Global color map
        style_map = assign_styles(df, apply_phase_style=apply_phase_style)

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
                unit=unit,
                use_dots=use_dots,
                max_tokens=max_tokens,
                checkpoint_index=checkpoint_index,
            )

            handles, labels = axes[i].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_dict[label] = handle

        # Dedicated subplot for legend
        legend_ax = axes[-1]
        legend_ax.axis("off")
        # Set legend handle alpha to 1.0
        for handle in legend_dict.values():
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(1.0)

        legend_ax.legend(
            legend_dict.values(),
            legend_dict.keys(),
            title="Experiment name",
            loc="center",
            fontsize=12,
        )

        # Hide any unused subplots
        for j in range(len(list_of_tasks_to_plot), len(axes) - 1):
            fig.delaxes(axes[j])

        if title is not None:
            fig.suptitle(title)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        print(f"Saved figure to {output_file}")


def plot_experiments(df, args, max_subplot=19):
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
        elif g == "agg":
            # Take all the row that have metric == "agg"
            list_of_tasks_to_plot = list(
                df[df["metric"] == "agg"][["task", "metric"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
        else:
            list_of_tasks_to_plot = task_group_mapping[g]

        filename = f'{args.filename_prefix}{g}{"_xlog" if args.xlog else ""}{"_fit" if args.fit else ""}{"_flops" if args.unit == "FLOPs" else ""}{args.filename_suffix}.png'

        output_file = (
            os.path.join(args.output_path, filename) if args.output_path else None
        )
        plot_list_of_tasks(
            df,
            list_of_tasks_to_plot,
            output_file,
            xlog=args.xlog,
            fit=args.fit,
            unit=args.unit,
            use_dots=args.use_dots,
            max_tokens=args.max_tokens,
            max_subplot=max_subplot,
            apply_phase_style=args.apply_phase_style,
            checkpoint_index=args.checkpoint_index,
            details=args.details,
            dpi=args.dpi,
        )

    if not args.output_path:
        plt.show()


def process_experiments(args):
    # Read and aggregate all results
    all_results = []

    if args.legend:
        assert len(args.legend) == len(
            args.experiment_path
        ), "Length of legend must match number of experiment paths."

    for iexpe, path in enumerate(args.experiment_path):
        # Step 1: read experiment results
        df = read_experiment_results(
            path,
            evaluation_dir=args.evaluation_dir,
            expe_name=args.legend[iexpe].replace("_", " ") if args.legend else None,
        )

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
        choices=["all", "agg"] + list(task_group_mapping.keys()),
        default=["all"],
        help="List of predefined groups of tasks you want to plot. You can add groups in the mapping if you want.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )
    parser.add_argument(
        "--filename_prefix",
        type=str,
        default="",
        help="Prefix for the output filename.",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="",
        help="Suffix for the output filename.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation",
    )
    parser.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")
    parser.add_argument("--fit", action="store_true", help="Fit a linear regression")
    parser.add_argument(
        "--unit",
        choices=["B_tokens", "T_tokens", "FLOPs"],
        default="T_tokens",
        help="Unit for x-axis.",
    )
    parser.add_argument(
        "--use_dots",
        action="store_true",
        help="Use dots to represent data points when there is only one point for the curve (otherwise, a horizontal line will be used)",
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
    parser.add_argument("--apply_phase_style", action="store_true")
    parser.add_argument(
        "--checkpoint_index",
        type=int,
        default=None,
        help="If set, only show the specified checkpoint index (ex: 0, -1).",
    )
    parser.add_argument(
        "--legend",
        type=str,
        nargs="*",
        default=[],
        help="List of experiment names to include in the legend.",
    )
    parser.add_argument(
        "--details",
        default=False,
        action="store_true",
        help="If set, show detailed plots for the RULER benchmark (--group ruler).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="If set, save the processed results to a CSV file instead of plotting.",
    )

    args = parser.parse_args()

    df = process_experiments(args)
    print(df)

    if args.save_csv:
        if args.group != ["all"]:
            list_of_tasks_to_plot = [
                task for g in args.group for task in task_group_mapping.get(g, [])
            ]
            mask = (
                df[["task", "metric"]].apply(tuple, axis=1).isin(list_of_tasks_to_plot)
            )
            df = df[mask]
        df["score"] = df["score"].apply(lambda x: x[-1] if isinstance(x, list) else x)
        df["stderr"] = df["stderr"].apply(lambda x: x[-1] if isinstance(x, list) else x)
        df["FLOPs"] = df["FLOPs"].apply(lambda x: x[-1] if isinstance(x, list) else x)
        df["tokens"] = df["tokens"].apply(lambda x: x[-1] if isinstance(x, list) else x)
        df.to_csv(os.path.join(args.output_path, "results.csv"), index=False)

    else:
        plot_experiments(df, args, max_subplot=19)
