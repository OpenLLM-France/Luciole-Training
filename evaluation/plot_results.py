import os
import re
import math
import argparse
from utils import (
    process_results,
    read_experiment_results,
    format_task_for_title,
    task_group_mapping,
)
from agg_score import (
    calculate_agg_score,
    get_info,
    check_benchmarks_by_tasktype,
    normalize_within_range,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Global style configuration
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Curated color palette (colorblind-friendly, high contrast)
_PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
    "#CCB974",  # olive
    "#64B5CD",  # cyan
    "#1F77B4",  # dark blue
    "#FF7F0E",  # vivid orange
    "#2CA02C",  # vivid green
    "#D62728",  # vivid red
    "#9467BD",  # vivid purple
    "#8C564B",  # dark brown
    "#E377C2",  # vivid pink
    "#7F7F7F",  # medium grey
    "#BCBD22",  # yellow-green
    "#17BECF",  # teal
]


def resolve_group(g):
    """Resolve a group spec into a list of (task, metric) tuples.

    A spec is either the name of a predefined group (e.g. ``finetune``) or
    ``group/regex`` (e.g. ``finetune/mixeval``), in which case the part after
    the ``/`` is a regular expression used to keep only the tasks of the base
    group whose task name matches it.
    """
    base, sep, pattern = g.partition("/")
    if base not in task_group_mapping:
        raise ValueError(
            f"Unknown group '{base}'. Available groups: "
            f"{', '.join(sorted(task_group_mapping))}."
        )
    tasks = task_group_mapping[base]
    if not sep:
        selected = tasks
    else:
        if not pattern:
            raise ValueError(f"Empty regex in group spec '{g}'.")
        regex = re.compile(pattern)
        selected = [t for t in tasks if regex.search(t[0])]
        if not selected:
            raise ValueError(f"Regex '{pattern}' matched no task in group '{base}'.")
    # A metric may be a function (derived metric); downstream code matches it against
    # df["metric"] by its name, so normalize callables to their __name__ here.
    return [
        (task, metric.__name__ if callable(metric) else metric)
        for task, metric in selected
    ]


def format_expe_name_for_color(expe_name):
    return (
        expe_name.replace("-Instruct", "")
        .replace("-instruct", "")
        .replace("-Base", "")
        # .replace("-SFT", "")
        .replace("-v1.1", "")
    )


_VALID_COLOR_CHARS = set("bgrcmykw")
_VALID_LINESTYLES = {"-", "--", "-.", ":"}


def parse_color_spec(spec):
    """Parse a --color string into a list of (color, linestyle) tuples.

    Each system is described by a single-character matplotlib color code (one
    of 'bgrcmykw') optionally followed by a linestyle ('-', '--', '-.', ':').
    A missing linestyle defaults to solid '-'. Systems are concatenated with no
    separator; the next color letter starts the next system.

    Example:
        'gg--g:bb--b:' (equivalently 'g-g--g:b-b--b:') ->
        [('g', '-'), ('g', '--'), ('g', ':'),
         ('b', '-'), ('b', '--'), ('b', ':')]
    """
    linestyle_chars = set("-.:")
    result = []
    i, n = 0, len(spec)
    while i < n:
        color = spec[i]
        if color not in _VALID_COLOR_CHARS:
            raise ValueError(
                f"Invalid color '{color}' in --color spec '{spec}'. "
                f"Expected one of '{''.join(sorted(_VALID_COLOR_CHARS))}'."
            )
        i += 1
        j = i
        while j < n and spec[j] in linestyle_chars:
            j += 1
        linestyle = spec[i:j] if j > i else "-"
        if linestyle not in _VALID_LINESTYLES:
            raise ValueError(
                f"Invalid linestyle '{linestyle}' in --color spec '{spec}'. "
                f"Expected one of {sorted(_VALID_LINESTYLES)}."
            )
        result.append((color, linestyle))
        i = j
    if not result:
        raise ValueError(f"Empty --color spec '{spec}'.")
    return result


def assign_colors(df, apply_phase_style=True, color_spec=None):
    unique_experiments = df["expe_name"].unique()
    if color_spec is not None:
        parsed = parse_color_spec(color_spec)
        return {
            name: parsed[i % len(parsed)][0]
            for i, name in enumerate(unique_experiments)
        }
    colors = _PALETTE
    color_map = {}
    i = -1
    previous_name = ""
    for name in unique_experiments:
        if (
            not apply_phase_style
            or name.split("_phase")[0] != previous_name.split("_phase")[0]
        ) and (
            format_expe_name_for_color(name).split(" (")[0]
            != format_expe_name_for_color(previous_name).split(" (")[0]
        ):
            i += 1
        previous_name = name
        color_map[name] = colors[i % len(colors)]
    return color_map


def assign_styles(df, apply_phase_style=True, color_spec=None):
    unique_experiments = df["expe_name"].unique()
    if color_spec is not None:
        parsed = parse_color_spec(color_spec)
        return {
            name: parsed[i % len(parsed)][1]
            for i, name in enumerate(unique_experiments)
        }
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


def _resolve_ymax(ymax, i):
    """Return the ymax value for the i-th detail plot.

    ``ymax`` is a list of floats (or None). If it has fewer entries than plots,
    the last value applies to all remaining plots.
    """
    if not ymax:
        return None
    return ymax[i] if i < len(ymax) else ymax[-1]


def _annotate_numbers(ax, X, Y, color, yerr=None):
    """Print each Y value as text, horizontally centered above its point.

    A small constant vertical gap (in points) separates the bottom of the text
    from the top of the marker/bar (or the top of the error bar when ``yerr`` is
    given), so it is independent of the data scale.
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    err = np.atleast_1d(yerr) if yerr is not None else None
    for i in range(len(Y)):
        y_top = Y[i]
        if err is not None:
            e = err[i] if i < len(err) else err[-1]
            if e is not None and not np.isnan(e):
                y_top = y_top + e
        ax.annotate(
            f"{Y[i]:.3g}",
            xy=(X[i], y_top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            zorder=6,
            clip_on=False,
        )


def _plot_curves(
    ax,
    series,
    color_map,
    style_map,
    unit="T_tokens",
    xlog=False,
    use_dots=False,
    print_numbers=False,
):
    """Plot a list of series on a single axis.

    Each element of `series` is a dict with keys:
        expe_name, X (array), Y (array),
        and optionally: stderr, r2, slope, intercept (for fit mode).
    """
    xscale = 1 / 1000.0 if unit == "T_tokens" else 1.0
    use_bars = all(len(s["Y"]) == 1 for s in series)

    maxX_nodots = (
        None
        if use_bars
        else max((max(s["X"]) * xscale for s in series if len(s["Y"]) > 1), default=0)
    )

    # Collect out-of-range single-point stars to draw after axis limits are set
    deferred_stars = []

    for i, s in enumerate(series):
        color = color_map[s["expe_name"]]
        linestyle = style_map[s["expe_name"]]
        label = format_expename_for_title(s["expe_name"])

        X = np.array(s["X"]) * xscale
        Y = np.array(s["Y"])

        if use_bars:
            ax.bar(
                i,
                Y,
                color=color,
                label=label,
                yerr=s.get("stderr"),
                capsize=4,
                edgecolor="white",
                linewidth=0.5,
            )
            if print_numbers:
                _annotate_numbers(ax, [i], Y, color, yerr=s.get("stderr"))
        elif "r2" in s:
            ax.plot(
                X,
                Y,
                alpha=np.clip(1 - s["r2"], 0.2, 0.8),
                linestyle=":",
                color=color,
            )

            xaxis = np.linspace(min(X), max(X), 100)
            y_pred = s["intercept"] + s["slope"] * np.log(xaxis)
            ax.plot(
                xaxis,
                y_pred,
                linestyle="-",
                alpha=np.clip(s["r2"], 0.2, 0.8),
                color=color,
                label=label,
            )

            ax.text(
                xaxis[-1],
                y_pred[-1],
                f"$R^2$={s['r2']:.2f}",
                color=color,
                fontsize=8,
                ha="left",
                va="center",
            )
            if print_numbers:
                _annotate_numbers(ax, X, Y, color)
        else:
            if len(Y) == 1:
                if use_dots:
                    ax.plot(
                        X,
                        Y,
                        marker="+",
                        color=color,
                        markersize=10,
                        markeredgewidth=2,
                        linewidth=2,
                        label=label,
                    )
                    if print_numbers:
                        _annotate_numbers(ax, X, Y, color)
                else:
                    ax.axhline(
                        y=Y,
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label=label + f" ({X[0]:.3g}{format_unit(unit)})",
                    )
                    if X[0] <= maxX_nodots:
                        ax.plot(
                            X[-1],
                            Y[-1],
                            marker="*",
                            color=color,
                            markersize=15,
                            markeredgecolor="white",
                            markeredgewidth=0.5,
                        )
                        if print_numbers:
                            _annotate_numbers(ax, [X[-1]], [Y[-1]], color)
                    else:
                        deferred_stars.append((Y[0], color))
            else:
                ax.plot(
                    X,
                    Y,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    label=label,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )
                if print_numbers:
                    _annotate_numbers(ax, X, Y, color)
                if not use_dots:
                    ax.axhline(
                        y=Y[-1],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                    )
                    ax.plot(
                        X[-1],
                        Y[-1],
                        marker="*",
                        color=color,
                        markersize=15,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                    )

    # Draw deferred out-of-range stars at the right edge with an arrow
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if deferred_stars:
        # Force matplotlib to compute axis limits from existing data
        ax.autoscale_view()
        x_star = xmax
        arrow_len = (xmax - xmin) * 0.12
        y_offset = (ymax - ymin) * 0.04
        arrow_tip_x = xmax + (xmax - xmin) * 0.06
        for y_val, color in deferred_stars:
            ax.plot(
                x_star,
                y_val,
                marker="*",
                color=color,
                markersize=15,
                markeredgecolor="white",
                markeredgewidth=0.5,
                clip_on=False,
                zorder=5,
            )
            if print_numbers:
                _annotate_numbers(ax, [x_star], [y_val], color)
            # Horizontal arrow pointing right, just above the star
            arrow_y = y_val + y_offset
            ax.annotate(
                "",
                xy=(arrow_tip_x, arrow_y),
                xytext=(arrow_tip_x - arrow_len, arrow_y),
                arrowprops=dict(
                    arrowstyle="->,head_length=0.6,head_width=0.4",
                    color=color,
                    lw=2,
                ),
                clip_on=False,
                annotation_clip=False,
            )

    if use_bars:
        ax.set_xticks([])
    else:
        ax.set_xlim(max(0, xmin), xmax)
        ax.set_xlabel(format_unit(unit))
        if xlog:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="both", which="both", length=3)


def get_checkpoint_index(checkpoint_index, expe_name):
    if checkpoint_index is None:
        return None
    if checkpoint_index == "last":
        return -1
    if checkpoint_index == "last-not-luciole":
        if "luciole" in expe_name.lower():
            return None
        else:
            return -1
    try:
        return int(checkpoint_index)
    except ValueError:
        raise ValueError(
            f"Invalid checkpoint index '{checkpoint_index}' for experiment '{expe_name}'. Must be an integer or 'last'."
        )


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
    print_numbers=False,
):
    xaxis_column = "FLOPs" if unit == "FLOPs" else "tokens"
    df = df[(df["task"] == task) & (df["metric"] == metric)]
    if max_tokens:

        def truncate_row(row, max_tokens):
            tokens = row["tokens"]
            cutoff = sum(t <= max_tokens for t in tokens)
            row["tokens"] = tokens[:cutoff]
            row["FLOPs"] = row["FLOPs"][:cutoff]
            row["score"] = row["score"][:cutoff]
            row["stderr"] = row["stderr"][:cutoff]
            return row

        df = df.apply(truncate_row, axis=1, max_tokens=max_tokens)

    if checkpoint_index is not None:

        def select_checkpoint(row, checkpoint_index):
            expe_name = row["expe_name"]
            actual_checkpoint_index = get_checkpoint_index(checkpoint_index, expe_name)
            if actual_checkpoint_index is not None:
                try:
                    row["tokens"] = [row["tokens"][actual_checkpoint_index]]
                    row["FLOPs"] = [row["FLOPs"][actual_checkpoint_index]]
                    row["score"] = [row["score"][actual_checkpoint_index]]
                    row["stderr"] = [row["stderr"][actual_checkpoint_index]]
                except IndexError:
                    raise RuntimeError(
                        f"Checkpoint index {actual_checkpoint_index} out of range for {row['expe_name']} ({len(row['tokens'])} values for task={task}, metric={metric})"
                    )
            return row

        df = df.apply(select_checkpoint, axis=1, checkpoint_index=checkpoint_index)

    # Draw random baseline
    df_info = get_info()
    task_no_fewshot = "|".join(task.split("|")[:-1])
    if task_no_fewshot in df_info["task"].values:
        num_classes = df_info.loc[
            df_info["task"] == task_no_fewshot, "num_classes"
        ].iloc[0]
        random = 1.0 / num_classes
        ax.axhline(
            y=random, color="#999999", linestyle=":", linewidth=2, label="random"
        )

    # Build series list
    series = []
    for _, row in df.iterrows():
        s = {
            "expe_name": row["expe_name"],
            "X": row[xaxis_column],
            "Y": row["score"],
        }
        if "stderr" in row:
            s["stderr"] = row["stderr"]
        if fit and "r2" in row:
            s["r2"] = row["r2"]
            s["slope"] = row["slope"]
            s["intercept"] = row["intercept"]
        series.append(s)

    _plot_curves(
        ax,
        series,
        color_map,
        style_map,
        unit=unit,
        xlog=xlog,
        use_dots=use_dots,
        print_numbers=print_numbers,
    )
    ax.set_ylabel(format_metric_for_title(metric))
    ax.set_title(format_task_for_title(task))


# Filler words ignored when deriving a metric title (e.g. extractive_match -> "MATCH").
_METRIC_TITLE_IGNORE = {"answer", "extractive"}
# Metric name suffixes that denote the kind of score, longest first.
_METRIC_TITLE_KINDS = [
    ("_f1_score", "F1"),
    ("_f1", "F1"),
    ("_acc", "ACC"),
    ("_em", "EM"),
    ("_precision_score", "PRECISION"),
    ("_recall_score", "RECALL"),
    ("_recall", "RECALL"),
    ("_precision", "PRECISION"),
]


def format_metric_for_title(metric):
    # Metrics named "<prefix>_<kind>" (kind in f1/f1_score/acc/em): show the kind,
    # prefixed by a single meaningful word if one remains (e.g. refusal_f1 -> "Refusal F1",
    # qa_f1_score -> "QA F1"), otherwise just the kind (e.g. answer_em -> "EM").
    for suffix, kind in _METRIC_TITLE_KINDS:
        if metric.endswith(suffix):
            tokens = [
                t
                for t in metric[: -len(suffix)].split("_")
                if t and t not in _METRIC_TITLE_IGNORE
            ]
            if len(tokens) == 1:
                word = tokens[0]
                word = word.upper() if len(word) <= 2 else word.title()
                return f"{word} {kind}"
            return kind
    # Default heuristic: first token (ignoring filler words), uppercased.
    tokens = [
        t
        for t in metric.replace("exact_match_", "em_").split("_")
        if t and t not in _METRIC_TITLE_IGNORE
    ]
    return tokens[0].upper() if tokens else metric.upper()


def format_expename_for_title(expe_name):
    if expe_name.endswith("_noct"):
        return expe_name[:-5]
    return expe_name


def format_unit(unit):
    return unit.replace("_", " ").replace("tokens", "training tokens")


def format_group_name_for_title(group_name):
    return {
        "en": "English",
        "fr": "French",
        "multilingual": "Other Languages",
        "translation": "Translation",
    }.get(group_name, None)


def _sort_legend_dict(legend_dict, df):
    """Sort legend entries by experiment order in df, with 'random' always last."""
    label_order = [format_expename_for_title(name) for name in df["expe_name"].unique()]

    def sort_key(label):
        if label == "random":
            return (1, 0)
        try:
            return (0, label_order.index(label))
        except ValueError:
            return (0, len(label_order))

    return dict(sorted(legend_dict.items(), key=lambda item: sort_key(item[0])))


def plot_aggregate(
    ax,
    df,
    list_of_tasks_to_plot,
    color_map,
    style_map,
    xlog=False,
    unit="T_tokens",
    use_dots=False,
    max_tokens=None,
    checkpoint_index=None,
    title=None,
    print_numbers=False,
):
    """Plot the average normalized score across all benchmarks in the list."""
    df_info = get_info()
    xaxis_column = "FLOPs" if unit == "FLOPs" else "tokens"

    # Build a lookup for random baselines: task_base -> random
    random_lookup = {}
    for _, row in df_info.iterrows():
        random_lookup[row["task"]] = row["random"]

    # Track which tasks each experiment has results for
    # expe_name -> set of (task, metric)
    experiment_tasks = {}

    # Collect per-experiment normalized scores at each checkpoint
    # expe_name -> {(tokens, xval) -> {"scores": [...], "tasks": set()}}
    # We track the contributing tasks per checkpoint (not just the scores) so we
    # can later keep only checkpoints that cover the full benchmark set.
    experiment_data = {}

    averaged_metrics = set()
    all_tasks = set()
    for task, metric in list_of_tasks_to_plot:
        if metric in ("bleu", "bleu_4", "metricx"):
            continue
        averaged_metrics.add(metric)
        all_tasks.add((task, metric))

        task_base = "|".join(task.split("|")[:-1])
        random_baseline = random_lookup.get(task_base, 0.0)

        df_task = df[(df["task"] == task) & (df["metric"] == metric)]
        for _, row in df_task.iterrows():
            expe_name = row["expe_name"]
            if expe_name not in experiment_data:
                experiment_data[expe_name] = {}
                experiment_tasks[expe_name] = set()
            experiment_tasks[expe_name].add((task, metric))

            tokens_list = row["tokens"]
            scores_list = row["score"]
            flops_list = row[xaxis_column]

            if checkpoint_index is not None:
                actual_checkpoint_index = get_checkpoint_index(
                    checkpoint_index, expe_name
                )
                if actual_checkpoint_index is not None:
                    try:
                        tokens_list = [tokens_list[actual_checkpoint_index]]
                        scores_list = [scores_list[actual_checkpoint_index]]
                        flops_list = [flops_list[actual_checkpoint_index]]
                    except IndexError:
                        continue

            if max_tokens:
                cutoff = sum(t <= max_tokens for t in tokens_list)
                tokens_list = tokens_list[:cutoff]
                scores_list = scores_list[:cutoff]
                flops_list = flops_list[:cutoff]

            for tokens_val, score_val, xval in zip(
                tokens_list, scores_list, flops_list
            ):
                key = (tokens_val, xval)
                if key not in experiment_data[expe_name]:
                    experiment_data[expe_name][key] = {"scores": [], "tasks": set()}
                norm_score = normalize_within_range(score_val, random_baseline, 1.0)
                experiment_data[expe_name][key]["scores"].append(norm_score)
                experiment_data[expe_name][key]["tasks"].add((task, metric))

    # Exclude experiments missing at least one task
    num_tasks = len(all_tasks)
    incomplete = {
        name: all_tasks - tasks
        for name, tasks in experiment_tasks.items()
        if tasks != all_tasks
    }
    for name, missing in incomplete.items():
        missing_names = [format_task_for_title(t) for t, _ in sorted(missing)]
        print(
            f"WARNING: '{name}' excluded from aggregate (missing {len(missing)}/{num_tasks} tasks: {', '.join(missing_names)})"
        )
        del experiment_data[name]

    # Build series from aggregated data.
    # Only keep checkpoints that cover the full benchmark set, so that every
    # plotted point is the average over the *same* benchmarks. Benchmarks are
    # often evaluated at different sets of checkpoints (e.g. some only at the
    # final checkpoint): averaging whatever happens to be present at each
    # checkpoint mixes a varying subset and produces a misleading curve (an
    # apparent "drop" at the last point where the final-only benchmarks join).
    series = []
    for expe_name, data in experiment_data.items():
        full_keys = [k for k, v in data.items() if len(v["tasks"]) == num_tasks]
        dropped = len(data) - len(full_keys)
        if dropped:
            print(
                f"INFO: '{expe_name}' aggregate: kept {len(full_keys)}/{len(data)} "
                f"checkpoint(s) covering all {num_tasks} benchmarks; dropped "
                f"{dropped} partially-evaluated checkpoint(s) from the average."
            )
        if not full_keys:
            print(
                f"WARNING: '{expe_name}' has no checkpoint covering all {num_tasks} "
                f"benchmarks; excluded from aggregate."
            )
            continue
        sorted_keys = sorted(full_keys)
        series.append(
            {
                "expe_name": expe_name,
                "X": [k[1] for k in sorted_keys],
                "Y": [np.mean(data[k]["scores"]) for k in sorted_keys],
            }
        )

    _plot_curves(
        ax,
        series,
        color_map,
        style_map,
        unit=unit,
        xlog=xlog,
        use_dots=use_dots,
        print_numbers=print_numbers,
    )
    ax.set_ylabel(
        "Averaged "
        + (
            "Normalized Score"
            if len(averaged_metrics) > 1
            else format_metric_for_title(next(iter(averaged_metrics)))
        )
    )
    ax.set_title(title if title else "Overall Performance")


def _draw_legend(ax_or_fig, legend_dict, as_figure=False):
    """Draw a styled legend on an axis or as a standalone figure."""
    for handle in legend_dict.values():
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)

    target = ax_or_fig
    if as_figure:
        target = ax_or_fig  # a figure, use figlegend
        leg = target.legend(
            legend_dict.values(),
            legend_dict.keys(),
            title="Model",
            loc="center",
            fontsize=11,
            title_fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.9,
            borderpad=1.0,
            labelspacing=0.8,
            handlelength=2.5,
        )
    else:
        leg = target.legend(
            legend_dict.values(),
            legend_dict.keys(),
            title="Model",
            loc="center",
            fontsize=11,
            title_fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.9,
            borderpad=1.0,
            labelspacing=0.8,
            handlelength=2.5,
        )
    leg.get_title().set_fontweight("bold")


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
    hide_details=False,
    dpi=150,
    max_subplot=20,
    add_aggregate=False,
    separate_legend=False,
    rows_cols=None,
    color_spec=None,
    suptitle=None,
    print_numbers=False,
    ymax=None,
):
    legend_fig = None
    if all([metric == "ruler_match" for _, metric in list_of_tasks_to_plot]):

        def full_expe_name(expe_name, tokens):
            if "B training tokens" not in expe_name:
                return f"{expe_name} ({int(tokens)}B training tokens)"
            return expe_name

        # Ruler
        color_map = assign_colors(
            df, apply_phase_style=apply_phase_style, color_spec=color_spec
        )
        style_map = assign_styles(
            df, apply_phase_style=apply_phase_style, color_spec=color_spec
        )
        ruler_color_map = {}  # maps expe_name_with_tokens -> color
        ruler_style_map = {}  # maps expe_name_with_tokens -> linestyle
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
                    actual_checkpoint_index = get_checkpoint_index(
                        checkpoint_index, expe_name
                    )
                    if actual_checkpoint_index is not None:
                        try:
                            row_tokens = [row["tokens"][actual_checkpoint_index]]
                            row_score = [row["score"][actual_checkpoint_index]]
                        except IndexError:
                            raise RuntimeError(
                                f"Checkpoint index {actual_checkpoint_index} out of range for {expe_name} ({len(row['tokens'])} values for task={task})"
                            )
                for tokens, score in zip(row_tokens, row_score):
                    expe_name_with_tokens = full_expe_name(expe_name, tokens)
                    if expe_name_with_tokens not in data:
                        data[expe_name_with_tokens] = {
                            "context_length": [],
                            "score": [],
                        }
                    if (
                        expe_name_with_tokens not in ruler_color_map
                        and expe_name in color_map
                    ):
                        ruler_color_map[expe_name_with_tokens] = color_map[expe_name]
                        ruler_style_map[expe_name_with_tokens] = style_map[expe_name]
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
                        if (
                            expe_name_with_tokens not in ruler_color_map
                            and expe_name in color_map
                        ):
                            ruler_color_map[expe_name_with_tokens] = color_map[
                                expe_name
                            ]
                            ruler_style_map[expe_name_with_tokens] = style_map[
                                expe_name
                            ]
                        if expe_name_with_tokens not in all_data[subtask]:
                            all_data[subtask][expe_name_with_tokens] = {
                                "context_length": [],
                                "score": [],
                            }
                        all_data[subtask][expe_name_with_tokens][
                            "context_length"
                        ].append(context_length)
                        all_data[subtask][expe_name_with_tokens]["score"].append(score)

        if hide_details:
            all_data = {"average": data}
        elif add_aggregate:
            all_data["average"] = data

        has_average = "average" in all_data
        detail_subtasks = sorted(k for k in all_data if k != "average")
        n_details = len(detail_subtasks)

        # Determine stable ordering of experiments (first one gets solid line)
        ruler_expe_order = list(
            dict.fromkeys(
                name for subtask_data in all_data.values() for name in subtask_data
            )
        )

        def _plot_ruler_on_ax(ax, subtask_name):
            subtask_data = all_data[subtask_name]
            for expe_name_with_tokens, values in subtask_data.items():
                sorted_indices = np.argsort(values["context_length"])
                values["context_length"] = np.array(values["context_length"])[
                    sorted_indices
                ]
                values["score"] = np.array(values["score"])[sorted_indices]
                is_first = (
                    ruler_expe_order.index(expe_name_with_tokens) == 0
                    if expe_name_with_tokens in ruler_expe_order
                    else False
                )
                color = ruler_color_map.get(expe_name_with_tokens)
                if color_spec is not None:
                    linestyle = ruler_style_map.get(expe_name_with_tokens, "-")
                else:
                    linestyle = "-" if is_first else "--"
                ax.plot(
                    values["context_length"],
                    values["score"],
                    marker="o",  # if is_first else None,
                    # markersize=10,
                    # markeredgewidth=2,
                    linestyle=linestyle,
                    label=expe_name_with_tokens,
                    color=color,
                )
                if print_numbers:
                    _annotate_numbers(
                        ax, values["context_length"], values["score"], color
                    )
            ax.set_xlabel("Context Length")
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                sorted(all_context_lengths),
                labels=[str(cl) for cl in sorted(all_context_lengths)],
            )
            ax.set_ylabel("Ruler Match Score")
            ax.set_title(format_task_for_title(subtask_name))
            if subtask_name == "average":
                # Visually emphasize the average subplot
                ax.set_facecolor("#f7f7f7")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#888888")
                    spine.set_linewidth(1.5)
                ax.set_title(
                    "RULER" if hide_details else "Overall Performance (RULER)",
                    fontsize=12,
                    fontweight="heavy",
                    # fontstyle="italic",
                )
                ax.set_ylabel("Average Ruler Match Score")
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_dict[label] = handle
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, 1)  # ymax)

        if has_average and n_details > 0:
            # First row: average + legend; remaining rows: detail subtasks
            detail_cols = math.ceil(math.sqrt(n_details))
            cols = max(2, detail_cols) if not separate_legend else max(1, detail_cols)
            detail_rows = math.ceil(n_details / cols)
            rows = 1 + detail_rows
            detail_start = cols
        elif has_average:
            # Only average (hide_details)
            cols = 1 if separate_legend else 2
            rows = 1
            detail_start = cols
        else:
            # Only details, no average
            n_plots = n_details + (0 if separate_legend else 1)
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)
            detail_start = 0

        if rows_cols is not None:
            rows, cols = rows_cols
            n_needed = (
                (1 if has_average else 0) + n_details + (0 if separate_legend else 1)
            )
            assert (
                rows * cols >= n_needed
            ), f"--rows_cols {rows}x{cols} = {rows * cols} subplots, but {n_needed} are needed"
            detail_start = cols if has_average else 0

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        legend_dict = {}

        if has_average:
            _plot_ruler_on_ax(axes[0], "average")
            # Hide xlabel when detail rows exist below
            if n_details > 0:
                axes[0].set_xlabel("")

        for i, subtask in enumerate(detail_subtasks):
            _plot_ruler_on_ax(axes[detail_start + i], subtask)
            ymax_val = _resolve_ymax(ymax, i)
            if ymax_val is not None:
                axes[detail_start + i].set_ylim(top=ymax_val)

        # Hide xlabel on non-bottom-row detail subplots
        for i in range(n_details):
            if i + cols < n_details:
                axes[detail_start + i].set_xlabel("")

        # Clean up unused axes
        used = set()
        if has_average:
            used.add(0)
        for i in range(n_details):
            used.add(detail_start + i)

        legend_dict = _sort_legend_dict(legend_dict, df)
        if separate_legend:
            legend_fig = plt.figure(figsize=(4, max(2, 0.4 * len(legend_dict))))
            _draw_legend(legend_fig, legend_dict, as_figure=True)
        else:
            legend_idx = 1 if has_average else (rows * cols - 1)
            used.add(legend_idx)
            axes[legend_idx].axis("off")
            _draw_legend(axes[legend_idx], legend_dict)

        for j in range(rows * cols):
            if j not in used:
                fig.delaxes(axes[j])

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
                    hide_details=hide_details,
                    dpi=dpi,
                    max_subplot=max_subplot,
                    add_aggregate=add_aggregate,
                    separate_legend=separate_legend,
                    rows_cols=rows_cols,
                    color_spec=color_spec,
                    suptitle=suptitle,
                    print_numbers=print_numbers,
                    ymax=ymax,
                )
            return

        if hide_details and add_aggregate:
            num_tasks = 0
        else:
            num_tasks = len(list_of_tasks_to_plot)

        color_map = assign_colors(
            df, apply_phase_style=apply_phase_style, color_spec=color_spec
        )  # Global color map
        style_map = assign_styles(
            df, apply_phase_style=apply_phase_style, color_spec=color_spec
        )

        if add_aggregate:
            # Layout: first row for aggregate + legend, remaining rows for details
            if num_tasks > 0:
                detail_cols = math.ceil(math.sqrt(num_tasks))
                cols = (
                    max(2, detail_cols) if not separate_legend else max(1, detail_cols)
                )
                detail_rows = math.ceil(num_tasks / cols)
            else:
                cols = 1 if separate_legend else 2
                detail_rows = 0
            rows = 1 + detail_rows
            detail_start = cols
        else:
            # No aggregate: flat layout for details + legend
            num_plots = num_tasks + (0 if separate_legend else 1)
            cols = math.ceil(math.sqrt(num_plots))
            rows = math.ceil(num_plots / cols)
            detail_start = 0

        if rows_cols is not None:
            rows, cols = rows_cols
            n_needed = (
                (1 if add_aggregate else 0) + num_tasks + (0 if separate_legend else 1)
            )
            assert (
                rows * cols >= n_needed
            ), f"--rows_cols {rows}x{cols} = {rows * cols} subplots, but {n_needed} are needed"
            detail_start = cols if add_aggregate else 0

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        legend_dict = {}
        used = set()

        if add_aggregate:
            agg_ax = axes[0]
            used.add(0)
            plot_aggregate(
                agg_ax,
                df,
                list_of_tasks_to_plot,
                color_map=color_map,
                style_map=style_map,
                xlog=xlog,
                unit=unit,
                use_dots=use_dots,
                max_tokens=max_tokens,
                checkpoint_index=checkpoint_index,
                title=title
                if hide_details
                else (f"Overall Performance ({title})" if title else None),
                print_numbers=print_numbers,
            )
            # Visually emphasize the aggregate subplot
            agg_ax.set_facecolor("#f7f7f7")
            for spine in agg_ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#888888")
                spine.set_linewidth(1.5)
            agg_ax.set_title(
                agg_ax.get_title(),
                fontsize=12,
                fontweight="heavy",
                # fontstyle="italic",
            )
            handles, labels = agg_ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_dict[label] = handle
            # Hide xlabel when detail rows exist below
            if num_tasks > 0:
                agg_ax.set_xlabel("")

        for i, (task, metric) in enumerate(list_of_tasks_to_plot[:num_tasks]):
            ax_idx = detail_start + i
            used.add(ax_idx)
            plot_task(
                axes[ax_idx],
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
                print_numbers=print_numbers,
            )

            ymax_val = _resolve_ymax(ymax, i)
            if ymax_val is not None:
                axes[ax_idx].set_ylim(top=ymax_val)

            handles, labels = axes[ax_idx].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_dict[label] = handle

        legend_dict = _sort_legend_dict(legend_dict, df)
        if separate_legend:
            legend_fig = plt.figure(figsize=(4, max(2, 0.4 * len(legend_dict))))
            _draw_legend(legend_fig, legend_dict, as_figure=True)
        else:
            legend_idx = 1 if add_aggregate else (rows * cols - 1)
            used.add(legend_idx)
            axes[legend_idx].axis("off")
            _draw_legend(axes[legend_idx], legend_dict)

        # Remove unused axes
        for j in range(rows * cols):
            if j not in used:
                fig.delaxes(axes[j])

        # Hide xlabel on non-bottom-row detail subplots
        for i in range(num_tasks):
            if i + cols < num_tasks:
                axes[detail_start + i].set_xlabel("")

        if title and not add_aggregate and not suptitle:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)

    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_file}")

    # Save or show the separate legend figure if it exists
    if legend_fig is not None:
        if output_file:
            base, ext = os.path.splitext(output_file)
            legend_file = f"{base}_legend{ext}"
            legend_fig.savefig(legend_file, dpi=dpi, bbox_inches="tight")
            print(f"Saved legend to {legend_file}")
            plt.close(legend_fig)


def check_all_systems_have_results(
    df, list_of_tasks_to_plot, group_name, ignore_no_results=False
):
    """Raise if any system (expe_name) has no result among the tasks to plot.

    With ignore_no_results=True, warn instead of raising.
    """
    task_metric_pairs = set(list_of_tasks_to_plot)
    plotted_systems = set()
    for _, row in df.iterrows():
        if (row["task"], row["metric"]) not in task_metric_pairs:
            continue
        score = row["score"]
        has_data = (
            len(score) > 0
            if isinstance(score, (list, tuple, np.ndarray))
            else score is not None
        )
        if has_data:
            plotted_systems.add(row["expe_name"])
    missing = sorted(set(df["expe_name"].unique()) - plotted_systems)
    if missing:
        message = (
            f"No results to plot for group '{group_name}' for the following "
            f"system(s): {', '.join(missing)}."
        )
        if ignore_no_results:
            print(f"WARNING: {message}")
        else:
            raise RuntimeError(message)


def plot_experiments(df, args, max_subplot=20):
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
            list_of_tasks_to_plot = resolve_group(g)

        check_all_systems_have_results(
            df, list_of_tasks_to_plot, g, ignore_no_results=args.ignore_no_results
        )

        # Skip the aggregate/average subplot when there is a single benchmark:
        # the average would be identical to that benchmark's own plot.
        add_aggregate = (
            g not in ("all", "agg")
            and not args.hide_average
            and len(list_of_tasks_to_plot) > 1
        )
        info_str = f"{'_xlog' if args.xlog else ''}{'_fit' if args.fit else ''}{'_flops' if args.unit == 'FLOPs' else ''}"
        info_str += "_average" if (args.hide_details and add_aggregate) else "_details"
        g_for_filename = g.replace("/", "_")
        filename = f"{args.filename_prefix}{g_for_filename}{info_str}{args.filename_suffix}.png"

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
            hide_details=args.hide_details,
            dpi=args.dpi,
            add_aggregate=add_aggregate,
            separate_legend=args.separate_legend,
            title=format_group_name_for_title(g),
            rows_cols=args.rows_cols,
            color_spec=args.color,
            suptitle=args.title,
            print_numbers=args.print_numbers,
            ymax=args.ymax,
        )

    if not args.output_path:
        plt.show()


def process_experiments(args):
    # Read and aggregate all results
    all_results = []

    if args.legend:
        assert len(args.legend) <= len(
            args.experiment_path
        ), "Length of legend must match number of experiment paths."
        args.legend = [legend.replace("_", " ") for legend in args.legend]
        if len(args.legend) < len(args.experiment_path):
            args.legend += [None] * (len(args.experiment_path) - len(args.legend))

    benchmarks_per_tasktype_ref = None
    missing_systems = []
    for iexpe, path in enumerate(args.experiment_path):
        # Step 1: read experiment results
        expe_name = args.legend[iexpe] if args.legend else None
        df = read_experiment_results(
            path,
            evaluation_dir=args.evaluation_dir,
            expe_name=expe_name,
        )

        if df is None or df.empty:
            missing_systems.append(expe_name if expe_name else path)
            continue

        # Step 2: calculate aggregated scores if needed
        if "agg" in args.group:
            benchmarks_per_tasktype, df_agg = calculate_agg_score(
                df, check_aggregation=args.check_aggregation
            )
            df_agg = df_agg.dropna()

            # Check that the benchmarks per task type are the same across experiments
            # (otherwise, the aggregated scores are not comparable)
            if benchmarks_per_tasktype_ref is None:
                benchmarks_per_tasktype_ref = benchmarks_per_tasktype
                ref_name = expe_name if expe_name else path
            else:
                check_benchmarks_by_tasktype(
                    benchmarks_per_tasktype_ref,
                    benchmarks_per_tasktype,
                    ref_name,
                    expe_name if expe_name else path,
                )

            df = pd.concat([df, df_agg])

        # Step 3: process the results
        df = process_results(df, fit=args.fit, window=args.window)

        # Step 4: collect results
        all_results.append(df)

    if missing_systems:
        message = "No results found for the following system(s): " + ", ".join(
            missing_systems
        )
        if args.ignore_no_results:
            print(f"WARNING: {message}")
        else:
            raise RuntimeError(message)

    # Combine all results into a single DataFrame
    final_df = pd.concat(all_results, ignore_index=True)
    if final_df.empty:
        print("No results found for the given experiments.")
        exit(0)

    if benchmarks_per_tasktype_ref is not None:
        print("===== AGGREGATED BENCHMARKS PER TASK TYPE =====")
        for (task_type, language), benchmarks in sorted(
            benchmarks_per_tasktype_ref.items()
        ):
            print(f"[{task_type} / {language}]")
            for benchmark in sorted(benchmarks):
                print(f"  - {format_task_for_title(benchmark)}")
            print()

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
        default=["all"],
        help="List of predefined groups of tasks you want to plot (you can add "
        "groups in the mapping if you want), plus the special groups 'all' and "
        "'agg'. A group can be restricted to a subset of its tasks with the "
        "'group/regex' syntax: e.g. 'finetune/mixeval' keeps only the tasks of "
        "the 'finetune' group whose name matches the regex 'mixeval'. "
        f"Available groups: {', '.join(['all', 'agg'] + list(task_group_mapping.keys()))}.",
    )
    parser.add_argument(
        "--ignore-no-results",
        dest="ignore_no_results",
        action="store_true",
        help="Do not raise an exception when a system has no results to plot; "
        "warn and skip it instead.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Global title (suptitle) added at the top of each figure.",
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
        type=str,
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
        "--hide_details",
        default=False,
        action="store_true",
        help="If set, hide detailed sub-benchmark plots and show only averages/aggregates.",
    )
    parser.add_argument(
        "--hide_average",
        default=False,
        action="store_true",
        help="If set, hide the average/aggregate subplot and show only individual benchmarks.",
    )
    parser.add_argument(
        "--check_aggregation",
        default=False,
        action="store_true",
        help="If set, check that the aggregated benchmarks are the same for all the models (--group agg).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--separate_legend",
        default=False,
        action="store_true",
        help="If set, save the legend as a separate figure instead of a subplot.",
    )
    parser.add_argument(
        "--rows_cols",
        type=str,
        default=None,
        help="Enforce subplot grid as ROWSxCOLS (e.g. 3x4). Fails if not enough subplots.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        help=(
            "Choose consecutive colors/linestyles per system (one curve/bar per "
            "system, in input order). Each system is a single-char color "
            "('bgrcmykw') optionally followed by a linestyle ('-', '--', '-.', "
            "':'); a missing linestyle defaults to solid. Systems are "
            "concatenated with no separator. Example: 'gg--g:bb--b:' (== "
            "'g-g--g:b-b--b:') means green, green dashed, green dotted, blue, "
            "blue dashed, blue dotted. Cycles if fewer entries than systems."
        ),
    )
    parser.add_argument(
        "--print_numbers",
        action="store_true",
        help="Print the performance value as text above each bar / data point.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        nargs="+",
        default=None,
        help="Maximum y-axis value for the individual benchmark plots (not the "
        "'Overall Performance' one). A single value applies to all plots; a list "
        "applies one value per plot, the last value being repeated if there are "
        "fewer values than plots.",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="If set, save the processed results to a CSV file instead of plotting.",
    )

    args = parser.parse_args()

    # Validate group specs early (resolve_group raises on unknown group / bad regex)
    for g in args.group:
        if g not in ("all", "agg"):
            resolve_group(g)

    if args.rows_cols is not None:
        parts = args.rows_cols.split("x")
        assert (
            len(parts) == 2
        ), f"--rows_cols must be in ROWSxCOLS format (e.g. 3x4), got '{args.rows_cols}'"
        args.rows_cols = (int(parts[0]), int(parts[1]))

    df = process_experiments(args)
    print(df)

    if args.save_csv:
        if args.group != ["all"]:
            list_of_tasks_to_plot = [
                task
                for g in args.group
                if g not in ("all", "agg")
                for task in resolve_group(g)
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
        plot_experiments(df, args, max_subplot=20)
