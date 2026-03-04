from utils import read_experiment_results
import io
import pandas as pd
import argparse
import os


def read_info():
    folder = os.path.dirname(__file__)
    with open(os.path.join(folder, "agg_tasks.jsonl")) as f:
        content = "\n".join(line for line in f if not line.lstrip().startswith("//"))
    df_info = pd.read_json(io.StringIO(content), lines=True)
    df_info["random"] = (1.0 / df_info["num_classes"]).fillna(0.0)
    return df_info


_df_info = None


def get_info():
    global _df_info
    if _df_info is None:
        _df_info = read_info()
    return _df_info


def normalize_within_range(value, lower_bound, higher_bound):
    assert (
        lower_bound < higher_bound
        and value <= higher_bound
        and value >= min(0, lower_bound)
    ), f"Value {value} is out of bounds [{lower_bound}, {higher_bound}]"
    if value < lower_bound:
        return 0
    else:
        return (value - lower_bound) / (higher_bound - lower_bound) * 100


def check_benchmarks_by_tasktype(
    benchmarks_per_tasktype, second_dict, name1, name2, raise_if_fail=True
):
    missings = []
    extras = []
    for key, expected in benchmarks_per_tasktype.items():
        task_type, language = key
        actual = second_dict.get(key, [])
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        if missing:
            missings.append(
                f"* Missing benchmarks for ({task_type}, {language}): {sorted(missing)}"
            )
        if extra:
            extras.append(
                f"* New unexpected benchmarks for ({task_type}, {language}): {sorted(extra)}"
            )
    if missings or extras:
        error_msg = (
            f"When comparing benchmarks by task type between {name1} (reference) and {name2} (new), found the following discrepancies:\n"
            + "\n".join(missings + extras)
        )
        if raise_if_fail:
            raise ValueError(error_msg)
        else:
            print("WARNING: " + error_msg)
    return missings, extras


def calculate_agg_score(df, check_aggregation=False):
    df_info = get_info()

    # Create a mapping from base task to full task (with fewshot)
    df["task_base"] = df["task"].str.rsplit("|", n=1).str[0]
    task_fewshot_mapping = df[["task_base", "task"]].drop_duplicates()
    df_info_with_fewshot = (
        df_info.merge(
            task_fewshot_mapping, left_on="task", right_on="task_base", how="left"
        )
        .drop(columns=["task_x", "task_base"])
        .rename(columns={"task_y": "task"})
    )

    df_no_max_samples = df.copy()
    df_no_max_samples["max_samples"] = "None"
    # Sort by increasing tokens
    df_no_max_samples = df_no_max_samples.sort_values(by=["tokens"], ascending=True)

    all_results = []  # List to collect DataFrames
    benchmarks_per_tasktype_ref = None
    for (expe_name, tokens, FLOPs, max_samples), df_group in df_no_max_samples.groupby(
        ["expe_name", "tokens", "FLOPs", "max_samples"]
    ):
        df_group = df_info_with_fewshot.merge(
            df_group, on=["task", "metric"], how="left"
        )
        df_group = df_group.dropna(subset=["score"])

        # List each benchmark per task_type and languages
        benchmarks_per_tasktype = (
            df_group.groupby(["task_type", "language"])["task"]
            .apply(list)
            .apply(sorted)
            .to_dict()
        )
        if benchmarks_per_tasktype_ref is None:
            benchmarks_per_tasktype_ref = benchmarks_per_tasktype
            ref_name = f"experiment {expe_name} with tokens={tokens}"
        else:
            missing, extras = check_benchmarks_by_tasktype(
                benchmarks_per_tasktype_ref,
                benchmarks_per_tasktype,
                ref_name,
                f"experiment {expe_name} with tokens={tokens}",
                raise_if_fail=check_aggregation,
            )
            if not missing and extras:
                # Restart from here
                benchmarks_per_tasktype_ref = benchmarks_per_tasktype
                ref_name = f"experiment {expe_name} with tokens={tokens}"
                all_results = []

        df_group["norm_score"] = df_group.apply(
            lambda x: normalize_within_range(x["score"], x["random"], 1.0), axis=1
        )

        # Group by task type and language
        results_task = (
            df_group.groupby(["language", "task_type"])
            .agg({"norm_score": lambda x: x.mean(skipna=False)})
            .reset_index()
        )
        # Group by language
        # (ignoring if there is only one task for the language)
        results_final = (
            results_task.groupby("language")
            .agg(
                {
                    "norm_score": lambda x: x.mean(skipna=False)
                    if len(x) > 1
                    else float("nan")
                }
            )
            .reset_index()
        )

        # Reformat
        results_task["expe_name"] = expe_name
        results_task["tokens"] = tokens
        results_task["FLOPs"] = FLOPs
        results_task["max_samples"] = max_samples
        results_task["metric"] = "agg"
        results_task["task"] = (
            results_task["task_type"]  # .str.upper()
            + " ("
            + results_task["language"]  # .str.upper()
            + ")"
        )
        results_task = results_task.rename(columns={"norm_score": "score"})
        all_results.append(results_task)

        results_final["expe_name"] = expe_name
        results_final["tokens"] = tokens
        results_final["FLOPs"] = FLOPs
        results_final["max_samples"] = max_samples
        results_final["metric"] = "agg"
        results_final["task"] = (
            "All together (" + results_final["language"] + ")"
        )  # .str.upper()
        results_final = results_final.rename(columns={"norm_score": "score"})
        all_results.append(results_final)

    if len(all_results) == 0:
        print("No results found for the given experiments.")
        return pd.DataFrame(
            columns=[
                "expe_name",
                "tokens",
                "FLOPs",
                "task",
                "max_samples",
                "metric",
                "score",
            ]
        )

    df = pd.concat(all_results, ignore_index=True)
    return benchmarks_per_tasktype_ref, df[
        ["expe_name", "tokens", "FLOPs", "task", "max_samples", "metric", "score"]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        nargs="+",
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )
    args = parser.parse_args()

    df = pd.concat([read_experiment_results(path) for path in args.experiment_path])
    df_agg = calculate_agg_score(df)
    print(df_agg)
