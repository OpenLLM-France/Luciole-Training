from pathlib import Path
import json
import pandas as pd
import re
import numpy as np


def get_training_tokens_and_model_size(file_path):
    if "OLMo-2-0425-1B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 1.279_395_840
    elif "OLMo-2-1124-7B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 7.0
    elif "OLMo-2-1124-13B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 13.0
    elif "OLMo-2-0325-32B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 32.234_279_936
    elif "Apertus-8B-2509" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 15000
        model_size = 8.0
    elif "Gaperon-1125-1B" in str(file_path):
        tokens = 3000
        model_size = 1.0
    elif "Gaperon-1125-8B" in str(file_path):
        match = re.search(r"_tokens-([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 4110
        model_size = 8.0
    elif "Gaperon-1125-24B" in str(file_path):
        match = re.search(r"_tokens-([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 2059
        model_size = 24.0
    elif "EuroLLM-1.7B" in str(file_path):
        tokens = 4000
        model_size = 1.394_706_432
    elif "EuroLLM-9B" in str(file_path):
        tokens = 4000
        model_size = 9.0
    elif "EuroLLM-22B" in str(file_path):
        tokens = 4000
        model_size = 22.0
    elif "salamandra-7b" in str(file_path):
        tokens = 12_875
        model_size = 7.768_117_248
    elif "Teuken-7B" in str(file_path):
        tokens = 6_000
        model_size = 7.0
    elif "SmolLM2-1.7B" in str(file_path):
        match = re.search(r"step-([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        if steps is None:
            tokens = 11000
        else:
            tokens = steps * 2 * 1e-3
        model_size = 1.711_376_384
    elif "SmolLM3-3B" in str(file_path):
        tokens = 11200
        model_size = 3.075_098_624
    elif "Lucie-7B" in str(file_path):
        match = re.search(r"step([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        if steps:
            if "extension" in str(file_path):
                steps += 753851
            tokens = steps * 4096 * 1024 / 10**9
        else:
            tokens = 3131.7
        model_size = 7.0
    elif "CroissantLLMBase" in str(file_path):
        tokens = 3000
        model_size = 1.3
    elif "Llama-2-7b" in str(file_path):
        model_size = 6.9
        tokens = 2000
    elif "Llama-3.2-1B" in str(file_path):
        model_size = 1.23
        tokens = 9000
    elif "Llama-3.1-8B" in str(file_path):
        model_size = 8.0
        tokens = 15000
    elif "Mistral-Small" in str(file_path) and "24B" in str(file_path):
        model_size = 24.0
        tokens = 8000
    elif "Mistral-7B" in str(file_path):
        model_size = 7
        tokens = 8000
    elif "Ministral-3" in str(file_path):
        match = re.search(r"Ministral-3-([0-9.]+)B", str(file_path))
        model_size = int(match.group(1))
        tokens = 8000
    elif "Qwen3-" in str(file_path):
        match = re.search(r"Qwen3-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 36000
    elif "Qwen2.5-" in str(file_path):
        match = re.search(r"Qwen2.5-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 18000
    elif "Qwen2-" in str(file_path):
        match = re.search(r"Qwen2-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 7000
    elif (
        ("luciol" in str(file_path).lower())
        or ("llama1b" in str(file_path).lower())
        or ("ablation" in str(file_path).lower())
    ):
        if "llama1b" in str(file_path).lower():
            model_size = 1.235290112
        elif "1b" in str(file_path).lower():
            model_size = 1.319309312
        elif "8b" in str(file_path).lower():
            model_size = 8.075686912
        elif "23b" in str(file_path).lower():
            model_size = 23.216467968
        else:
            raise ValueError(f"Unknown model size for model in: {file_path}")

        match = re.search(r"totalstep([0-9.]+)", str(file_path))
        if match:
            steps = float(match.group(1))
        else:
            match = re.search(r"step_([0-9.]+)", str(file_path))
            steps = float(match.group(1)) if match else None
            if steps is None:
                raise ValueError(f"Could not extract steps from file path: {file_path}")

            steps_phase1 = 715787
            steps_phase2 = 358930
            steps_phase3_annealing = 118238 if model_size < 23 else 71526
            steps_extension = 5960 if model_size < 23 else 11920

            if "phase2" in str(file_path):
                steps += steps_phase1
            if "_32k_" in str(file_path) or "_131k_v4_" in str(file_path):
                steps += steps_phase1 + steps_phase2 + steps_phase3_annealing
            elif "_65k_" in str(file_path) or "_131k_" in str(file_path):
                steps += (
                    steps_phase1
                    + steps_phase2
                    + steps_phase3_annealing
                    + steps_extension
                )
            elif "annealin" in str(file_path):
                steps += steps_phase1 + steps_phase2

        tokens = steps * 4096 * 1024 / 10**9
    else:
        raise ValueError(f"Unknown model in file path: {file_path}")
    return tokens, model_size


def read_json_file(file_path):
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)

    df = (
        pd.DataFrame(data["results"])
        .stack()
        .reset_index(name="score")
        .rename(columns={"level_0": "metric", "level_1": "task"})
    )

    # Multiply the values of the "score" column by 1/100 when the value of the "metric" column is "comet" or "comet_stderr"
    df.loc[df["metric"].isin(["comet", "comet_stderr"]), "score"] *= 1 / 100

    df["max_samples"] = str(data["config_general"]["max_samples"])

    # Filter out metrics ending in "_stderr"
    # df = df[~df["metric"].str.endswith("_stderr")]

    # mark whether the row is stderr or score
    df["value_type"] = df["metric"].str.endswith("_stderr")

    # normalize metric name (remove _stderr suffix)
    df["metric_base"] = df["metric"].str.replace("_stderr$", "", regex=True)

    # pivot score vs stderr into columns
    df = (
        df.pivot_table(
            index=["metric_base", "task", "max_samples"],
            columns="value_type",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={False: "score", True: "stderr", "metric_base": "metric"})
    )

    # Get training flops
    tokens, num_parameters = get_training_tokens_and_model_size(file_path)
    if not tokens:
        print(f"WARNING: Could not determine training tokens for {file_path}")
    df["tokens"] = tokens
    df["model_size"] = num_parameters
    df["FLOPs"] = df["model_size"] * df["tokens"] * 6 * 1e18

    # Get evaluation timestamp
    df["timestamp"] = pd.to_datetime(
        file_path.stem.replace("results_", ""), format="%Y-%m-%dT%H-%M-%S.%f"
    )
    return df


def read_experiment_results(
    main_dir, evaluation_dir="evaluation", expe_name=None, split_per_tokens=False
):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)

    assert main_dir.is_dir(), f"{main_dir} is not an existing directory"

    if expe_name is None:
        expe_name = main_dir.name

    dataframes = [
        read_json_file(f)
        for f in main_dir.rglob("results_*.json")
        if evaluation_dir in f.parts and "deprecated" not in str(f)
    ]
    if not dataframes:
        print(f"No valid JSON result files found in {main_dir}")
        return
    df = pd.concat(dataframes, ignore_index=True)
    if split_per_tokens:
        df["expe_name"] = df["tokens"].apply(
            lambda t: f"{expe_name} ({int(t)}B training tokens)"
        )
    else:
        df["expe_name"] = expe_name

    # Remove duplicates
    len_before_dup = len(df)
    df = df.sort_values("timestamp", ascending=False).drop_duplicates(
        subset=df.columns.difference(["timestamp"]), keep="first"
    )
    len_after_dup = len(df)
    if len_before_dup > len_after_dup:
        print(f"Removed {len_before_dup - len_after_dup} duplicate rows")

    print("Example:")
    print(df.iloc[0])
    print("\n")
    return df


def read_datamix(main_dir):
    json_files = list(Path(main_dir).glob("datamix/*.json"))
    if not json_files:
        print(f"No JSON datamix file found in {main_dir}")
        return
    if len(json_files) > 1:
        print(f"More than one JSON datamix file found in {main_dir}")
        return
    json_file = json_files[0]
    with open(json_file, "r") as f:
        data = json.load(f)
    datamix = data["train"]
    return datamix


def compute_regression(group):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    group = group[group["tokens"] > 0]  # adjust to your x column name

    group = group.sort_values("tokens")
    x = group["tokens"].values.reshape(-1, 1)
    y = group["score"].values

    tokens = x.flatten().tolist()
    score = y.tolist()

    if len(group) < 2:
        return pd.DataFrame(
            [{"slope": np.nan, "intercept": np.nan, "tokens": tokens, "score": score}]
        )

    model = LinearRegression()
    model.fit(np.log(x), y)
    y_pred = model.predict(np.log(x))

    return pd.DataFrame(
        [
            {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r2": r2_score(y, y_pred),
                "tokens": tokens,
                "score": score,
            }
        ]
    )


def moving_average(values, window=5):
    values = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")  # only valid positions
    return smoothed


def process_group(group, window=1):
    group = group.loc[group["tokens"] > 0].sort_values("tokens")

    tokens = group["tokens"].to_numpy()
    flops = group["FLOPs"].to_numpy()
    scores = group["score"].to_numpy()
    stderr = group["stderr"].to_numpy()

    if window < 2 or len(scores) < window:
        scores = scores
    else:
        scores = moving_average(scores, window=window)
        pad = window // 2
        tokens = tokens[pad : -pad or None]
        flops = flops[pad : -pad or None]

    return pd.DataFrame(
        [
            {
                "tokens": tokens.tolist(),
                "FLOPs": flops.tolist(),
                "score": scores.tolist(),
                "stderr": stderr.tolist(),
            }
        ]
    )


def process_results(df, window=1, fit=False):
    if fit:
        group_df = (
            df.groupby(["task", "max_samples", "metric", "expe_name"])
            .apply(compute_regression)
            .reset_index()
        )
        return group_df
    else:
        group_df = (
            df.groupby(["task", "max_samples", "metric", "expe_name"])
            .apply(lambda x: process_group(x, window=window), include_groups=False)
            .reset_index()
        )
        return group_df
