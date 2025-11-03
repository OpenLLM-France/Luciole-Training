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
        tokens = float(match.group(1)) if match else None
        model_size = 8.0
    elif "EuroLLM-1.7B" in str(file_path):
        tokens = 4000
        model_size = 1.394_706_432
    elif "EuroLLM-9B" in str(file_path):
        tokens = 4000
        model_size = 9.0
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
        tokens = steps * 4096 * 1024 / 10**9
        model_size = 7.0
    elif "CroissantLLMBase" in str(file_path):
        tokens = 3000
        model_size = 1.3
    elif ("luciol" in str(file_path)) or ("llama1b" in str(file_path)):
        match = re.search(r"step_([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        if "llama1b" in str(file_path):
            model_size = 1.235290112
        elif "nemotron1b" in str(file_path):
            model_size = 1.319309312
        elif "nemotronh8b" in str(file_path):
            model_size = 8.075686912
        elif "nemotron23b" in str(file_path):
            model_size = 23.216467968
        else:
            raise ValueError(f"Unknown model size for model in: {file_path}")
        if "phase2" in str(file_path):
            steps += 715786
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
    df["max_samples"] = str(data["config_general"]["max_samples"])

    # Filter out metrics ending in "_stderr"
    df = df[~df["metric"].str.endswith("_stderr")]

    # Get training flops
    tokens, num_parameters = get_training_tokens_and_model_size(file_path)
    df["tokens"] = tokens
    df["model_size"] = num_parameters
    df["FLOPs"] = df["model_size"] * df["tokens"] * 6 * 1e18

    # Get evaluation timestamp
    df["timestamp"] = pd.to_datetime(
        file_path.stem.replace("results_", ""), format="%Y-%m-%dT%H-%M-%S.%f"
    )
    return df


def read_experiment_results(main_dir, evaluation_dir="evaluation"):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)
    expe_name = main_dir.name

    dataframes = [
        read_json_file(f)
        for f in main_dir.rglob("results_*.json")
        if evaluation_dir in f.parts
    ]
    if not dataframes:
        print(f"No valid JSON result files found in {main_dir}")
        return
    df = pd.concat(dataframes, ignore_index=True)
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
