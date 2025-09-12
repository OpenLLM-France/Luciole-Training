from pathlib import Path
import json
import pandas as pd
import re
import numpy as np


def read_json_file(file_path, seq_length=2048):
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)

    df = (
        pd.DataFrame(data["results"])
        .stack()
        .reset_index(name="score")
        .rename(columns={"level_0": "metric", "level_1": "task"})
    )

    # Filter out metrics ending in "_stderr"
    df = df[~df["metric"].str.endswith("_stderr")]

    # Add model name and timestamp
    df["model_name"] = data["config_general"]["model_name"]
    df["max_samples"] = str(data["config_general"]["max_samples"])

    if "OLMo-2-0425-1B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        df["tokens"] = tokens
        df["num_parameters"] = 1.279_395_840
    elif "OLMo-2-1124-7B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        df["tokens"] = tokens
        df["num_parameters"] = 7.0
    elif "EuroLLM-1.7B" in str(file_path):
        df["tokens"] = 4000
        df["num_parameters"] = 1.394_706_432
    elif "SmolLM2-1.7B" in str(file_path):
        df["tokens"] = 11000
        df["num_parameters"] = 1.711_376_384
    elif "SmolLM3-3B" in str(file_path):
        df["tokens"] = 11200
        df["num_parameters"] = 3.075_098_624
    elif "Lucie-7B" in str(file_path):
        match = re.search(r"step([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        df["tokens"] = steps * 4096 * 1024 / 10**9
        df["num_parameters"] = 7.0
    elif "luciole_llama1b" in str(file_path):
        match = re.search(r"step_([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        df["tokens"] = steps * 4096 * 1024 / 10**9
        df["num_parameters"] = 1.2
    elif "CroissantLLMBase" in str(file_path):
        df["tokens"] = 3000
        df["num_parameters"] = 1.3
    else:
        raise ValueError(f"Unknown model in file path: {file_path}")

    df["FLOPs"] = df["num_parameters"] * df["tokens"] * 6 * 1e18
    match = re.match(r"results_(.*)\.json", file_path.name)
    timestamp = match.group(1) if match else None
    df["timestamp"] = timestamp

    # Reorder columns
    df = df[["tokens", "FLOPs", "timestamp", "task", "max_samples", "metric", "score"]]
    return df


def read_experiment_results(main_dir, seq_length=2048):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)

    json_files = list(main_dir.rglob("results_*.json"))
    if not json_files:
        print(f"No JSON result files found in {main_dir}")
        return

    # Read files and store relative path + DataFrame
    dataframes = []
    for file in json_files:
        file_path = Path(file)
        parts = file_path.parts
        try:
            eval_index = parts.index("evaluation")
            expe_name = parts[
                eval_index - 1
            ]  # Name of the folder just before 'evaluation'
        except ValueError:
            expe_name = "unknown"

        df = read_json_file(file, seq_length=seq_length)
        df["expe_name"] = expe_name
        dataframes.append(df)

    # Concatenate all DataFrames
    df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H-%M-%S.%f")

    # Keep the most recent row for each (model_name, tokens, task, metric) tuple
    df_latest = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["expe_name", "tokens", "FLOPs", "task", "max_samples", "metric"],
            keep="last",
        )
        .drop("timestamp", axis=1)
    )
    # print(df_latest)
    return df_latest[
        ["expe_name", "tokens", "FLOPs", "task", "max_samples", "metric", "score"]
    ]


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


def process_group(group):
    group = group.loc[group["tokens"] > 0].sort_values("tokens")
    # print()
    # print(group)
    return pd.DataFrame(
        [
            {
                "tokens": group["tokens"].tolist(),
                "FLOPs": group["FLOPs"].tolist(),
                "score": group["score"].tolist(),
            }
        ]
    )


def process_results(df):
    group_df = (
        df.groupby(["task", "max_samples", "metric", "expe_name"])
        .apply(process_group, include_groups=False)
        .reset_index()
    )
    return group_df
