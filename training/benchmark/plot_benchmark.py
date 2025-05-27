import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mtick


def plot_training_and_gpu_hours(df, output_folder=""):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

    metrics = [
        ("training_time", "Training Time for 1T Tokens", "Training time (in days)"),
        (
            "consumed_gpu_hours",
            "Consumed GPU Hours for 1T Tokens",
            "Consumed GPU hours (in thousand)",
        ),
    ]

    for ax, (y, title, ylabel) in zip(axes, metrics):
        sns.lineplot(data=df, x="num_nodes", y=y, hue="config", marker="o", ax=ax)

        for config_name, group in df.groupby("config"):
            last_point = group.sort_values("num_nodes").iloc[-1]
            ax.text(
                last_point["num_nodes"] + 0.5,
                last_point[y],
                f"{last_point[y]:.2f}"
                if last_point[y] < 1000
                else f"{last_point[y]/1000:.0f}k",
                fontsize=10,
                va="center",
                ha="left",
            )

        ax.set_title(title)
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if y == "consumed_gpu_hours":
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:g}"
                )
            )

    plt.tight_layout()
    output_path = (
        f"{output_folder}/training_and_gpu_hours.png"
        if output_folder
        else "training_and_gpu_hours.png"
    )
    plt.savefig(output_path)
    plt.close()


def plot_curves(
    y="mean_step_timing", title=None, ylabel=None, ylim=None, output_folder=""
):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="num_nodes", y=y, hue="config", marker="o")
    for config_name, group in df.groupby("config"):
        last_point = group.sort_values("num_nodes").iloc[-1]
        plt.text(
            last_point["num_nodes"] + 0.5,
            last_point[y],
            f"{last_point[y]:.2f}"
            if last_point[y] < 1000
            else f"{last_point[y]/1000:.0f}k",
            fontsize=12,
            va="center",
            ha="left",
        )
    plt.title(title)
    plt.xlabel("Number of Nodes")
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:g}")
    )
    if ylim:
        plt.ylim(ylim)
    plt.grid(True)
    plt.tight_layout()
    output_path = (
        f"{output_folder}/{title.lower().replace(' ', '_')}.png"
        if output_folder
        else f"{title.lower().replace(' ', '_')}.png"
    )
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--output_folder", default="")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    folders = os.listdir(input_folder)

    data = []

    for folder in folders:
        xp_folder = os.path.join(input_folder, folder)
        if not os.path.isfile(xp_folder):
            files = os.listdir(xp_folder)
            stat_files = [
                f for f in files if f.startswith("stats_") and f.endswith(".json")
            ]
            if len(stat_files) > 0:
                stat_file = os.path.join(xp_folder, stat_files[0])
                with open(stat_file, "r") as f:
                    json_data = json.load(f)
                    data.append(json_data)

    records = []
    for entry in data:
        number_of_steps_per_trillion_tokens = 1e12 / (
            entry["batch_size"] * entry["seq_length"]
        )
        config_key = f"{entry['arch']}_tp{entry['tensor_parallelism']}_pp{entry['pipeline_parallelism']}_fp8{entry['fp8']}_seq_length{entry['seq_length']}"
        if "context_parallelism" in entry:
            config_key += f"_cp{entry['context_parallelism']}"
        else:
            config_key += "_cp1"
        records.append(
            {
                "num_nodes": entry["num_nodes"],
                "mean_step_timing": entry["mean_step_timings"],
                "training_time": entry["mean_step_timings"]
                * number_of_steps_per_trillion_tokens
                / (3600 * 24),
                "consumed_gpu_hours": (
                    entry["mean_step_timings"]
                    * number_of_steps_per_trillion_tokens
                    * entry["num_nodes"]
                    * 4
                    / 3600
                ),
                "config": config_key,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values(by="config")

    plot_training_and_gpu_hours(df, output_folder=output_folder)
