import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from plot_utils import setup_data
import argparse


def create_config_name(entry):
    parts = []

    if "arch" in entry:
        parts.append(f"{entry['arch']}")

    if "tp" in entry:
        parts.append(f"tp{int(entry['tp'])}")

    if "pp" in entry:
        parts.append(f"pp{int(entry['pp'])}")

    if "cp" in entry:
        parts.append(f"cp{int(entry['cp'])}")

    if "precision" in entry:
        parts.append(entry["precision"])

    if "seq_length" in entry:
        parts.append(f"seq_length{int(entry['seq_length'])}")

    if "batch_size" in entry:
        parts.append(f"batch_size{int(entry['batch_size'])}")

    if entry.get("sequence_parallel", False):
        parts.append("sequence_parallel")

    return "_".join(parts)


def plot_training_and_gpu_hours(
    df, plot_name="plot.png", plot_title="", output_folder="", only_last_text=True
):
    # Find columns with only one unique value
    if len(df) == 0:
        print(f"Dataframe empty for {os.path.join(output_folder, plot_name)} ")
        return

    constant_columns = df.loc[:, df.nunique() == 1]

    # Log the dropped columns and their constant values
    dropped_info = {
        col: constant_columns[col].iloc[0] for col in constant_columns.columns
    }
    dropped_info.pop("num_nodes", None)
    dropped_info.pop("num_gpus", None)

    # Drop those columns from the DataFrame
    df = df.drop(columns=constant_columns.columns)

    # Create config names
    df["config"] = df.apply(create_config_name, axis=1)

    # Check for duplicate (config, num_nodes) pairs
    duplicates = df[df.duplicated(subset=["config", "num_nodes"], keep=False)]
    if not duplicates.empty:
        print("\n⚠️  Warning: The following config values are duplicated:")
        print(duplicates[["config", "num_nodes"]])
        print(df)
    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    metrics = [
        ("training_time", "Training Time for 1T Tokens", "Training time (in days)"),
        (
            "consumed_gpu_hours",
            "Consumed GPU Hours for 1T Tokens",
            "Consumed GPU hours (in thousand)",
        ),
    ]

    handles = []
    labels = []

    for ax, (y, title, ylabel) in zip(axes, metrics):
        sns.lineplot(data=df, x="num_gpus", y=y, hue="config", marker="o", ax=ax)

        for config_name, group in df.groupby("config"):
            if only_last_text:
                last_point = group.sort_values("num_gpus").iloc[-1]
                ax.text(
                    last_point["num_gpus"] + 3,
                    last_point[y],
                    f"{last_point[y]:.2f}"
                    if last_point[y] < 1000
                    else f"{last_point[y]/1000:.0f}k",
                    fontsize=10,
                    va="center",
                    ha="left",
                )
            else:
                last_points = group.sort_values("num_gpus")
                for _, point in last_points.iterrows():
                    ax.text(
                        point["num_gpus"] + 5,
                        point[y],
                        f"{point[y]:.2f}"
                        if point[y] < 1000
                        else f"{point[y]/1000:.0f}k",
                        fontsize=10,
                        va="center",
                        ha="left",
                    )

        ax.set_title(title)
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0)
        ax.grid(True)

        if y == "consumed_gpu_hours":
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:g}"
                )
            )

        if not handles:
            handles, labels = ax.get_legend_handles_labels()

        ax.get_legend().remove()

    # Add shared legend
    fig.legend(
        handles,
        labels,
        title="",
        loc="upper left",
        bbox_to_anchor=(0.92, 0.6),
        borderaxespad=0.0,
        fontsize="medium",
        title_fontsize="large",
    )

    # Add dropped info as annotation near the legend
    if dropped_info:
        info_str = "\n".join(f"- {k}: {v}" for k, v in dropped_info.items())
        legend_lines = len(labels)
        legend_height = 0.033 * legend_lines  # adjust this scaling if needed
        annotation_y = max(0.6 - legend_height - 0.05, 0.05)
        fig.text(
            0.92,
            annotation_y,
            f"Constant params:\n{info_str}",
            ha="left",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )

    # Final layout & save
    fig.suptitle(plot_title, fontsize=16)
    if plot_name:
        output_path = f"{output_folder}/{plot_name}" if output_folder else plot_name
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--output_folder", default="plots")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    df = setup_data(input_folder)
    df = df.sort_values(by=["arch", "seq_length", "batch_size", "precision"])
    plot_training_and_gpu_hours(
        df,
        plot_name="all.png",
        plot_title="",
        output_folder=output_folder,
        only_last_text=False,
    )
