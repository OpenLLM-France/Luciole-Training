import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
from plot_utils import load_data
import argparse


def df_to_png_adjusted(
    df, filename="df_table.png", fontsize=10, row_height=0.5, col_width_factor=0.2
):
    """
    Save a DataFrame as a PNG with columns properly adjusted to fit content.

    Parameters:
    - df: pandas.DataFrame
    - filename: str, output PNG file
    - fontsize: int, font size
    - row_height: float, height of each row
    - col_width_factor: float, width factor per character
    """
    df = df.drop(
        columns=["step_timings_mean", "step_timings_std", "total_time"], errors="ignore"
    )
    df["estimated_time"] = df["estimated_time"].apply(lambda x: f"{5*x/24:.2f} days")
    df["estimated_gpu_hours"] = df["estimated_gpu_hours"].apply(
        lambda x: f"{5*x/1000:.2f}k"
    )
    df["job_gpu_hours"] = df["job_gpu_hours"].apply(lambda x: f"{x/1000:.2f}k")

    # Sort by date then convert to string
    df["creation_date"] = df["creation_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Compute figure width based on max length of each column

    col_widths = [
        max(df[col].astype(str).map(len).max(), len(col)) * col_width_factor
        for col in df.columns
    ]
    fig_width = sum(col_widths)
    fig_height = row_height * (len(df) + 1)  # +1 for header

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_yellow_orange_red", ["#b6fcb6", "#fff89e", "#ffc074", "#ff8080"]
    )

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Set individual column widths
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)

    # Apply coloring to "estimated_gpu_hours"
    def apply_color(table, df, column_name="estimated_gpu_hours", max_value=900):
        col_idx = df.columns.get_loc(column_name)
        for row_idx, val_str in enumerate(df[column_name]):
            gpuh = float(val_str.replace("k", ""))
            ratio = min(gpuh / max_value, 1.0)
            color = cmap(ratio)
            table[row_idx + 1, col_idx].set_facecolor(color)
        return table

    table = apply_color(table, df, column_name="estimated_gpu_hours", max_value=900)
    table = apply_color(table, df, column_name="job_gpu_hours", max_value=50)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--sort_column", default="estimated_gpu_hours")
    parser.add_argument("--ascending", action="store_true")
    args = parser.parse_args()

    input_folder = args.input_folder

    data = load_data(input_folder)
    data = [{k: v for k, v in d.items() if k != "log"} for d in data]

    df = pd.json_normalize(data, sep=".")
    df["data.tokens_per_batch"] = df["data.global_batch_size"] * df["data.seq_length"]
    df["estimated_time"] = (
        df["step_timings_mean"] / df["data.tokens_per_batch"] * 1e12 / 3600
    )
    df["estimated_gpu_hours"] = (
        df["estimated_time"] * df["trainer.num_nodes"] * df["trainer.devices"]
    )
    df["job_gpu_hours"] = (
        df["total_time"] * df["trainer.num_nodes"] * df["trainer.devices"] / 3600
    )

    # Remove columns
    df = df.loc[
        :,
        df.apply(
            lambda col: (col.dropna().map(str).nunique() > 1)
            or (
                col.name
                in [
                    "creation_date",
                    "job_id",
                    "args.arch",
                    "estimated_time",
                    "estimated_gpu_hours",
                    "job_gpu_hours",
                ]
            )
        ),
    ]
    columns_to_remove = [
        "args.name",
        "open_llm_training_version",
        "data.index_mapping_dir",
        "args.output_dir",
        "trainer.plugins.grad_reduce_in_fp32",
        "data.tokens_per_batch",
        "trainer.callbacks",
    ]

    df = df.drop(columns=columns_to_remove, errors="ignore")
    df.columns = [col.rsplit(".", 1)[-1] for col in df.columns]
    cols = ["creation_date", "job_id", "job_gpu_hours", "arch"] + [
        c
        for c in df.columns
        if c not in ["creation_date", "job_id", "arch", "job_gpu_hours"]
    ]
    df = df[cols]

    df = df.sort_values(args.sort_column, ascending=args.ascending)

    print(df)
    df_to_png_adjusted(df, os.path.join(input_folder, "benchmark_table.png"))
