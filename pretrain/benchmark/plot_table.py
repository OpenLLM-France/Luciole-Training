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
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # print dataframe
    print(df)

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
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder"
    )  # Folder containing benchmark logs or JSON files
    parser.add_argument(
        "--sort_column", default="estimated_gpu_hours"
    )  # Column to sort the output by
    parser.add_argument(
        "--ascending", action="store_true"
    )  # Sort order (default: descending)
    args = parser.parse_args()

    input_folder = args.input_folder

    # --- Load and preprocess data ---
    data = load_data(
        input_folder
    )  # Custom function: loads JSON/dict data from the folder
    # Remove large nested "log" fields to simplify normalization
    data = [{k: v for k, v in d.items() if k != "log"} for d in data]

    # Flatten nested dictionaries into a DataFrame (e.g., {"trainer": {"devices": 8}} → "trainer.devices")
    df = pd.json_normalize(data, sep=".")

    # --- Compute derived metrics ---
    # Tokens processed per batch = batch size × sequence length
    df["data.tokens_per_batch"] = df["data.global_batch_size"] * df["data.seq_length"]

    # Estimate per-step GPU time in hours per 1T tokens
    df["estimated_time"] = (
        df["step_timings_mean"] / df["data.tokens_per_batch"] * 1e12 / 3600
    )

    # Total estimated GPU hours for 1T tokens across all nodes/devices
    df["estimated_gpu_hours"] = (
        df["estimated_time"] * df["trainer.num_nodes"] * df["trainer.devices"]
    )

    # Actual total GPU hours consumed by the job
    df["job_gpu_hours"] = (
        df["total_time"] * df["trainer.num_nodes"] * df["trainer.devices"] / 3600
    )

    # --- Remove uninformative or redundant columns ---
    # Keep columns that vary across runs OR are explicitly useful for reporting
    df = df.loc[
        :,
        df.apply(
            lambda col: (col.dropna().map(str).nunique() > 1)
            or (
                col.name
                in [
                    "start_time",
                    "job_id",
                    "min_iteration",
                    "max_iteration",
                    "args.arch",
                    "estimated_time",
                    "estimated_gpu_hours",
                    "job_gpu_hours",
                ]
            )
        ),
    ]

    # Drop columns known to be irrelevant or noisy
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

    # Simplify column names (keep only last component, e.g., "trainer.devices" → "devices")
    df.columns = [col.rsplit(".", 1)[-1] for col in df.columns]

    # Reorder columns for readability: metadata first, then computed values
    cols = ["start_time", "job_id", "job_gpu_hours", "arch"] + [
        c
        for c in df.columns
        if c not in ["start_time", "job_id", "arch", "job_gpu_hours"]
    ]
    df = df[cols]

    # --- Sort the table and export ---
    df = df.sort_values(args.sort_column, ascending=args.ascending)

    # Export the final benchmark table to an image
    df_to_png_adjusted(df, os.path.join(input_folder, "benchmark_table.png"))
