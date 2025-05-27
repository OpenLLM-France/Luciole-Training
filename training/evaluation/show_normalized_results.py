import os
import argparse
from utils import task_group_mapping, get_task_info, read_experiment_results


def normalize_within_range(value, lower_bound, higher_bound):
    return (value - lower_bound) / (higher_bound - lower_bound)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="List of all the experiments you want to plot",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=list(task_group_mapping.keys()),
        default="en",
        help="List of predefined groups of tasks you want to plot. You can add groups in the mapping if you want.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path where your plot are storred",
    )

    args = parser.parse_args()
    experiment_path = args.experiment_path
    group = args.group
    output_path = args.output_path

    if output_path:
        os.makedirs(output_path, exist_ok=True)

    df = read_experiment_results(experiment_path)

    # Take last sample
    df = df[df["tokens"] == df["tokens"].max()]

    for task, metric in task_group_mapping[group]:
        task_info = get_task_info(task)
        lower_bound = task_info["random"]
        task_type = task_info["task_type"]
        higher_bound = 1.0
        raw_score = df[df["task"] == task][metric]
        assert len(raw_score) == 1
        raw_score = raw_score.iloc[0]
        if raw_score < lower_bound:
            norm_score = 0
        else:
            norm_score = (
                normalize_within_range(raw_score, lower_bound, higher_bound) * 100
            )

        print(f"{task_type} - {task} - {metric}: {norm_score:.2f}")
