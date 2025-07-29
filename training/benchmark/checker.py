import os
import re
import json
import argparse

def search_cuda_out_of_memory(log_path):
    """
    Looks for exact matches of 'CUDA out of memory'.
    """
    pattern = re.compile(r"CUDA out of memory", re.IGNORECASE)

    with open(log_path, "r") as f:
        content = f.readlines()
        return bool(pattern.search("\n".join(content)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_folder")
    parser.add_argument("job_id")
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.experiment_folder, "completed.txt")):
        job_folder = os.path.join(args.experiment_folder, "job_" + args.job_id)
        failed_out_path = os.path.join(args.experiment_folder, "failed.out")
        out_of_memory = search_cuda_out_of_memory(failed_out_path)
        if out_of_memory:
            print(f"Experiment {args.job_id}, found CUDA out of memory error.")
            with open(os.path.join(args.experiment_folder, "completed.txt"), "w") as f:
                f.write("")
            stats_filename = f"stats_{os.path.basename(args.experiment_folder)}.json"
            stats_path = os.path.join(job_folder, stats_filename)
            json_data = dict(error="OOM")
            with open(stats_path, "w") as f:
                json.dump(json_data, f, indent=2)
    else:
        print(f"Experiment {args.job_id} completed correctly.")