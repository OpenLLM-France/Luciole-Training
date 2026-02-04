import os
import re
import shutil
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Remove folders where (step + 1) % 1000 != 0"
)
parser.add_argument(
    "parent_dir",
    type=str,
    help="Path to the parent directory containing folders to check",
)
parser.add_argument(
    "--multiple_of",
    type=int,
    default=1000,
)
args = parser.parse_args()
parent_dir = args.parent_dir

# Regex to extract step number
pattern = re.compile(r"step[=_](\d+)")

for folder_name in sorted(os.listdir(parent_dir)):
    folder_path = os.path.join(parent_dir, folder_name)

    if not os.path.isdir(folder_path):
        continue

    match = pattern.search(folder_name)
    if match:
        step = int(match.group(1))
        if step > 0 and ((step + 1) % args.multiple_of != 0):
            # Prompt user for confirmation
            answer = input(f"Remove folder '{folder_name}'? [y/N]: ").strip().lower()
            if answer == "y":
                shutil.rmtree(folder_path)
                print(f"Removed {folder_name}")
            else:
                print(f"Skipped {folder_name}")
