import argparse
import os
import json
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stats")
    parser.add_argument("--input_path", type=str, help="")
    args = parser.parse_args()

    output_file = "merged.json"  # output file path
    merged_counter = Counter()

    for filename in os.listdir(args.input_path):
        if filename.endswith(".json") and filename != output_file:
            filepath = os.path.join(args.input_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Assuming data is dict-like to feed into Counter
                merged_counter.update(data)

    # Sort by key (alphabetically)
    sorted_counter = dict(
        sorted(merged_counter.items(), key=lambda item: item[1], reverse=True)
    )

    with open(os.path.join(args.input_path, output_file), "w", encoding="utf-8") as f:
        json.dump(sorted_counter, f, ensure_ascii=False, indent=2)
