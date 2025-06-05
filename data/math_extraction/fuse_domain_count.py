import argparse
import os
import json
from collections import Counter

parser = argparse.ArgumentParser(description="Stats")
parser.add_argument("--get_science", action="store_true", help="")
args = parser.parse_args()

input_dir = os.path.join(
    os.getenv("OpenLLM_OUTPUT"),
    f"data/raw_data/math_extraction/fqdn_counts{'_science' if args.get_science else ''}",
)
output_file = "merged.json"  # output file path

merged_counter = Counter()

for filename in os.listdir(input_dir):
    if filename.endswith(".json") and filename != output_file:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Assuming data is dict-like to feed into Counter
            merged_counter.update(data)

# Sort by key (alphabetically)
sorted_counter = dict(
    sorted(merged_counter.items(), key=lambda item: item[1], reverse=True)
)

with open(os.path.join(input_dir, output_file), "w", encoding="utf-8") as f:
    json.dump(sorted_counter, f, ensure_ascii=False, indent=2)
