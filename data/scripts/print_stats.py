import os
import sys
import matplotlib.pyplot as plt
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training statistics.")
    parser.add_argument(
        "--stats_dir",
        type=str
        )
    args = parser.parse_args()

    stats_dir = args.stats_dir

    data = json.load(open(os.path.join(stats_dir, "merged_stats.json"), "r"))
    reader = data[0]
    ndoc_reader = reader['stats']['documents']['total']
    writer = data[-1]
    ndoc_writer = writer['stats']['total']['total']

    print('Number of documents in reader:', ndoc_reader)
    print('Number of documents in writer:', ndoc_writer)