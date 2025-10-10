import pandas as pd

df = pd.read_csv("chronicles/all_stats_merged.csv").sort_values(["name"])[["name"]]
df["repeat"] = 0

df.to_csv("repeats.csv", index=False)
