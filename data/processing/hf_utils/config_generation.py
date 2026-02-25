from huggingface_hub import HfApi
import re

api = HfApi()

info = api.repo_info(
    repo_id="OpenLLM-BPI/Luciole-Training-Dataset",
    repo_type="dataset",
    files_metadata=True,
)

TEMPLATE = """\
  - config_name: {name}
    data_files:  
      - path: {path}
        split: train
"""

OUTPUT_FILE = "dataset_configs.md"


def snake_to_pascal(name: str) -> str:
    SPECIAL = {"hq", "insee", "hal", "hplt2"}
    slash_parts = name.split("/")

    def convert(part: str) -> str:
        tokens = re.split(r"[_-]+", part)
        return "".join(
            token.upper() if token.lower() in SPECIAL else token.capitalize()
            for token in tokens
            if token
        )

    return "/".join(convert(p) for p in slash_parts)


def create_config(path):
    parts = path.split("/")
    if len(parts) == 1:
        return TEMPLATE.format(
            name=snake_to_pascal(path), path=f"data/{path}/**/*.parquet"
        )
    elif path.startswith(("fineweb2", "culturax", "hplt2")):
        if path.endswith("score"):
            path = path.removesuffix("/score")
            paths = "\n        - " + "\n        - ".join(
                [f"data/{path}/score_{i}/*.parquet" for i in range(4, -1, -1)]
            )
            return TEMPLATE.format(
                name=snake_to_pascal(parts[0]) + "-" + parts[1], path=paths
            )
        else:
            return TEMPLATE.format(
                name=snake_to_pascal(parts[0]) + "-" + parts[1],
                path=f"data/{path}/**/*.parquet",
            )
    elif len(parts) == 2:
        return TEMPLATE.format(
            name=snake_to_pascal(parts[0]) + "-" + parts[1],
            path=f"data/{path}/**/*.parquet",
        )
    else:
        raise NotImplementedError


folders = []
configs = []

# Main folder configs
for f in info.siblings:
    filename = f.rfilename
    if not filename.startswith("data/"):
        continue
    parts = filename.split("/")[1:-1]

    if filename.startswith(("data/fineweb2", "data/culturax", "data/hplt2")):
        if "score" in parts[-1]:
            parts = parts[:-1] + ["score"]
    elif filename.startswith("data/nemotron_postraining"):
        parts = parts[0:2]
    else:
        parts = [parts[0]]

    folder = "/".join(parts)
    if folder and folder not in folders:
        folders.append(folder)
        configs.append(create_config(folder))

# Language configs
languages = []
for f in info.siblings:
    filename = f.rfilename
    if not filename.startswith("data/"):
        continue
    if "starcoder" in filename or "score_" in filename:
        continue
    language = filename.split("/")[-2]
    if language and language not in languages:
        languages.append(language)


def sort_without_dash_first(strings):
    return sorted(strings, key=lambda s: ("-" in s, s))


for language in sort_without_dash_first(languages):
    configs.append(
        TEMPLATE.format(name=language, path=f"data/**/{language}/**/*.parquet")
    )

header = """---
license: cc-by-sa-4.0
configs:
  - config_name: default
    data_files:
      - path: data/**/*.parquet
        split: train
"""

# Write all configs to a markdown file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(header)
    for config in configs:
        f.write(config)
    f.write("---")
