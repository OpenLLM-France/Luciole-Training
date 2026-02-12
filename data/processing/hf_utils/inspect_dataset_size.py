from huggingface_hub import HfApi
from collections import defaultdict

api = HfApi()

info = api.repo_info(
    repo_id="OpenLLM-BPI/Luciole-Training-Dataset",
    repo_type="dataset",
    files_metadata=True,
)


folder_sizes = defaultdict(int)

for f in info.siblings:
    if f.size is None:
        continue

    parts = f.rfilename.split("/")
    for i in range(1, len(parts)):
        folder = "/".join(parts[:i])
        folder_sizes[folder] += f.size

for folder, size in sorted(folder_sizes.items()):
    print(f"{folder}/ : {size / (1024**3):.3f} GB")

with open("all_folder_sizes.txt", "w") as f_out:
    for folder, size in sorted(folder_sizes.items()):
        f_out.write(f"{folder}/ : {size / (1024**3):.3f} GB\n")

with open("big_folder_sizes.txt", "w") as f_out:
    for folder, size in sorted(folder_sizes.items()):
        if size > 10 * (1024**3):  # Only write folders larger than 10 GB
            f_out.write(f"{folder}/ : {size / (1024**3):.3f} GB\n")

with open("bigger_folder_sizes.txt", "w") as f_out:
    for folder, size in sorted(folder_sizes.items()):
        if size > 100 * (1024**3):  # Only write folders larger than 10 GB
            f_out.write(f"{folder}/ : {size / (1024**3):.3f} GB\n")
