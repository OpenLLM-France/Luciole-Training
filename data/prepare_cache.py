from datatrove.io import cached_asset_path_or_download
from pathlib import Path
from huggingface_hub import hf_hub_url

# Fasttext
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_SUBFOLDER = "ft176"

model_file = cached_asset_path_or_download(
    MODEL_URL,
    namespace="lid",
    subfolder=MODEL_SUBFOLDER,
    desc="fast-text language identifier model",
)
print(model_file)

# CCNET
MODEL_REPO = "edugp/kenlm"
# /lustre/fswork/projects/rech/fwx/commun/.cache/huggingface/assets/datatrove/default/default/
model_dataset = "wikipedia"
for model_name in ["fr"]:
    path = cached_asset_path_or_download(
        hf_hub_url(MODEL_REPO, str(Path(model_dataset, f"{model_name}.arpa.bin")))
    )
    path = cached_asset_path_or_download(
        hf_hub_url(MODEL_REPO, str(Path(model_dataset, f"{model_name}.sp.model")))
    )
