import os
import json

# Path containing the folders
BASE_PATH = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_llama1b/huggingface_checkpoints/"

for folder in os.listdir(BASE_PATH):
    config_path = os.path.join(BASE_PATH, folder, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                config["rope_scaling"]["original_max_position_embeddings"] = 4096
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {config_path}: {e}")
