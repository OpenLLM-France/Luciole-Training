from datasets import Dataset
import os
from huggingface_hub import HfApi

ROOT_DIR = os.getenv("OpenLLM_OUTPUT") + "/ruler_evaluation"
TOKENIZER_NAME = "OpenLLM-BPI/tokenizer_128k-arab-regional_v2"
BENCHMARK = "synthetic"
TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
]
MAX_SEQ_LENGTHS = [131072, 65536, 32768, 16384, 8192, 4096]

api = HfApi(token=os.getenv("HF_TOKEN"))

for TASK in TASKS:
    for MAX_SEQ_LENGTH in MAX_SEQ_LENGTHS:
        DATA_DIR = (
            f"{ROOT_DIR}/data/{TOKENIZER_NAME}/{BENCHMARK}/{MAX_SEQ_LENGTH}/{TASK}"
        )

        # Convert to parquet
        dataset = Dataset.from_json(os.path.join(DATA_DIR, "validation.jsonl"))
        dataset.to_parquet(os.path.join(DATA_DIR, "validation.parquet"))

        # Push to hub
        path_in_repo = f"{TASK}/{MAX_SEQ_LENGTH}/validation.parquet"

        api.upload_file(
            path_or_fileobj=os.path.join(DATA_DIR, "validation.parquet"),
            path_in_repo=path_in_repo,
            repo_id="OpenLLM-BPI/RULER-luciole_tokenizer_128k-arab-regional_v2",
            repo_type="dataset",
        )

    # api.delete_file(
    #     path_in_repo=path_in_repo,
    #     repo_id="OpenLLM-BPI/RULER-luciole_tokenizer_128k-arab-regional_v2",
    #     repo_type="dataset",
    # )

api.upload_file(
    path_or_fileobj="dataset_readme.md",
    path_in_repo="README.md",
    repo_id="OpenLLM-BPI/RULER-luciole_tokenizer_128k-arab-regional_v2",
    repo_type="dataset",
)
