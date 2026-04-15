import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd
import hashlib

def count_unique_samples(dataset, column="messages"):
    # Using a set of hashes saves massive amounts of RAM
    seen_hashes = set()

    for text in dataset[column]:
        c = ''.join([e['content'] for e in text])
        # We hash the string to keep the 'seen' set small
        # .encode() is needed for hashing
        h = hashlib.md5(c.encode('utf-8')).hexdigest()
        seen_hashes.add(h)

    return len(seen_hashes)

def deduplicate_dataset(dataset, column="messages"):
    seen_hashes = set()
    kept_indices = []

    for idx, example in enumerate(dataset[column]):
        c = ''.join([e['content'] for e in example])

        # 2. Create a hash
        h = hashlib.md5(c.encode('utf-8')).hexdigest()

        # 3. Only keep the index if we haven't seen this hash before
        if h not in seen_hashes:
            seen_hashes.add(h)
            kept_indices.append(idx)

    # 4. Create the new deduplicated dataset using the kept indices
    return dataset.select(kept_indices)

data_path= '/path/to/dataset.jsonl'
data = load_dataset('json', data_files= {'train': data_path})['train']
unique_dataset = deduplicate_dataset(data)

unique_dataset.to_json('dataset_dedupe.jsonl')
print(f'{len(unique_dataset)} samples out of {len(data)} total.')

