import argparse
import os
from nemo.collections.llm.gpt.data import FineTuningDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections import llm
import fiddle as fdl


def create_data(data_path, tokenizer_path, batch_size=512, seq_length=2048, packed_seq_length=2048):
    tokenizer = AutoTokenizer(tokenizer_path)
    data = FineTuningDataModule(
        dataset_root=data_path,
        global_batch_size=batch_size,   
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=seq_length,
        tokenizer=tokenizer,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=packed_seq_length,
            tokenizer_model_name="-".join(tokenizer_path.split("/")[-3:]),
        ),
    )
    return data