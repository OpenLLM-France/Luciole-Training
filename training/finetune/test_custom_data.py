from nemo_patch.data.fine_tuning import FineTuningDataModule
from nemo_patch.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

tokenizer_name = "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct"
chat = True
packing = True
seq_length = 4096

tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)

packed_sequence_specs = PackedSequenceSpecs(
    packed_sequence_size=seq_length,
    tokenizer_model_name=tokenizer_name.split("/")[-1],
)

# No chat - no packing
if not chat and not packing:
    data = FineTuningDataModule(
        dataset_root="./databricks",
        tokenizer=tokenizer,
        packed_sequence_specs=None,
        dataset_kwargs=None,
    )
elif not chat and packing:
    data = FineTuningDataModule(
        dataset_root="./databricks",
        micro_batch_size=1,
        tokenizer=tokenizer,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs=None,
    )
    data.prepare_data()
elif chat and packing:
    data = FineTuningDataModule(
        dataset_root="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/instruct_data/sft_mix_test3",
        micro_batch_size=1,
        tokenizer=tokenizer,
        seq_length=seq_length,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
    )
    data.prepare_data()
else:
    raise NotImplementedError

dataloader = data.train_dataloader()
batch = next(iter(dataloader))

print("\n\nBatch keys:", batch.keys())
print(batch)
print("\n\nFinished")

"""
Batch keys: dict_keys(['tokens', 'labels', 'loss_mask', 'position_ids', 'token_count', 'attention_mask', 'cu_seqlens', 'cu_seqlens_argmin', 'max_seqlen', 'cu_seqlens_unpadded', 'cu_seqlens_unpadded_argmin'])
{'tokens': tensor([[  260, 10944,   316,  6154,  7916,  1408, 35015, 16378,   344,   261,
           316,   260,  3477,   316, 68189,  5558,   300,   322,  2035,   300,
           322,  2779,   261,   316,   260, 44428,   316,   322,  2035,   300,
           322,  5558,   300,   324,   344,   261,   317,   260,  3477,   316,
          6467, 29796,  5558,   300,   323,  2035,   300,   321,  2779,   261,
           316,   260, 44428,   316,   262,   317,   263,   317,   323,  2035,
           300,   321,  5558,  3030,   300,   324,   344,   261,   317,   260,
         10944,   316,  6154,  7916,  1408, 35015, 16378,   344,   261,   316,
           260,  3477,   316, 68189,  5558,   300,   322,  2035,   300,   322,
          2779,   261,   316,   260, 44428,   316,   322,  2035,   300,   322,
          5558,   300,   324,   344,   261,   317,   260,  3477,   316,  6467,
         29796,  5558,   300,   323,  2035,   300,   321,  2779,   261,   316,
           260, 44428,   316,   262,   317,   263,   317,   323,  2035,   300,
           321,  5558,  3030,   300,   324,   344,   261,   317,     1,     1,
             1,     1,     1,     1]]), 'labels': tensor([[10944,   316,  6154,  7916,  1408, 35015, 16378,   344,   261,   316,
           260,  3477,   316, 68189,  5558,   300,   322,  2035,   300,   322,
          2779,   261,   316,   260, 44428,   316,   322,  2035,   300,   322,
          5558,   300,   324,   344,   261,   317,   260,  3477,   316,  6467,
         29796,  5558,   300,   323,  2035,   300,   321,  2779,   261,   316,
           260, 44428,   316,   262,   317,   263,   317,   323,  2035,   300,
           321,  5558,  3030,   300,   324,   344,   261,   317,     1, 10944,
           316,  6154,  7916,  1408, 35015, 16378,   344,   261,   316,   260,
          3477,   316, 68189,  5558,   300,   322,  2035,   300,   322,  2779,
           261,   316,   260, 44428,   316,   322,  2035,   300,   322,  5558,
           300,   324,   344,   261,   317,   260,  3477,   316,  6467, 29796,
          5558,   300,   323,  2035,   300,   321,  2779,   261,   316,   260,
         44428,   316,   262,   317,   263,   317,   323,  2035,   300,   321,
          5558,  3030,   300,   324,   344,   261,   317,     1,     1,     1,
             1,     1,     1,     1]]), 'loss_mask': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]), 'position_ids': tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
         54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,  0,  1,  2,
          3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,  0,  0,  0,  0,  0,  0]]), 'token_count': [138], 'attention_mask': tensor([1]), 'cu_seqlens': tensor([[  0,  69, 138, 144,  -1]], dtype=torch.int32), 'cu_seqlens_argmin': tensor([[4]]), 'max_seqlen': tensor([[69]], dtype=torch.int32), 'cu_seqlens_unpadded': tensor([[  0,  69, 138, 138,  -1]], dtype=torch.int32), 'cu_seqlens_unpadded_argmin': tensor([[4]])}
"""
