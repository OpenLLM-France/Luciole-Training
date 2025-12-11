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
    tokenizer_model_name=tokenizer_name.split("/")[-2],
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
        dataset_root="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/instruct_data/sft_mix_test2",
        micro_batch_size=1,
        tokenizer=tokenizer,
        seq_length=seq_length,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
    )
    data.prepare_data()
else:
    raise NotImplementedError

print("\n\nFinished")
