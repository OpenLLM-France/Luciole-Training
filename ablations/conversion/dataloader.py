from nemo.lightning.data import WrappedDataLoader
from torch.utils.data._utils.collate import default_collate
from nemo.collections.llm.gpt.data import PreTrainingDataModule
import torch


def create_random_offset(reset_positions, mean=1024):
    p = 1.0 / mean
    nnz = reset_positions.count_nonzero()
    samples = (
        torch.distributions.Geometric(probs=torch.tensor([p]))
        .sample((nnz,))
        .long()
        .squeeze(1)
    )
    values = torch.zeros_like(reset_positions, dtype=torch.long)
    values[reset_positions] = samples
    offsets = values.cumsum(dim=1)
    return offsets


def custom_collate_with_positional_offset(batch, offset=100, eos_token_id=1):
    batch = default_collate(batch)  # collate normally
    if "position_ids" in batch:
        reset_postions = batch["tokens"] == eos_token_id
        batch["position_ids"] += create_random_offset(reset_postions, 1024)
    return batch


class WrappedPreTrainingDataModule(PreTrainingDataModule):
    def __init__(self, offset_collate=False, dataloader_kwargs=None):
        super().__init__(**dataloader_kwargs)
        self.offset_collate = offset_collate

    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step

        if self.offset_collate:
            collate_fn = getattr(
                dataset, "collate_fn", custom_collate_with_positional_offset
            )
        else:
            collate_fn = getattr(dataset, "collate_fn", default_collate)

        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
        return dataloader
