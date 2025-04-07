import torch
from llama32_config import Llama32Config1B

from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


if __name__ == "__main__":
    data_path = "/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/Wikipedia--fr_text_document"
    tokenizer_name = "OpenLLM-France/Lucie-7B"
    ## setup the dummy dataset
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    data = PreTrainingDataModule(
        paths=data_path,
        global_batch_size=512,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=2048,  # 8192 for llama 32 1b
        tokenizer=tokenizer,
    )

    ## initialize a small GPT model

    model_config = Llama32Config1B()
    model = llm.LlamaModel(model_config, tokenizer=data.tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        num_nodes=1,
        devices=4,  ## you can change the number of devices to suit your setup
        max_steps=5,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/test_logdir2",  ## logs and checkpoints will be written here
    )
    print(trainer)
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
    )
