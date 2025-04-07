import torch
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig

if __name__ == "__main__":
    seq_length = 2048
    global_batch_size = 16

    ## setup the dummy dataset
    data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size)

    ## initialize a small GPT model
    gpt_config = llm.Llama2Config7B(
        num_layers=1,
        hidden_size=384,
        ffn_hidden_size=1536,
        num_attention_heads=8,
        seq_length=seq_length,
        init_method_std=0.023,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
    )
    model = llm.LlamaModel(gpt_config, tokenizer=data.tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        num_nodes=2,
        devices=4, ## you can change the number of devices to suit your setup
        max_steps=5,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/test_logdir", ## logs and checkpoints will be written here
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )