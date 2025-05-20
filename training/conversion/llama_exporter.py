import torch

from pathlib import Path

from nemo.utils import logging
from nemo.collections import llm
from nemo.lightning import io
from nemo.collections.llm.gpt.model.llama import (
    LlamaConfig,
    Llama31Config,
    _export_qkv,
    _export_linear_fc1,
    _export_embedding,
    _export_head,
)
from nemo.collections.llm.gpt.model.base import torch_dtype_from_mcore_config


@io.model_exporter(llm.LlamaModel, "hf")
class UpdatedHFLlamaExporter(llm.gpt.model.llama.HFLlamaExporter):
    # from nemo 2.2.1

    def init(self, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(output_path, state_dict=state_dict)
        else:
            target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        transforms = [_export_qkv, _export_linear_fc1, _export_embedding]
        if not self.config.tie_word_embeddings:
            transforms.append(_export_head)

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self):
        # from nemo main
        source: LlamaConfig = io.load_context(str(self), subpath="model.config")

        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                "factor": float(source.scale_factor),
                "low_freq_factor": float(source.low_freq_factor),
                "high_freq_factor": float(source.high_freq_factor),
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            }
        return HFLlamaConfig(
            architectures=["LlamaForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,  # 131072
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )
