from dataclasses import dataclass
from nemo.collections.llm.gpt.model.llama import Llama31Config


@dataclass
class Llama32Config1B(Llama31Config):
    """Configuration for a 1B parameter Llama 3.2 model.

    Specific configuration for the 1B Llama 3.2 model with 16 layers,
    2048 hidden size, and 32 attention heads (8 query groups).
    """

    scale_factor: float = 32.0
    share_embeddings_and_output_weights: bool = True
    rotary_base: int = 500_000
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128


def get_config():
    # return Llama32Config1B()
    return Llama31Config(
        scale_factor=32.0,
        share_embeddings_and_output_weights=True,
        rotary_base=500_000,
        num_layers=16,
        hidden_size=2048,
        ffn_hidden_size=8192,
        num_attention_heads=32,
        num_query_groups=8,
        make_vocab_size_divisible_by=128,
    )


def convert_to_llama32_1b(model_config):
    model_config.config.scale_factor = 32.0
    model_config.config.share_embeddings_and_output_weights = True
    model_config.config.rotary_base = 500_000
    model_config.config.num_layers = 16
    model_config.config.hidden_size = 2048
    model_config.config.ffn_hidden_size = 8192
    model_config.config.num_attention_heads = 32
    model_config.config.num_query_groups = 8
    model_config.config.make_vocab_size_divisible_by = 128
    return model_config
