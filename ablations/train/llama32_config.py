from dataclasses import dataclass
from nemo.collections.llm.gpt.model.llama import Llama31Config, Llama31Config8B


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


def get_config(size_1b=True):
    # return Llama32Config1B()
    if not size_1b:
        return Llama31Config8B()
    else:
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
