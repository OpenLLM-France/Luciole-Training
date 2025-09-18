import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step, torch_dtype_from_mcore_config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, get_vocab_size, io, teardown
from nemo.lightning.io.state import _ModelState
from nemo.utils import logging

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model.ssm import MambaModel, HFNemotronHExporter

@io.model_exporter(MambaModel, "hf")
class CustomHFNemotronHExporter(HFNemotronHExporter):
    """
    A model exporter for converting Mamba models to Hugging Face format.

    Attributes:
        path (str): The path to save the exported model.
        model_config (SSMConfig): The configuration for the model.
    """
    
    @property
    def tokenizer(self):
        """
        Loads the tokenizer from the specified path.
        Returns:
            AutoTokenizer: The tokenizer object.
        """

        # return AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        return io.load_context(str(self), subpath="model").tokenizer
    
    @property
    def config(self):
        """
        Loads the model configuration from the specified path.
        Returns:
            SSMConfig: The model configuration object.
        """
        source = io.load_context(str(self), subpath="model.config")

        from .hf_nemotronh_config import NemotronHConfig as HFNemotronConfig
        
        return HFNemotronConfig(
                # architectures=["NemotronHForCausalLM"],
                vocab_size=self.tokenizer.vocab_size,
                bos_token_id=self.tokenizer.bos_id,
                eos_token_id=self.tokenizer.eos_id,
        )
