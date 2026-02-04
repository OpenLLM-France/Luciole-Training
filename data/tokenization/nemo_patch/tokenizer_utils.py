# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path
from typing import Dict, Optional

from nemo.utils import logging


def get_nmt_tokenizer(
    library: str = "sentencepiece",
    model_name: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    use_fast: Optional[bool] = False,
    bpe_dropout: Optional[float] = 0.0,
    r2l: Optional[bool] = False,
    legacy: Optional[bool] = False,
    delimiter: Optional[str] = None,
    trust_remote_code: Optional[bool] = False,
    chat_template: Optional[Dict] = None,
):
    """
    Args:
        model_name: if using a pretrained model from NeMo, HuggingFace, or Megatron
        tokenizer_model: tokenizer model file of sentencepiece
        special_tokens: dict of special tokens
        vocab_file: path to vocab file
        use_fast: (only for HuggingFace AutoTokenizer) set to True to use fast HuggingFace tokenizer
        bpe_dropout: (experimental) BPE dropout tries to corrupt the standard segmentation procedure
            of BPE to help model better learn word compositionality and become robust to segmentation errors.
            It has empirically been shown to improve inference time BLEU scores.
        r2l: Whether to return subword IDs from right to left
    """
    if special_tokens is None:
        special_tokens_dict = {}
    else:
        special_tokens_dict = special_tokens

    if (library != "byte-level") and (
        model_name is None
        and (tokenizer_model is None or not os.path.isfile(tokenizer_model))
    ):
        raise ValueError("No Tokenizer path provided or file does not exist!")

    if library == "huggingface":
        from .auto_tokenizer import AutoTokenizer

        logging.info(
            f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {model_name}"
        )
        return AutoTokenizer(
            pretrained_model_name=model_name,
            vocab_file=vocab_file,
            merges_file=merges_file,
            **special_tokens_dict,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise NotImplementedError(
            'Currently we only support "huggingface", "sentencepiece", "megatron", and "byte-level" tokenizer'
            "libraries."
        )
