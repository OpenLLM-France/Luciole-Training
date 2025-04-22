import torch
import logging
import os
from argparse import ArgumentParser

from nemo.collections import llm
from nemo.lightning import io
from nemo.collections.llm.gpt.model.base import torch_dtype_from_mcore_config

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@io.model_exporter(llm.LlamaModel, "hf")
class UpdatedHFLlamaExporter(llm.gpt.model.llama.HFLlamaExporter):
    def apply(self, output_path):
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
            self.tokenizer.save_pretrained(output_path)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer {e}")

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        transforms = [
            llm.gpt.model.llama._export_qkv,
            llm.gpt.model.llama._export_linear_fc1,
            llm.gpt.model.llama._export_embedding,
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(llm.gpt.model.llama._export_head)

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )


def convert_checkpoint(input_path, output_path):
    exporter = llm.LlamaModel.exporter("hf", input_path)
    exporter.init()
    exporter.apply(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.getenv("OpenLLM_OUTPUT"),
        help="Path to a checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model.bin",
        help="Path to HF .bin file",
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.input_path, args.input_checkpoint)
    convert_checkpoint(checkpoint_path, args.output_path)
