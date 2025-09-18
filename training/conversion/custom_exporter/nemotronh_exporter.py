import torch

from nemo.lightning import io


from nemo.collections.llm.gpt.model.ssm import MambaModel, HFNemotronHExporter
from .modeling_nemotron_h import NemotronHForCausalLM


@io.model_exporter(MambaModel, "hf")
class CustomHFNemotronHExporter(HFNemotronHExporter):
    """
    A model exporter for converting Mamba models to Hugging Face format.

    Attributes:
        path (str): The path to save the exported model.
        model_config (SSMConfig): The configuration for the model.
    """

    def init(self, dtype=torch.bfloat16) -> "NemotronHForCausalLM":
        """
        Initializes the model for export.
        """
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return NemotronHForCausalLM(self.config)

    @property
    def tokenizer(self):
        """
        Loads the tokenizer from the specified path.
        Returns:
            AutoTokenizer: The tokenizer object.
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self):
        """
        Loads the model configuration from the specified path.
        Returns:
            SSMConfig: The model configuration object.
        """
        print("Load the model")
        # source = io.load_context(str(self), subpath="model.config")

        from .hf_nemotronh_config import NemotronHConfig as HFNemotronConfig

        return HFNemotronConfig(
            architectures=["NemotronHForCausalLM"],
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )
