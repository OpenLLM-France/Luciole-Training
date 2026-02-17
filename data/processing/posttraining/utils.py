import os
import importlib.util

# Directly load pretraining/utils.py under a unique module name to avoid
# a circular import, since this file is also named 'utils'.

spec = importlib.util.spec_from_file_location(
    "pretraining_utils",
    os.path.join(os.path.dirname(__file__), "..", "pretraining", "utils.py"),
)
pretraining_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretraining_utils)

create_parser = pretraining_utils.create_parser
parse_args = pretraining_utils.parse_args
create_executor = pretraining_utils.create_executor
add_sampler_filter = pretraining_utils.add_sampler_filter
_custom_adapter_for_hf = pretraining_utils._custom_adapter_for_hf
HF_SCHEMA = pretraining_utils.HF_SCHEMA
