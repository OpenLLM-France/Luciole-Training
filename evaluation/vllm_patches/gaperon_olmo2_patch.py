"""Make vLLM honour an explicit ``head_dim`` in Olmo2/Olmo3 configs.

Gaperon (almanach/Gaperon-1125-*) was trained with the HF Transformers Olmo2
implementation, which reads ``head_dim`` from the config. vLLM instead derives
it as ``hidden_size // num_attention_heads``. For Gaperon-24B (hidden_size=5120,
num_attention_heads=32, head_dim=128) that gives 160 instead of 128, and loading
fails with::

    AssertionError: Attempted to load weight (torch.Size([1024])) into parameter
    (torch.Size([1280]))

See https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1

Rather than shipping a copy of ``Olmo2Attention.__init__`` (whose body changes
between vLLM releases -- e.g. the ``get_rope`` signature differs between 0.11
and 0.21), we rewrite the source of the *installed* method: two lines are
substituted and the result is re-compiled in the module namespace. If the
expected lines are not found, we raise instead of silently doing nothing.

The patch is a no-op for regular Olmo2 checkpoints, where ``head_dim`` equals
``hidden_size // num_attention_heads``.
"""

import inspect
import os
import sys
import textwrap

MODULE_NAME = "vllm.model_executor.models.olmo2"

# (needle, replacement, expected occurrences)
_SUBSTITUTIONS = [
    # zero-arg super() needs a __class__ cell, which an exec'd function lacks.
    (
        "super().__init__()",
        "nn.Module.__init__(self)",
        1,
    ),
    # The actual fix: trust head_dim from the config when it is set.
    (
        "self.head_dim = hidden_size // self.total_num_heads",
        'self.head_dim = getattr(self.config, "head_dim", None)'
        " or hidden_size // self.total_num_heads",
        1,
    ),
    # q_norm is sized from hidden_size, which is only correct when
    # head_dim == hidden_size // num_attention_heads.
    (
        "RMSNorm(self.config.hidden_size,",
        "RMSNorm(self.total_num_heads * self.head_dim,",
        1,
    ),
]

_PATCHED_FLAG = "_gaperon_head_dim_patched"


def patch_module(module):
    """Patch ``Olmo2Attention.__init__`` in an already-imported olmo2 module."""
    cls = module.Olmo2Attention
    if getattr(cls, _PATCHED_FLAG, False):
        return

    source = textwrap.dedent(inspect.getsource(cls.__init__))
    for needle, replacement, count in _SUBSTITUTIONS:
        found = source.count(needle)
        if found != count:
            raise RuntimeError(
                f"gaperon_olmo2_patch: expected {count} occurrence(s) of "
                f"{needle!r} in Olmo2Attention.__init__ ({module.__file__}), "
                f"found {found}. The patch needs to be updated for this vLLM "
                "version."
            )
        source = source.replace(needle, replacement)

    namespace = dict(module.__dict__)
    exec(compile(source, f"<gaperon_olmo2_patch:{module.__file__}>", "exec"), namespace)
    cls.__init__ = namespace["__init__"]
    setattr(cls, _PATCHED_FLAG, True)
    # Printed so the Slurm log tells "patch never ran" apart from "patch ran but
    # did not help" -- the two have very different fixes.
    print(
        f"gaperon_olmo2_patch: patched Olmo2Attention.__init__ (pid {os.getpid()})",
        file=sys.stderr,
        flush=True,
    )


def patch_now():
    """Import the olmo2 module (pulling in vLLM) and patch it."""
    import importlib

    patch_module(importlib.import_module(MODULE_NAME))


if __name__ == "__main__":
    patch_now()
    print("gaperon_olmo2_patch: OK")
