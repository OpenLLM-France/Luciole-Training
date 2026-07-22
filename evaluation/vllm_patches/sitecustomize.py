"""Applied by putting this directory first on PYTHONPATH.

Python imports ``sitecustomize`` automatically at interpreter startup, which is
the only hook we have on the ``lighteval`` CLI. We do not import vLLM here (far
too expensive, and most processes never need it): instead we register a
meta-path finder that patches ``vllm.model_executor.models.olmo2`` the moment it
is imported. See gaperon_olmo2_patch.py for the why.
"""

import importlib
import importlib.util
import os
import sys

from gaperon_olmo2_patch import MODULE_NAME, patch_module


class _PatchOnImport:
    """Patches the olmo2 module right after it is executed, then unhooks."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE_NAME:
            return None
        # Let the regular finders resolve the module (without recursing).
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        original_exec_module = spec.loader.exec_module

        def exec_module(module):
            original_exec_module(module)
            try:
                patch_module(module)
            finally:
                if self in sys.meta_path:
                    sys.meta_path.remove(self)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PatchOnImport())

# We shadow any sitecustomize the environment itself provides, so chain to it.
_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    if os.path.abspath(_entry or os.curdir) == _here:
        continue
    _candidate = os.path.join(_entry, "sitecustomize.py")
    if os.path.isfile(_candidate):
        _spec = importlib.util.spec_from_file_location("_sitecustomize_next", _candidate)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        break
