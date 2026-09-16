"""Helpers for held-out DPO prompt pools."""
from __future__ import annotations

import hashlib
import re

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


_FILENAME_RE = re.compile(r"^\* filename: (.+?)\s*$", re.M)

# The 4 training citation styles of formatting_sft.CITATION_STYLES.
STYLE_MARKERS = {
    "default": "##begin_quote##",
    "xml": "<quote>",
    "bracket": "[QUOTE]",
    "double_angle": "<<quote>>",
}

def context_titles(system_prompt: str) -> list[str]:
    return _FILENAME_RE.findall(system_prompt)


def requested_citation_style(system_prompt: str) -> tuple[str | None, str]:
    """(style name, placement) requested by the system prompt's citation rule."""
    rule = next((l for l in system_prompt.splitlines()
                 if "When quoting from the context" in l or "Lorsque vous citez" in l), "")
    style = None
    # double_angle must be tested before xml ("<<quote>>" contains "<quote>").
    for name in ("double_angle", "default", "bracket", "xml"):
        if f"`{STYLE_MARKERS[name]}`" in rule:
            style = name
            break
    placement = "eoc" if ("Do **not** add citation" in rule or "N'ajoutez **pas**" in rule) else "inline"
    return style, placement

