"""Citation validation shared by SFT and DPO; standard library only.

Kept beside the DPO scripts so copying that directory to a GPU host remains sufficient.
Callers supply document numbers, titles and the text shown to the model.
"""
import re

_REF_RE = re.compile(r'<ref id="(\d+)">(.*?)</ref>', re.DOTALL)
_REF_DOC_RE = re.compile(r'<ref id="(\d+)"/>')
# Left over once the valid <ref> tags are removed, any of these means a malformed
# tag or a citation written in another format.
_STRAY_CITATION_RE = re.compile(
    r"</?ref\b|##Cite|##begin_quote##|##end_quote##|<<quote>>|<<cite:|<quote>|<cite>|\[QUOTE\]|\[Source:"
)


# An elision inside an excerpt: "...", "…" or the bracketed "[...]" / "[…]".
_ELISION_RE = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)\s*")


def excerpt_in_document(excerpt: str, document: str) -> bool:
    """Whether *excerpt* is verbatim in *document*, elisions allowed.

    An excerpt cut with "..." (e.g. a table row header and its value) passes when
    every fragment between the elisions is verbatim in the document, in order.
    Both strings are whitespace-normalized by the caller.
    """
    if excerpt in document:
        return True
    fragments = [f for f in _ELISION_RE.split(excerpt) if f]
    if not fragments:
        return False
    pos = 0
    for fragment in fragments:
        found = document.find(fragment, pos)
        if found < 0:
            return False
        pos = found + len(fragment)
    return True


def validate_ref_citations(
    answer: str, documents: dict[int, tuple[str | None, str]], end_of_completion: bool = False,
) -> tuple[list[str], int, int]:
    """Return cited titles, tag count and invalid count (including misplaced References)."""
    def ws(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    titles: list[str] = []
    invalid = 0
    tag_re = _REF_DOC_RE if end_of_completion else _REF_RE
    refs = list(tag_re.finditer(answer))
    for ref in refs:
        n = int(ref.group(1))
        if n not in documents:
            invalid += 1
            continue
        if not end_of_completion:
            excerpt = ws(ref.group(2))
            if not excerpt or not excerpt_in_document(excerpt, ws(documents[n][1])):
                invalid += 1
                continue
        title = documents[n][0]
        if title and title not in titles:
            titles.append(title)
    if end_of_completion and refs:
        # References must be a trailing section, never markers embedded in prose.
        before = answer[:refs[0].start()]
        heading = re.search(r"(?im)^[ \t#*]*(?:References|Références)[ \t:*]*(?:\n|$)", before)
        prefix = before[heading.end():] if heading else before
        tail = tag_re.sub("", answer[refs[0].start():])
        if heading is None or re.fullmatch(r"[\s*\-•]*", prefix + tail) is None:
            invalid += 1
    invalid += len(_STRAY_CITATION_RE.findall(tag_re.sub("", answer)))
    return titles, len(refs), invalid
