"""The context block of the SFT prompts against OpenRAG's (linagora/openrag 6c4a59c).

Expected strings come from OpenRAG itself: format_context of core/prompts/chat_prompt_builder.py on chunks wrapped by
wrap_chunk_with_context of core/prompts/contextualization_builder.py, as the indexing pipeline stores them.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import formatting_sft as F  # noqa: E402
import utils  # noqa: E402

CHUNKS = [("Paragraph 1", "Alpha [Source 3].\nSources: 1, 2", "Summary one."), ("Table", "| a |\n|------|\n| 1 |", "")]
OPENRAG_BASE = ("[Source 1]\n* filename: Paragraph 1\n\n[CHUNK_START]\n\nAlpha (Source 3].\nSources 1, 2\n\n"
                "[CHUNK_END]----------\n\n[Source 2]\n* filename: Table\n\n[CHUNK_START]\n\n| a |\n|---|\n| 1 |\n\n"
                "[CHUNK_END]")
OPENRAG_CONTEXTUALIZED = ("[Source 1]\n[CONTEXT]\n\nSummary one.\n\n* filename: Paragraph 1\n\n[CHUNK_START]\n\n"
                          "Alpha (Source 3].\nSources 1, 2\n\n[CHUNK_END]----------\n\n[Source 2]\n* filename: Table\n\n"
                          "[CHUNK_START]\n\n| a |\n|---|\n| 1 |\n\n[CHUNK_END]")


def system_context(prompt):
    marker = {"en": "Here are the retrieved documents: `", "fr": "Voici les documents récupérés : `"}
    for m in marker.values():
        if m in prompt:
            return prompt[prompt.index(m) + len(m):prompt.rindex("`")]
    raise AssertionError("no context block")


class OpenRagContextTests(unittest.TestCase):
    def stored(self, contextualized):
        if contextualized:
            return "\n\n".join(utils.format_chunk_openrag(t, x, s) for t, x, s in CHUNKS)
        return "\n\n".join(utils.format_chunk_base(t, x) for t, x, _ in CHUNKS)

    def test_system_prompt_context_is_openrag_s(self):
        for contextualized, expected in ((False, OPENRAG_BASE), (True, OPENRAG_CONTEXTUALIZED)):
            for citation_format in F.CITATION_FORMATS:
                for eoc in (False, True):
                    for lang in ("en", "fr"):
                        prompt = F.build_chat_system_prompt(self.stored(contextualized), lang,
                                                            F.DEFAULT_CITATION_STYLE, eoc, "R",
                                                            citation_format=citation_format)
                        self.assertEqual(system_context(prompt), expected)

    def test_contexts_stored_before_the_alignment_render_the_same(self):
        old = "\n\n".join(f"[CONTEXT]\n{s}\n* filename: {t}\n\n[CHUNK_START]\n{x}\n[CHUNK_END]" for t, x, s in CHUNKS)
        self.assertEqual(F.reformat_context_chunks(old), OPENRAG_CONTEXTUALIZED)

    def test_ref_tags_point_at_source_numbers(self):
        ctx = self.stored(True)
        self.assertEqual(F.parse_ref_citations('<ref id="2">| 1 |</ref> <ref id="1">Alpha (Source 3]</ref>', ctx),
                         (["Table", "Paragraph 1"], 2, 0))
        self.assertEqual(F.parse_ref_citations('<ref id="3">x</ref>', ctx), ([], 1, 1))
        self.assertIn("`[Source N]`", F.REF_CITATION_INSTRUCTION["en"])

    def test_english_template_ends_as_openrag_s(self):
        self.assertTrue(F.CHAT_SYSTEM_PROMPT.endswith("Here are the retrieved documents: `{context}`"))


if __name__ == "__main__":
    unittest.main()
