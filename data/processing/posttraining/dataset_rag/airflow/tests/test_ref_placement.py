"""Reject References markers embedded in prose while retaining valid EN/FR sections."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from formatting_sft import parse_ref_citations, reformat_context_chunks
from dpo.dpo_common import ref_citations
from dpo.citations import excerpt_in_document
from utils import format_chunk_base, filter_row


class ReferencePlacementTests(unittest.TestCase):
    def setUp(self):
        self.context = format_chunk_base("Paris", "Paris est en France.")
        self.system = "Voici les documents récupérés : `" + reformat_context_chunks(self.context) + "`"

    def validate_both(self, text, end_of_completion=False):
        sft = parse_ref_citations(text, self.context, end_of_completion)
        self.assertEqual(ref_citations(text, self.system, end_of_completion), sft)
        return sft

    def test_inline_markers_and_prose_after_references_are_rejected(self):
        for text in ('Paris <ref id="1"/> est en France.',
                     'Paris est en France.\n<ref id="1"/>',
                     'Paris.\nReferences\n<ref id="1"/>\nMore prose.'):
            with self.subTest(text=text):
                _, _, invalid = self.validate_both(text, end_of_completion=True)
                self.assertGreater(invalid, 0)
                self.assertFalse(filter_row({'eval_ref_invalid': invalid, 'is_unanswerable': True}))

    def test_valid_sections_and_uncited_refusals(self):
        for text in ('Paris.\n\n**References**\n- <ref id="1"/>',
                     'Paris.\n### Références :\n• <ref id="1"/>',
                     'Paris.\n**References:** <ref id="1"/>'):
            with self.subTest(text=text):
                self.assertEqual(self.validate_both(text, end_of_completion=True), (["Paris"], 1, 0))
        self.assertEqual(self.validate_both("I cannot answer.", True), ([], 0, 0))
        self.assertEqual(self.validate_both('<ref id="1">Paris est en France.</ref>'), (["Paris"], 1, 0))

    def test_invalid_inline_citations_agree(self):
        for text in ('<ref id="2">Paris</ref>', '<ref id="1">Lyon</ref>',
                     '<ref id="1"></ref>', '<ref id="1"/>', '##Cite "Paris"##'):
            with self.subTest(text=text):
                self.assertGreater(self.validate_both(text)[2], 0)
        self.assertEqual(self.validate_both('<ref id="1">Paris [...] France.</ref>'), (["Paris"], 1, 0))

    def test_elisions_preserve_fragment_order(self):
        doc = "Revenue 2019 12 5 2018 10 1 Net income 3 2"
        for excerpt, expected in (("Revenue 2019 12 5", True), ("Revenue ... 12 5", True),
                                  ("Revenue [...] 12 5 [...] Net income", True),
                                  ("Net income ... Revenue", False), ("Revenue … 99", False),
                                  ("...", False), ("x [ … ] y", False)):
            with self.subTest(excerpt=excerpt):
                self.assertEqual(excerpt_in_document(excerpt, doc), expected)
