"""Reject References markers embedded in prose while retaining valid EN/FR sections."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from formatting_sft import parse_ref_citations
from utils import format_chunk_base, filter_row


class ReferencePlacementTests(unittest.TestCase):
    def setUp(self):
        self.context = format_chunk_base("Paris", "Paris est en France.")

    def test_inline_markers_and_prose_after_references_are_rejected(self):
        for text in ('Paris <ref id="1"/> est en France.',
                     'Paris est en France.\n<ref id="1"/>',
                     'Paris.\nReferences\n<ref id="1"/>\nMore prose.'):
            with self.subTest(text=text):
                _, _, invalid = parse_ref_citations(text, self.context, end_of_completion=True)
                self.assertGreater(invalid, 0)
                self.assertFalse(filter_row({'eval_ref_invalid': invalid, 'is_unanswerable': True}))

    def test_valid_sections_and_uncited_refusals(self):
        for text in ('Paris.\n\n**References**\n- <ref id="1"/>',
                     'Paris.\n### Références :\n• <ref id="1"/>',
                     'Paris.\n**References:** <ref id="1"/>'):
            with self.subTest(text=text):
                self.assertEqual(parse_ref_citations(text, self.context, end_of_completion=True), (["Paris"], 1, 0))
        self.assertEqual(parse_ref_citations("I cannot answer.", self.context, True), ([], 0, 0))
        self.assertEqual(parse_ref_citations('<ref id="1">Paris est en France.</ref>', self.context), (["Paris"], 1, 0))
