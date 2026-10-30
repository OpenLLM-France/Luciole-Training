"""TAT-QA table captions travel with their table."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convert_tatqa_to_augment_format as C  # noqa: E402

DOC = {
    "table": {"uid": "doc1", "table": [["", "2019", "2018"], ["Revenue", "1,234", "1,100"]]},
    "paragraphs": [
        {"order": 1, "text": "Revenue grew on higher volumes in every region we operate in, led by Europe."},
        {"order": 2, "text": "Revenue by year is as follows (in millions):"},
        {"order": 3, "text": "The 2018 figure was restated after the disposal of the Asian business unit."},
    ],
    "questions": [
        {"uid": "q1", "question": "What was the revenue in 2019?", "answer": "1,234",
         "answer_type": "span", "answer_from": "table-text", "rel_paragraphs": ["2"]},
        {"uid": "q2", "question": "What was the change in revenue?", "answer": 134,
         "answer_type": "arithmetic", "answer_from": "table-text", "rel_paragraphs": ["3"],
         "derivation": "1,234-1,100"},
        {"uid": "q3", "question": "Why was 2018 restated?", "answer": "after the disposal of the Asian business unit",
         "answer_type": "span", "answer_from": "table-text", "rel_paragraphs": ["3"]},
    ],
}


class CaptionTests(unittest.TestCase):
    def test_caption_detection(self):
        for text in ("Revenue by year is as follows (in millions):", "(Dollars in thousands, except per share data)",
                     "The following table summarizes our leases."):
            self.assertTrue(C.is_table_caption(text), text)
        for text in ("Revenue grew on higher volumes.", "x" * 700 + ":"):
            self.assertFalse(C.is_table_caption(text), text[:40])

    def test_caption_folded_into_table(self):
        _, chunks, order_to_title = C.tatqa_document_chunks(DOC)
        self.assertEqual([c["id"] for c in chunks], ["Table", "Paragraph 1", "Paragraph 2"])
        self.assertTrue(chunks[0]["text"].startswith("Revenue by year is as follows (in millions):\n\n| "))
        self.assertEqual(order_to_title[2], "Table")
        rows = C.convert_doc(DOC)
        self.assertEqual(rows[0]["supporting_facts"]["title"], ["Table"])
        self.assertEqual(rows[1]["supporting_facts"]["title"], ["Table", "Paragraph 2"])


if __name__ == "__main__":
    unittest.main()
