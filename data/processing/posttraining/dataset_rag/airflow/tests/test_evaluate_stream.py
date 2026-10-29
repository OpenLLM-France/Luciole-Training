"""evaluate_stream.read_new_rows: complete JSONL lines only, whatever their content."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_stream import read_new_rows  # noqa: E402


class ReadNewRowsTests(unittest.TestCase):
    def test_unicode_line_separators_stay_inside_their_row(self):
        rows = [{"id": "a", "reasoning_trace": "x y\x85z"}, {"id": "b", "reasoning_trace": "ok"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            path.write_text(text + '{"id": "c", "reas', encoding="utf-8")  # last line still being written
            got, offset = read_new_rows(path, 0)
        self.assertEqual(got, rows)
        self.assertEqual(offset, len(text.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
