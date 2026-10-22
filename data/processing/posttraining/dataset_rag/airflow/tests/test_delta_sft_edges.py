"""SFT sources without a converter and delta runs without refusal judgments."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "dpo"))
import filter_jz_delta as F
import gold_columns as G


class DeltaSftTests(unittest.TestCase):
    def test_gold_from_evaluated_cache_based_runs(self):
        for ds in ("hotpotqa_en", "hotpotqa_fr"):
            with self.subTest(ds=ds), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sft = root / "sft"
                sft.mkdir()
                (sft / f"{ds}_augmented_evaluated.jsonl").write_text(json.dumps({"id": "q", "answer": "Paris"}) + "\n")
                d = root / "run/qwen35_35b_a3b" / ds
                d.mkdir(parents=True)
                (d / "prompts.jsonl").write_text(json.dumps({"sft_run": str(sft), "qid": "q"}) + "\n")
                self.assertEqual(G.lookup(ds, root / "run")("q"), ("Paris", None))
                (sft / "other_augmented_evaluated.jsonl").write_text('{}\n')
                with self.assertRaisesRegex(ValueError, "un seul"):
                    G.lookup(ds, root / "run")

    def filter_without_verdicts(self, category):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "run/qwen35_35b_a3b/hotpotqa_fr"
            d.mkdir(parents=True)
            (d / "prompts.jsonl").write_text('{}\n')
            src = root / "pairs"
            src.mkdir()
            (src / "manifest.json").write_text('{"sha256": "test"}')
            pair = {"pid": "p", "rejected_k": 0, "category": category, "dataset": "hotpotqa_fr"}
            (src / "delta_pairs.jsonl").write_text(json.dumps(pair) + "\n")
            (src / "delta_pairs_datatrove.jsonl").write_text('{"id": "p"}\n')
            argv = ["filter", "--src", str(src), "--out", str(root / "out"), "--jz-run", str(root / "run")]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                F.main()
            self.assertEqual(json.loads((root / "out/delta_pairs.jsonl").read_text()), pair)

    def test_no_refusals_does_not_require_judgments(self):
        self.filter_without_verdicts("A_wrong_answer")

    def test_missing_required_refusal_judgment_still_fails(self):
        with self.assertRaises(KeyError):
            self.filter_without_verdicts("U_bad_refusal")
