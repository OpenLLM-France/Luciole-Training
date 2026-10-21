"""Exercise the delta release with small pools and real parquet serialization."""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "dpo"))
import gold_columns as G
import publish_jz_full_offpolicy as R


class EmptySplitTests(unittest.TestCase):
    def release(self, val_share, too_long=False):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, pairs, out = root / "run", root / "pairs", root / "release"
            pairs.mkdir()
            pair = {"pid": "p", "qid": "q", "dataset": "hotpotqa", "prompt": [],
                    "chosen": [{"role": "assistant", "content": "yes"}],
                    "rejected": [{"role": "assistant", "content": "no"}],
                    "chosen_k": 0, "rejected_k": 0, "category": "A_wrong_answer", "refusal_instruction": "generic"}
            (pairs / "delta_pairs.jsonl").write_text(json.dumps(pair) + "\n")
            (pairs / "manifest.json").write_text(json.dumps({"files": {"delta_pairs.jsonl": {"sha256": "test"}}}))
            for gen in ("qwen35_35b_a3b", "qwen35_4b"):
                for ds in ("hotpotqa", "tatqa"):
                    d = run / gen / ds
                    d.mkdir(parents=True)
                    (d / "prompts.jsonl").write_text('{}\n')
            state = ({"p": {"is_unanswerable": False}}, {"p": {0: {"text": "text"}}}, {}, {}, {})
            tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **k: [1] * (4097 if too_long else 4))
            transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda _: tokenizer))
            argv = ["release", "--tokenizer", "unused", "--out", str(out), "--pairs-dir", str(pairs),
                    "--jz-run", str(run), "--revision", "none", "--crossjudge", "none", "--gold-columns",
                    "--val-share", str(val_share)]
            with patch.dict(sys.modules, {"transformers": transformers}), patch.object(sys, "argv", argv), \
                    patch.object(R.Q, "state", return_value=state), patch.object(R.Q, "last_by_key", return_value={}), \
                    patch.object(G, "lookup", return_value=lambda q: ("yes", None)), contextlib.redirect_stdout(io.StringIO()):
                if too_long:
                    with self.assertRaisesRegex(RuntimeError, "Aucune paire publiable"):
                        R.main()
                else:
                    R.main()
            manifest = json.loads((out / "manifest_full_offpolicy_qwen35_35b_a3b.json").read_text())
            cfg = "hotpotqa_full_offpolicy_qwen35_35b_a3b"
            if too_long:
                self.assertTrue(manifest["configs"][cfg]["empty"])
                return
            tables = [pq.read_table(out / cfg / f"{s}.parquet") for s in ("train", "val")]
            self.assertEqual([t.num_rows for t in tables], [1, 0] if val_share == 0 else [0, 1])
            self.assertEqual(tables[0].schema, tables[1].schema)
            self.assertIn("gold_answer", tables[0].column_names)
            self.assertEqual(next(t for t in tables if t.num_rows).column("gold_answer").to_pylist(), ["yes"])

    def test_empty_validation(self):
        self.release(0)

    def test_empty_training(self):
        self.release(1)

    def test_all_pairs_over_token_budget(self):
        self.release(0, too_long=True)
