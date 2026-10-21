"""Offline checks of the DPO adapter's data and resume safeguards."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("dpo_pipeline", Path(__file__).resolve().parents[1] / "dpo_pipeline.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.out = Path(self.tmp.name) / "out"
        self.out.mkdir()
        self.config = {"input_run": str(self.source), "seed": 42,
                       "limit_questions": 1, "calibrate": False}
        for ds in P.DATASETS:
            prompts = [{"pid": f"{q}-{u}", "qid": q, "is_unanswerable": u,
                        "messages": [], "pilot_split": "train"}
                       for q in ("q1", "q2") for u in (False, True)]
            verdicts = [{"pid": r["pid"], "verdict": "unanswerable" if r["is_unanswerable"] else "answerable"}
                        for r in prompts]
            for gen in P.GENERATORS:
                d = self.source / gen / ds
                d.mkdir(parents=True)
                P.write(d / "prompts.jsonl", prompts)
                P.write(d / "context_verdicts.jsonl", verdicts)

    def mutate_prompts(self, change, ds="tatqa"):
        for gen in P.GENERATORS:
            path = self.source / gen / ds / "prompts.jsonl"
            records = P.rows(path)
            change(records)
            P.write(path, records)

    def test_smoke_keeps_both_versions_and_identical_generator_inputs(self):
        P.prepare(self.config, self.out)
        for ds in P.DATASETS:
            a = P.rows(self.out / P.GENERATORS[0] / ds / "prompts.jsonl")
            b = P.rows(self.out / P.GENERATORS[1] / ds / "prompts.jsonl")
            self.assertEqual(a, b)
            self.assertEqual(len({r["qid"] for r in a}), 1)
            self.assertEqual({r["is_unanswerable"] for r in a}, {False, True})

    def test_rejects_legacy_unpadded_clones(self):
        self.mutate_prompts(lambda rs: rs[1].update(heldout_drop_reason="nogold_clone"))
        with self.assertRaisesRegex(ValueError, "sans rembourrage"):
            P.prepare(self.config, self.out)

    def test_rejects_evaluation_leak(self):
        self.mutate_prompts(lambda rs: rs[0].update(in_paired_eval=True))
        with self.assertRaisesRegex(ValueError, "fuite"):
            P.prepare(self.config, self.out)

    def test_tatqa_uses_table_split_instead_of_historical_pilot_split(self):
        self.mutate_prompts(lambda rs: [r.update(pilot_split="val") for r in rs])
        P.prepare(self.config, self.out)

    def test_rejects_context_contradiction(self):
        path = self.source / P.GENERATORS[0] / "hotpotqa/context_verdicts.jsonl"
        records = P.rows(path)
        records[0]["verdict"] = "unanswerable"
        P.write(path, records)
        with self.assertRaisesRegex(ValueError, "contradictoire"):
            P.prepare(self.config, self.out)

    def test_resume_fingerprint_covers_context_verdicts(self):
        P.prepare(self.config, self.out)
        before = json.loads((self.out / "inputs.manifest.json").read_text())
        path = self.source / P.GENERATORS[0] / "hotpotqa/context_verdicts.jsonl"
        records = P.rows(path)
        for r in records:
            r["reason"] = "changed"
        P.write(path, records)
        P.prepare(self.config, self.out)
        after = json.loads((self.out / "inputs.manifest.json").read_text())
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()


SPEC_PROMPTS = importlib.util.spec_from_file_location(
    "dpo_prompts_pipeline", Path(__file__).resolve().parents[1] / "dpo_prompts_pipeline.py")
PP = importlib.util.module_from_spec(SPEC_PROMPTS)
SPEC_PROMPTS.loader.exec_module(PP)


class ContextJudgeTests(unittest.TestCase):
    """A failed context verdict ("ambiguous" + error) is asked once more, then tolerated up to max_missing_share."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.key = self.out / "key"
        self.key.write_text("none")
        for ds in PP.DATASETS:
            d = self.out / "unans_dag" / ds
            d.mkdir(parents=True)
            P.write(d / "prompts.jsonl", [{"pid": f"p{i}"} for i in range(200)])
        self.calls = []

    def fake_run(self, still_failing):
        def run(script, *args, env=None):
            out = Path(args[args.index("--out") + 1])
            done = {r["pid"] for r in P.rows(out)} if out.exists() else set()
            new = [{"pid": f"p{i}", "verdict": "ambiguous", "error": "judge_parse_failure"}
                   if i < still_failing[len(self.calls) % 2] else {"pid": f"p{i}", "verdict": "unanswerable"}
                   for i in range(200) if f"p{i}" not in done]
            P.write(out, (P.rows(out) if out.exists() else []) + new)
            self.calls.append(script)
        return run

    def config(self):
        return {"judge_key_file": str(self.key), "judge_url": "http://x/v1", "judge_model": "m",
                "judge_max_tokens": 16000, "judge_concurrency": 4, "max_missing_share": 0.01}

    def test_failed_verdicts_are_asked_again(self):
        PP.run = self.fake_run(still_failing=(5, 0))  # 5 failures on the first pass, none on the second
        PP.contexts(self.config(), self.out)
        for ds in PP.DATASETS:
            rs = P.rows(self.out / "unans_dag" / ds / "context_verdicts.jsonl")
            self.assertEqual(len(rs), 200)
            self.assertFalse(any(r.get("error") for r in rs))

    def test_persistent_failures_above_tolerance_fail(self):
        PP.run = self.fake_run(still_failing=(10, 10))  # 10/200 = 5 % > 1 %
        with self.assertRaisesRegex(RuntimeError, "échec"):
            PP.contexts(self.config(), self.out)


class DotenvTests(unittest.TestCase):
    def test_environment_wins_over_dotenv(self):
        import os
        os.environ["DPO_TEST_KEY"] = "from-env"
        self.addCleanup(os.environ.pop, "DPO_TEST_KEY")
        self.assertEqual(P.dotenv("DPO_TEST_KEY"), "from-env")


class TriggerConfigTests(unittest.TestCase):
    """dataset_rag_dpo_prompts hands the delta params over as one JSON string (from_prompts)."""

    def test_from_prompts_overrides_params_and_keeps_types(self):
        handed = json.dumps({"run_name": "p1", "input_run": "airflow_runs/dataset_rag_dpo_prompts/p1/jz_run",
                             "judge_concurrency": 64, "calibrate": False})
        c = P.load_config(json.dumps({"run_name": "default", "input_run": "dpo/data/jz_run", "judge_concurrency": 12,
                                      "calibrate": True, "from_prompts": handed}))
        self.assertEqual(c["input_run"], "airflow_runs/dataset_rag_dpo_prompts/p1/jz_run")
        self.assertIs(c["calibrate"], False)
        self.assertEqual(c["judge_concurrency"], 64)
        self.assertNotIn("from_prompts", c)

    def test_without_handover_params_are_unchanged(self):
        c = P.load_config(json.dumps({"run_name": "x", "from_prompts": None}))
        self.assertEqual(c, {"run_name": "x"})


class RefCitationTests(unittest.TestCase):
    """dpo_common on the ref citation format (numbered documents, <ref id="N">verbatim excerpt</ref>)."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(P.BASE / "dpo"))
        import dpo_common
        cls.C = dpo_common
        cls.system = ("Règles\n   * Lorsque vous citez le contexte, encadrez chaque extrait en ligne avec "
                      "`<ref id=\"N\">extrait</ref>`, où `N` est le numéro du document cité.\n\n"
                      "Voici les documents récupérés : `[1] filename: Paris\n\n[CHUNK_START]\nParis est la capitale "
                      "de la France.\n[CHUNK_END]\n\n[2] filename: Lyon\n\n[CHUNK_START]\nLyon est sur le Rhône.\n"
                      "[CHUNK_END]`")

    def test_numbered_titles_include_the_first_document_after_the_backtick(self):
        self.assertEqual(self.C.context_titles(self.system), ["Paris", "Lyon"])
        self.assertEqual(self.C.requested_citation_style(self.system), ("ref", "inline"))
        self.assertTrue(self.C.context_block(self.system).startswith("Voici les documents récupérés"))

    def test_invalid_excerpts_ids_and_other_formats_are_counted(self):
        text = ('<ref id="1">Paris est la  capitale</ref> ; <ref id="2">Lyon est sur la Seine</ref> ; '
                '<ref id="7">x</ref> ##Cite "Paris"##')
        titles, n, invalid = self.C.ref_citations(text, self.system, end_of_completion=False)
        self.assertEqual((titles, n, invalid), (["Paris"], 3, 3))
        m = self.C.citation_metrics(text, ["Paris"], ["Paris", "Lyon"], "ref", self.system)
        self.assertEqual((m["n_ref_tags"], m["n_ref_invalid"], m["cit_f1"]), (3, 3, 1.0))
        self.assertTrue(m["wrong_style"])  # a markup citation next to the ref tags

    def test_markup_prompts_are_unchanged(self):
        system = "Here are the retrieved documents: `* filename: A\n\n[CHUNK_START]\nx\n[CHUNK_END]\n* filename: B\n"
        self.assertEqual(self.C.context_titles(system), ["B"])  # historical behaviour of the markup prompts
        m = self.C.citation_metrics('x ##Cite "B"##', ["B"], ["B"], "default")
        self.assertNotIn("n_ref_tags", m)
