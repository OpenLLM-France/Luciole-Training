"""Offline checks of the DPO adapter's data and resume safeguards."""
import importlib.util
import json
import shutil
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
        self.config = {"mode": "delta", "input_run": str(self.source), "seed": 42,
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

    def test_prompt_set_of_an_sft_run_is_prepared_and_read_by_the_later_stages(self):
        # jz_run of the sft_run source: one dataset of its own, no hotpotqa / tatqa
        for gen in P.GENERATORS:
            for ds in P.DATASETS:
                shutil.rmtree(self.source / gen / ds)
        for gen in P.GENERATORS:
            shutil.copytree(self._fr_prompts(), self.source / gen / "hotpotqa_fr")
        P.prepare(self.config, self.out)
        self.assertEqual(P.jz_datasets(self.out), ["hotpotqa_fr"])
        manifest = json.loads((self.out / "inputs.manifest.json").read_text())
        self.assertEqual(list(manifest["inputs"]), ["hotpotqa_fr"])
        self.assertEqual(RefCitationTests.load("dpo_common").delta_datasets(self.out), ["hotpotqa_fr"])

    def test_known_datasets_keep_their_order(self):
        P.prepare(self.config, self.out)
        self.assertEqual(P.jz_datasets(self.out), list(P.DATASETS))
        self.assertEqual(RefCitationTests.load("dpo_common").delta_datasets(self.out), list(P.DATASETS))

    def test_generator_key_file_feeds_the_sampler(self):
        self.assertIsNone(P.generator_env({"chosen_key_file": None}, "chosen"))
        key = Path(self.tmp.name) / "k"
        key.write_text("secret\n")
        self.assertEqual(P.generator_env({"chosen_key_file": str(key)}, "chosen")["SAMPLER_API_KEY"], "secret")
        key.write_text("")
        with self.assertRaisesRegex(ValueError, "chosen_key_file"):
            P.generator_env({"chosen_key_file": str(key)}, "chosen")

    def _fr_prompts(self):
        d = Path(self.tmp.name) / "fr_prompts"
        if not d.exists():
            d.mkdir()
            prompts = [{"pid": f"{q}-{u}", "qid": q, "is_unanswerable": u, "messages": [], "pilot_split": "train"}
                       for q in ("f1", "f2") for u in (False, True)]
            P.write(d / "prompts.jsonl", prompts)
            P.write(d / "context_verdicts.jsonl", [
                {"pid": r["pid"], "verdict": "unanswerable" if r["is_unanswerable"] else "answerable"} for r in prompts])
        return d


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
        return {"source": "heldout_pools", "judge_key_file": str(self.key), "judge_url": "http://x/v1", "judge_model": "m",
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


def base_config(**kw):
    c = {"run_name": "t", "mode": "onpolicy", "prompt_source": "jz_run", "replay": False, "seed": 42,
         "limit_questions": None, "calibrate": False, "generation_concurrency": 4, "judge_concurrency": 4,
         "judge_max_tokens": 16000, "mistral_concurrency": 6, "max_missing_share": 0.01,
         "policy_tokenizer": "tok", "policy_url": None, "policy_model": "luciole_policy",
         "policy_serve": {"ssh_host": "dgx"}, "heldout_v2_dir": None, "input_run": None}
    c.update(kw)
    return c


class OnPolicyPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"

    def prompts(self, strict=False):
        return [{"pid": f"{q}-{u}" + ("_s" if strict else ""), "qid": q, "is_unanswerable": u, "messages": [],
                 "pilot_split": "train"} for q in ("q1", "q2", "q3") for u in (False, True)]

    def verdicts(self, prompts):
        return [{"pid": r["pid"], "verdict": "unanswerable" if r["is_unanswerable"] else "answerable"} for r in prompts]

    def test_jz_run_gives_one_generic_pool_per_dataset(self):
        for ds in P.DATASETS:
            d = self.root / "jz" / P.GENERATORS[0] / ds
            d.mkdir(parents=True)
            P.write(d / "prompts.jsonl", self.prompts())
            P.write(d / "context_verdicts.jsonl", self.verdicts(self.prompts()))
        c = base_config(input_run=str(self.root / "jz"), limit_questions=2)
        P.prepare(c, self.out)
        self.assertEqual([p["name"] for p in P.pools(c)], ["hotpotqa_generic", "tatqa_generic"])
        for pool in P.pools(c):
            rs = P.rows(P.pool_dir(self.out, pool) / "prompts.jsonl")
            self.assertEqual(len({r["qid"] for r in rs}), 2)
            self.assertTrue(pool["drop_u_supported"])

    def heldout(self):
        v2 = self.root / "v2"
        for spec in P.HELDOUT_V2:
            d = v2 / spec["src"]
            d.mkdir(parents=True)
            ps = self.prompts(strict=spec["instruction"] == "strict")
            P.write(d / "prompts.jsonl", ps)
            P.write(d / "context_verdicts.jsonl", self.verdicts(ps))
            for name in ("samples.jsonl", "scored.jsonl", "verified.jsonl"):
                P.write(d / name, [])
            (d / "sampling.manifest.json").write_text("{}")
            for key in ("u_review", "exclude"):
                for path in (spec.get(key), spec["replay"].get(key)):
                    if path:
                        (v2 / path).parent.mkdir(parents=True, exist_ok=True)
                        (v2 / path).write_text("{}")
            (v2 / spec["replay"]["recheck"]).parent.mkdir(parents=True, exist_ok=True)
            P.write(v2 / spec["replay"]["recheck"], [{"pid": spec["name"], "k": 0, "judge": None}])
        return v2

    def test_strict_pool_keeps_the_questions_of_its_twin(self):
        c = base_config(prompt_source="heldout_v2", heldout_v2_dir=str(self.heldout()), limit_questions=1)
        P.prepare(c, self.out)
        q = {p["name"]: {r["qid"] for r in P.rows(P.pool_dir(self.out, p) / "prompts.jsonl")} for p in P.pools(c)}
        self.assertEqual(q["hotpotqa_strict"], q["hotpotqa_generic"])
        self.assertEqual(q["tatqa_strict"], q["tatqa_generic"])
        self.assertEqual(len(q["hotpotqa_generic"]), 1)

    def test_replay_uses_the_published_audit_and_a_new_run_does_not(self):
        v2 = self.heldout()
        replay = base_config(prompt_source="heldout_v2", heldout_v2_dir=str(v2), replay=True)
        P.prepare(replay, self.out)
        hp = next(p for p in P.pools(replay) if p["name"] == "hotpotqa_generic")
        self.assertEqual((hp["audit"], hp["drop_u_supported"]), ("v1.1", False))
        self.assertTrue((P.pool_dir(self.out, hp) / "u_review.json").exists())
        self.assertEqual(P.rows(P.pool_dir(self.out, hp) / "rejected_think.jsonl")[0]["pid"], "hotpotqa_generic")
        fresh = {p["name"]: p for p in P.pools(base_config(prompt_source="heldout_v2"))}
        self.assertEqual((fresh["hotpotqa_generic"]["audit"], fresh["hotpotqa_generic"]["u_review"]), ("recheck", None))
        # reviews about the prompt (context answerable, excluded pids) are kept in a new run
        self.assertIsNotNone(fresh["tatqa_generic"]["u_review"])
        self.assertIsNotNone(fresh["tatqa_strict"]["exclude"])


class OnPolicyStageTests(unittest.TestCase):
    def test_recheck_skips_the_bare_imposed_phrase_on_strict_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "pairs").mkdir()
            P.write(d / "pairs/train.jsonl", [
                {"pid": "a", "rejected_k": 0, "rejected_source": "policy", "rejected_judge": {}},
                {"pid": "b", "rejected_k": 1, "rejected_source": "policy", "rejected_is_exact_phrase": True},
                {"pid": "c", "rejected_k": -1, "rejected_source": "policy_generic_twin"}])
            P.write(d / "pairs/val.jsonl", [])
            self.assertEqual(P.recheck_todo({"instruction": "strict"}, d), [{"pid": "a", "k": 0}])
            self.assertEqual(len(P.recheck_todo({"instruction": "generic"}, d)), 2)

    def test_missing_verdicts_ignore_truncated_and_empty_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "scored.jsonl"
            P.write(f, [{"pid": "a", "k": 0, "judge": None, "finish_reason": "stop", "text": "x"},
                        {"pid": "a", "k": 1, "judge": None, "finish_reason": "length", "text": "x"},
                        {"pid": "a", "k": 2, "judge": None, "finish_reason": "stop", "text": " "},
                        {"pid": "a", "k": 3, "judge": None, "finish_reason": "stop", "text": "x"},
                        {"pid": "a", "k": 3, "judge": {}, "finish_reason": "stop", "text": "x"}])
            self.assertEqual(P.missing(f), (1, 4))
            with self.assertRaisesRegex(RuntimeError, "manquants"):
                P.tolerate({"max_missing_share": 0.01}, {"x": {"missing": 1, "of": 4}}, "jugement")

    def test_stop_policy_is_a_no_op_without_a_launched_server(self):
        for c in (base_config(mode="delta"), base_config(policy_url="http://x/v1"),
                  base_config(prompt_source="heldout_v2", replay=True)):
            P.stop_policy(c)  # no SSH attempted


class ServeTests(unittest.TestCase):
    def test_vllm_bound_to_localhost_and_free_memory_checked(self):
        c = base_config(run_name="a b/c")
        script = P.start_script(c, P.serve_config(c))
        self.assertIn("-p 127.0.0.1:8016:8000", script)
        self.assertIn('[ "$free_gb" -ge 50 ]', script)
        self.assertIn("vllm_dpo_pairs_a_b_c", script)
        self.assertNotIn("drop_caches", script)

    def test_serve_config_rejects_unknown_keys_and_shell_metacharacters(self):
        for serve, err in (({"ssh_host": "dgx", "gpu": 1}, "inconnues"), ({"ssh_host": "dgx;rm"}, "ssh_host"),
                           ({"ssh_host": "dgx", "model_dir": "a/../b"}, "model_dir"),
                           ({"ssh_host": "dgx", "image": "img $(x)"}, "image")):
            with self.assertRaisesRegex(ValueError, err):
                P.serve_config(base_config(policy_serve=serve))


class GeneratorServeTests(unittest.TestCase):
    """Delta generators started by the DAG (rejected_serve): an HF cache repo mounted whole, its snapshot served."""
    SERVE = {"ssh_host": "dgx", "model_dir": "lora_instruct_ablation/hf_cache/hub/models--Qwen--Qwen3.5-4B",
             "model_subdir": "snapshots/851bf6e"}

    def config(self, **kw):
        return base_config(**{"mode": "delta", "run_name": "fr1", "rejected_model": "Qwen3.5-4B", "rejected_url": None,
                              "rejected_serve": self.SERVE, "chosen_url": "http://x/v1",
                              "chosen_model": "Qwen3.5-35B-A3B", **kw})

    def test_hf_cache_repo_served_from_its_snapshot_with_its_own_template(self):
        c = self.config()
        script = P.start_script(c, P.serve_config(c, "rejected"), "rejected")
        self.assertIn('-v "$(readlink -f "$M"):/model:ro"', script)
        self.assertIn("vllm serve /model/snapshots/851bf6e --served-model-name Qwen3.5-4B", script)
        self.assertIn("-p 127.0.0.1:8017:8000", script)
        self.assertIn("nvcr.io/nvidia/vllm:26.04-py3", script)
        self.assertIn("vllm_dpo_pairs_rejected_fr1", script)
        self.assertNotIn("--chat-template", script)

    def test_policy_server_keeps_its_template_and_directory(self):
        c = base_config(run_name="op")
        script = P.start_script(c, P.serve_config(c))
        self.assertIn("vllm serve /model --served-model-name luciole_policy", script)
        self.assertIn("--chat-template /model/chat_template.jinja", script)
        self.assertIn("vllm_dpo_pairs_op", script)

    def test_teardown_covers_the_started_generators_only(self):
        self.assertEqual(P.served_roles(self.config()), ["rejected"])
        self.assertEqual(P.served_roles(self.config(rejected_serve=None, rejected_url="http://y/v1")), [])

    def test_url_and_serve_are_exclusive_and_model_dir_required(self):
        for kw, err in (({"rejected_url": "http://y/v1"}, "null avec rejected_serve"),
                        ({"rejected_serve": None}, "rejected_url ou rejected_serve"),
                        ({"rejected_serve": {"ssh_host": "dgx"}}, "model_dir")):
            with self.assertRaisesRegex(ValueError, err):
                P.validate(self.config(**kw))
        P.validate(self.config())

class ValidateTests(unittest.TestCase):
    def test_mode_constraints(self):
        for kw, err in (({"mode": "delta", "prompt_source": "heldout_v2"}, "delta"),
                        ({"replay": True}, "replay"),
                        ({"prompt_source": "heldout_v2", "replay": True, "limit_questions": 3}, "replay"),
                        ({"mistral_concurrency": 13}, "12"),
                        ({"policy_tokenizer": None}, "policy_tokenizer"),
                        ({"mode": "both"}, "mode")):
            with self.assertRaisesRegex(ValueError, err):
                P.validate(base_config(**kw))
        P.validate(base_config(prompt_source="heldout_v2", replay=True, policy_serve={"ssh_host": "bad host"}))


class GoldColumnTests(unittest.TestCase):
    @unittest.skipUnless((P.BASE / "tatqa_dataset_train.json").exists(), "tatqa_dataset_train.json absent")
    def test_tatqa_gold_after_qid_as_string_columns(self):
        import sys
        import pyarrow as pa
        import pyarrow.parquet as pq
        sys.path.insert(0, str(P.BASE / "dpo"))
        import gold_columns
        doc = json.load(open(P.BASE / "tatqa_dataset_train.json"))[0]
        q = doc["questions"][0]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "t.parquet"
            pq.write_table(pa.Table.from_pylist([{"pid": "p", "qid": f"{doc['table']['uid']}_{q['uid']}", "x": 1}]), f)
            gold_columns.add_to_parquet(f, "tatqa")
            t = pq.read_table(f)
        self.assertEqual(t.column_names, ["pid", "qid", "gold_answer", "gold_scale", "x"])
        self.assertEqual(str(t.schema.field("gold_scale").type), "string")


if __name__ == "__main__":
    unittest.main()


class RefCitationTests(unittest.TestCase):
    """dpo_common on the ref citation format (numbered documents, <ref id="N">verbatim excerpt</ref>)."""

    @staticmethod
    def load(name):
        import importlib
        import sys
        if str(P.BASE / "dpo") not in sys.path:
            sys.path.insert(0, str(P.BASE / "dpo"))
        return importlib.import_module(name)

    @classmethod
    def setUpClass(cls):
        cls.C = cls.load("dpo_common")
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

    def test_chosen_citation_rule_of_the_delta_run(self):
        Q = self.load("qwen_chosen_all")
        p = {"citation_style": "ref", "citation_placement": "inline", "messages": [{"content": self.system}]}
        self.assertTrue(Q.citation_ok(p, 'La capitale est Paris <ref id="1">Paris est la capitale</ref>.'))
        self.assertTrue(Q.quoted(p, "sans marqueur"))  # no quote marker besides the tag: neutral for --prefer-quoted
        for bad in ("Paris, sans citation.", 'Paris <ref id="2">Paris est la capitale</ref>.',
                    'Paris <ref id="1">Paris est la capitale</ref> ##Cite "Paris"##', 'Paris <ref id="1"/>.'):
            self.assertFalse(Q.citation_ok(p, bad), bad)
        eoc = {**p, "citation_placement": "eoc"}
        self.assertTrue(Q.citation_ok(eoc, 'Paris.\n\n**Références**\n- <ref id="1"/>'))
        self.assertFalse(Q.citation_ok(eoc, 'Paris <ref id="1"/>, puis la suite.'))

    def test_rejected_citfail_judged_only_where_a_pair_can_use_it(self):
        T = self.load("jz_todo_rejected")
        ok, bad = 'Paris <ref id="1">Paris est la capitale</ref>.', "Paris, sans citation."
        p = {"pid": "p", "citation_style": "ref", "citation_placement": "inline", "messages": [{"content": self.system}]}
        ss = {k: {"k": k, "text": t, "finish_reason": "stop"} for k, t in ((-1, ok), (0, bad), (1, bad + " Et plus encore."))}
        good = {"refused": False, "mixed": False, "answer_correct": True, "answer_supported": True,
                "reasoning_supported": True, "citation_supported": True, "citation_style_ok": True,
                "support_quote_in_context": True}
        st = __import__("collections").Counter()
        # the citation_ok sample not judged yet: wait; judged faulty: citfail useless; judged good: greedy first
        self.assertEqual(T.needed_citfail(p, ss, {}, {}, ok, st), [])
        wrong = {("p", -1): {"judge": {**good, "answer_correct": False}}}
        self.assertEqual(T.needed_citfail(p, ss, wrong, {}, ok, st), [])
        right = {("p", -1): {"judge": good}}
        self.assertEqual(T.needed_citfail(p, ss, right, {}, bad, st), [{"pid": "p", "k": 0}])
        # samples of the same rank: the closest length to the chosen first, as build_p10_delta.py picks the rejected
        self.assertEqual(T.needed_citfail(p, ss, right, {}, ok, st), [{"pid": "p", "k": 1}])
        # the first one judged wrong: the next one; judged right: the pair has its rejected, nothing more
        self.assertEqual(T.needed_citfail(p, ss, right, {("p", 0): {"judge": {**good, "answer_correct": False}}}, bad, st),
                         [{"pid": "p", "k": 1}])
        self.assertEqual(T.needed_citfail(p, ss, right, {("p", 0): {"judge": good}}, bad, st), [])

    def test_release_rules_on_french_ref_chosen(self):
        O = self.load("release_text")
        self.assertTrue(O.quotes_found('A <ref id="1">Paris est la capitale</ref>.', "ref", self.system))
        self.assertFalse(O.quotes_found('A <ref id="1">Paris est la plus belle</ref>.', "ref", self.system))
        text = ("D'après les documents fournis, Paris est la capitale. Je vous invite à reformuler votre question "
                "ou à ajouter des documents pertinents.")
        self.assertEqual(O.normalize(text, True), "Paris est la capitale.")
        self.assertEqual(O.normalize("Selon Human Rights Watch, X.", True), "Selon Human Rights Watch, X.")
        english = "Based on the provided documents, Paris is the capital. Please rephrase the query."
        self.assertEqual(O.normalize(english, True), "Paris is the capital.")

    def test_gold_columns_of_an_sft_run_dataset(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        G = self.load("gold_columns")
        with tempfile.TemporaryDirectory() as tmp:
            sft = Path(tmp) / "sft"
            sft.mkdir()
            P.write(sft / "hotpotqa_fr_converted.jsonl", [{"id": "q1", "answer": "Paris"}])
            d = Path(tmp) / "run" / "qwen35_35b_a3b" / "hotpotqa_fr"
            d.mkdir(parents=True)
            P.write(d / "prompts.jsonl", [{"pid": "p", "qid": "q1", "sft_run": str(sft)}])
            f = Path(tmp) / "t.parquet"
            pq.write_table(pa.Table.from_pylist([{"pid": "p", "qid": "q1"}]), f)
            G.add_to_parquet(f, "hotpotqa_fr", Path(tmp) / "run")
            self.assertEqual(pq.read_table(f).to_pylist(), [{"pid": "p", "qid": "q1", "gold_answer": "Paris",
                                                              "gold_scale": None}])
            with self.assertRaisesRegex(ValueError, "run delta"):
                G.lookup("hotpotqa_fr")

    def test_markup_prompts_are_unchanged(self):
        system = "Here are the retrieved documents: `* filename: A\n\n[CHUNK_START]\nx\n[CHUNK_END]\n* filename: B\n"
        self.assertEqual(self.C.context_titles(system), ["B"])  # historical behaviour of the markup prompts
        m = self.C.citation_metrics('x ##Cite "B"##', ["B"], ["B"], "default")
        self.assertNotIn("n_ref_tags", m)


class SftRunPromptsTests(unittest.TestCase):
    """source sft_run: questions of an SFT run left out of its SFT data -> SFT-format prompts without the invitation
    clause, in both languages (the real extract / format scripts, no LLM)."""

    @classmethod
    def setUpClass(cls):
        # a fresh module: other tests replace PP.run with a fake
        spec = importlib.util.spec_from_file_location(
            "dpo_prompts_pipeline_sft", Path(__file__).resolve().parents[1] / "dpo_prompts_pipeline.py")
        cls.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.M)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sft_run(self, citation_format):
        def context(titles):
            return "\n\n".join(f"* filename: {t}\n\n[CHUNK_START]\n{t} was founded in 1901 by a sailor.\n[CHUNK_END]"
                               for t in titles)
        rows = []
        for q in ("q1", "q2"):
            common = {"type": "bridge", "level": "easy", "citation_format": citation_format,
                      "citation_end_of_completion": False}
            rows.append({**common, "id": q, "question": f"When was {q} founded?", "answer": "1901",
                         "context": context([f"{q} Club", "Other", "Third"]), "reasoning_trace": "In 1901.",
                         "is_unanswerable": False, "grounded_refusal": False, "eval_factual_judge_score": 5,
                         "supporting_facts_titles": [f"{q} Club"], "chunks_relevant": 1, "chunks_total": 3})
            rows.append({**common, "id": f"{q}_unanswerable", "question": f"When was {q} founded?", "answer": "",
                         "context": context(["Other", "Third", "Fourth"]), "reasoning_trace": "No document says.",
                         "is_unanswerable": True, "grounded_refusal": True, "eval_model_refused": True,
                         "supporting_facts_titles": [], "chunks_relevant": 0, "chunks_total": 3})
        run = self.root / "run"
        run.mkdir()
        P.write(run / "x_augmented_evaluated.jsonl", rows)
        P.write(run / "x_augmented_evaluated_filtered.jsonl", rows[2:3])  # q2 reached the SFT data
        return run

    def test_held_out_questions_in_sft_format_without_invitation_clause(self):
        for language, citation_format in (("en", None), ("fr", "ref")):
            with self.subTest(language=language):
                out = self.root / language
                c = {"source": "sft_run", "sft_run": str(self.sft_run(citation_format)), "sft_dataset": "hp_test",
                     "sft_language": language, "seed": 42}
                self.M.sft_prompts(c, out)
                ps = P.rows(out / "sft_pool/prompts.jsonl")
                self.assertEqual({p["qid"] for p in ps}, {"q1"})
                self.assertEqual(sorted(p["is_unanswerable"] for p in ps), [False, True])
                clause = self.M.invite_clause(language)
                for p in ps:
                    system = p["messages"][0]["content"]
                    self.assertNotIn(clause, system)
                    self.assertNotIn(self.M.INVITE_MARKER[language], system)
                    self.assertEqual(p["dataset"], "hp_test")
                    self.assertEqual(p["citation_style"] == "ref", citation_format == "ref")
                shutil.rmtree(self.root / "run")
