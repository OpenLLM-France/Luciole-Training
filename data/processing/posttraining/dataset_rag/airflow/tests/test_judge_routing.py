"""Judge routing defaults and authenticated SSH readiness probes, without LLM calls."""
import ast
import importlib.util
import os
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]


class JudgeRoutingTests(unittest.TestCase):
    def test_environment_model_used_by_evaluator_stream_and_helpers(self):
        code = '''
import contextlib, inspect, io, json, tempfile
from pathlib import Path
from unittest.mock import patch
import utils, evaluate_and_filter as E, evaluate_stream as S
assert inspect.signature(E.evaluate).parameters['judge_model'].default == 'review-judge'
for fn in (utils.run_llm_judge, utils.run_factual_judge, utils.run_french_quality_judge, utils.run_refusal_judge):
    assert inspect.signature(fn).parameters['model'].default == 'review-judge'
assert inspect.signature(utils.call_llm).parameters['model'].default == 'review-generator'
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / 'rows.jsonl'
    p.write_text(json.dumps({'id':'q','reasoning_trace':'answer'})+'\\n')
    p.with_suffix('.done').touch()
    with patch('sys.argv', ['stream','--input',str(p),'--judge-cache',str(Path(tmp)/'cache'),'--llm-judge-factual']), patch.object(S,'judge_rows') as judge, contextlib.redirect_stdout(io.StringIO()):
        S.main()
        assert judge.call_args.kwargs['model'] == 'review-judge'
'''
        subprocess.run([sys.executable, "-c", code], cwd=BASE,
                       env={**os.environ, "LLM_JUDGE_MODEL": "review-judge", "LLM_MODEL": "review-generator"}, check=True)

    def probe(self):
        spec = importlib.util.spec_from_file_location("check_judge_server", BASE / "airflow/check_judge_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_readiness_sends_bearer_key(self):
        probe = self.probe()
        with patch.object(probe, "JUDGE_API_KEY", "test-key"), patch.object(probe.urllib.request, "urlopen") as call:
            self.assertTrue(probe.ready("http://localhost:8030/v1/models"))
            request = call.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(call.call_args.kwargs["timeout"], 5)

    def test_keyless_probe_and_authentication_failure(self):
        probe = self.probe()
        with patch.object(probe, "JUDGE_API_KEY", None), patch.object(probe.urllib.request, "urlopen") as call:
            self.assertTrue(probe.ready("http://localhost/v1/models"))
            self.assertIsNone(call.call_args.args[0].get_header("Authorization"))
            call.side_effect = urllib.error.HTTPError("http://localhost", 401, "Unauthorized", {}, None)
            self.assertFalse(probe.ready("http://localhost/v1/models"))

    def test_ssh_template_renders_valid_shell(self):
        import jinja2
        tree = ast.parse((BASE / "airflow/dags/dataset_rag_dags.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ENDPOINT_ENV" for t in n.targets))
        template = ast.literal_eval(node.value.func.value).replace("@PYTHON@", sys.executable)
        params = dict(gen_url="", gen_key_file="", judge_key_file="key", judge_model="judge",
                      judge_ssh_host="host", judge_ssh_config="config", judge_remote_port=8030)
        shell = jinja2.Template(template).render(params=params)
        self.assertEqual(shell.count("airflow/check_judge_server.py"), 2)
        subprocess.run(["bash", "-n"], input=shell, text=True, check=True)
