from pathlib import Path
import json
import pandas as pd
import re
import numpy as np
import glob


def _nan_to_zero(value):
    """Return 0.0 for a missing or NaN value, otherwise the value unchanged."""
    if value is None or np.isnan(value):
        return 0.0
    return value


# Derived metrics. A metric can be given in task_group_mapping either as a string
# (read directly from the results JSON) or as a function. A function takes one task's
# {metric_name: value} dict and returns the derived value. Its __name__ is the metric
# identifier used everywhere downstream. It is only computed for the tasks it is paired
# with in task_group_mapping, so it may assume those tasks expose the metrics it reads.
def refusal_f1(metrics):
    """refusal_recall * refusal_precision, treating NaN components as 0."""
    assert "refusal_recall" in metrics and "refusal_precision" in metrics
    return _nan_to_zero(metrics["refusal_recall"]) * _nan_to_zero(
        metrics["refusal_precision"]
    )


# Mapping from a group name to the list of (task, metric) pairs it contains.
# Task names follow lighteval's "suite|task|fewshot" convention. This lives here
# (rather than in plot_results) so it can also serve as the registry of known
# benchmarks used to restore suite prefixes dropped by newer lighteval versions.
task_group_mapping = {
    # "mcq": [
    #     ("leaderboard|hellaswag|0", "acc"),
    #     ("helm|hellaswag|0", "em_with_normalize_gold&normalize_pred"),
    #     ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
    #     ("lighteval|mlmm_arc_fra_mcf:challenge|0", "acc"),
    #     ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm_token"),
    #     ("lighteval|mlmm_hellaswag_fra_mcf|0", "acc"),
    #     ("custom|mmlu_pro_cf|0", "acc_norm_token"),
    #     ("custom|mmlu_pro_mcf|0", "acc"),
    # ],
    "en": [
        ("lighteval|openbookqa|0", "acc_with_logprob_normalization"),
        ("lighteval|triviaqa|0", "em_with_strip_strings&normalize_pred"),
        ("custom|mmlu_pro_cf|0", "acc_norm_token"),
        ("lighteval|arc:easy|0", "acc_with_logprob_normalization"),
        ("leaderboard|arc:challenge|0", "acc_with_logprob_normalization"),
        ("helm|commonsenseqa|0", "em_with_normalize_gold&normalize_pred"),
        ("helm|siqa|0", "em"),
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|piqa|0", "acc_with_logprob_normalization"),
        ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
        # ("helm|boolq:_average|0", "em_with_type_exact_match"),
    ],
    "smollm": [
        ("custom|piqa_cf|0", "acc_norm"),
        ("lighteval|piqa|0", "acc_with_logprob_normalization"),
        ("custom|hellaswag_cf|0", "acc_norm"),
        ("leaderboard|hellaswag|0", "acc"),
        ("custom|openbookqa_cf|0", "acc_norm"),
        ("lighteval|openbookqa|0", "acc_with_logprob_normalization"),
        ("custom|commonsenseqa_cf|0", "acc_norm"),
        ("helm|commonsenseqa|0", "em_with_normalize_gold&normalize_pred"),
        ("custom|boolq_cf|0", "acc_norm"),
        ("helm|boolq:_average|0", "em_with_type_exact_match"),
        ("custom|arc_cf:challenge|0", "acc_norm"),
        ("leaderboard|arc:challenge|0", "acc_with_logprob_normalization"),
        ("custom|arc_cf:easy|0", "acc_norm"),
        ("lighteval|arc:easy|0", "acc_with_logprob_normalization"),
        ("custom|winogrande_cf|0", "acc_norm"),
        ("leaderboard|winogrande|0", "acc"),
        ("custom|gsm8k|5", "extractive_match"),
        ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
    ],
    "cultural": [
        ("lighteval|global_mmlu_cs_eng_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_ca_eng_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_cs_fra_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_ca_fra_cf:_average|0", "acc_norm"),
    ],
    "idiomatic_expressions": [
        ("custom:idiomatic_expressions_fib_context:_average:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:different:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:similar:0", "acc"),
        ("custom:idiomatic_expressions_fib_context:word_by_word:0", "acc"),
    ],
    "mmlu": [
        ("custom|mmlu_pro_cf|0", "acc_norm"),
        ("custom|mmlu_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_eng_cf:_average|0", "acc_norm"),
    ],
    "en_new": [
        ("lighteval|belebele_eng_Latn_cf|0", "acc_norm"),
        ("lighteval|global_mmlu_all_eng_cf:_average|0", "acc_norm"),
    ],
    "fr": [
        ("lighteval|fquadv2_fra|0", "exact_match_fra_prefix"),  # f1_fra ?
        ("lighteval|mintaka_fra|0", "exact_match_fra_prefix"),  # f1_fra ?
        ("lighteval|global_mmlu_all_fra_cf:_average|0", "acc_norm"),
        ("lighteval|belebele_fra_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm_token"),
        ("lighteval|xcodah_fra_cf|0", "acc_norm"),
        ("lighteval|xcsqa_fra_cf|0", "acc_norm_token"),
        ("lighteval|xnli2.0_fra_cf|0", "acc_norm_token"),
    ],
    "multilingual": [
        ("lighteval|global_mmlu_all_deu_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_spa_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_ita_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_ara_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_por_cf:_average|0", "acc_norm"),
        ("lighteval|global_mmlu_all_nld_cf:_average|0", "acc_norm"),
        # ("lighteval|belebele_deu_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_spa_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_ita_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_arb_Arab_cf|0", "acc_norm"),
        # ("lighteval|belebele_por_Latn_cf|0", "acc_norm"),
        # ("lighteval|belebele_nld_Latn_cf|0", "acc_norm"),
        ("lighteval|mlmm_hellaswag_deu_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_spa_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_ita_cf|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_ara_cf|0", "acc_norm_token"),
    ],
    "translation": [
        # ("lighteval|flores200:fra_Latn-eng_Latn|5", "bleu"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "bleu_4"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "comet"),
        ("lighteval|flores200:fra_Latn-eng_Latn|5", "metricx"),
        # ("lighteval|flores200:eng_Latn-fra_Latn|5", "bleu"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "bleu_4"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "comet"),
        ("lighteval|flores200:eng_Latn-fra_Latn|5", "metricx"),
    ],
    "ruler": [
        ("custom|ruler_4096:_average|0", "ruler_match"),
        ("custom|ruler_8192:_average|0", "ruler_match"),
        ("custom|ruler_16384:_average|0", "ruler_match"),
        ("custom|ruler_32768:_average|0", "ruler_match"),
        ("custom|ruler_65536:_average|0", "ruler_match"),
        ("custom|ruler_131072:_average|0", "ruler_match"),
    ],
    "finetune": [
        ("leaderboard|hellaswag|0", "acc"),
        ("leaderboard|winogrande|0", "acc"),
        ("lighteval|mlmm_arc_fra_cf:challenge|0", "acc_norm_token"),
        ("lighteval|mlmm_hellaswag_fra_cf|0", "acc_norm_token"),
        ("custom|mmlu_pro_cf|0", "acc_norm_token"),
        # ("lighteval|gpqa:diamond|0", "gpqa_pass@k_with_k"),
        # ("community|gpqa-fr|0", "acc"),
        # ("leaderboard|gsm8k|5", "em_with_normalize_gold&normalize_pred"),
        ("lighteval|gsm8k|0", "extractive_match"),
        ("lighteval|gsm_plus|0", "extractive_match"),
        # ("lighteval|aime25|0", "pass@k_with_k&n"),
        ("extended|lcb:codegeneration|0", "codegen_pass@1:16"),
        ("extended|ifeval|0", "prompt_level_loose_acc"),
        ("community|ifeval-fr|0", "prompt_level_loose_acc"),
        ("extended|ifbench_test|0", "prompt_level_loose_acc"),
        ("extended|ifbench_multiturn|0", "prompt_level_loose_acc"),
        ("extended|mixeval_easy:_average|0", "judge_score_flow"),
        ("extended|mixeval_hard:_average|0", "judge_score_flow"),
        # ("custom|hotpotqa_hotpotqa|0", "qa_f1_score"),
        # ("custom|longbench_hotpotqa|0", "qa_f1_score"),
        # ("custom|longbench_musique|0", "qa_f1_score"),
        # ("custom|longbench_2wikimqa|0", "qa_f1_score"),
        # ("community|harmbench_standard:_average|0", "safety_rate_llama_guard"),
        # ("community|harmbench_contextual:_average|0", "safety_rate_llama_guard"),
    ],
    "rag": [
        ("community|luciole_rag:hotpotqa|0", "answer_em_fuzzy"),
        ("community|luciole_rag:tatqa|0", "answer_em_fuzzy"),
        ("community|luciole_rag:hotpotqa_fr|0", "answer_em_fuzzy"),
        ("community|luciole_rag:newsquadfr|0", "answer_em_fuzzy"),
        # ("community|luciole_rag:squad2_fr_pragnakalp|0", "answer_em"),
        # ("community|luciole_rag:piaf|0", "answer_em"),
        ("community|luciole_rag:hotpotqa|0", refusal_f1),
        ("community|luciole_rag:tatqa|0", refusal_f1),
        ("community|luciole_rag:hotpotqa_fr|0", refusal_f1),
        ("community|luciole_rag:newsquadfr|0", refusal_f1),
        # ("community|luciole_rag:squad2_fr_pragnakalp|0", refusal_f1),
        # ("community|luciole_rag:piaf|0", refusal_f1),
        # ("community|luciole_rag:hotpotqa|0", "refusal_recall"),
        # ("community|luciole_rag:tatqa|0", "refusal_recall"),
        # ("community|luciole_rag:hotpotqa_fr|0", "refusal_recall"),
        # ("community|luciole_rag:newsquadfr|0", "refusal_recall"),
        # ("community|luciole_rag:hotpotqa|0", "refusal_precision"),
        # ("community|luciole_rag:tatqa|0", "refusal_precision"),
        # ("community|luciole_rag:hotpotqa_fr|0", "refusal_precision"),
        # ("community|luciole_rag:newsquadfr|0", "refusal_precision"),
        ("custom|hotpotqa_hotpotqa|0", "qa_f1_score"),
        ("custom|longbench_hotpotqa|0", "qa_f1_score"),
        ("custom|longbench_musique|0", "qa_f1_score"),
        ("custom|longbench_2wikimqa|0", "qa_f1_score"),
    ],
    "longbench": [
        ("custom|hotpotqa_hotpotqa|0", "qa_f1_score"),
        ("custom|longbench_hotpotqa|0", "qa_f1_score"),
        ("custom|longbench_musique|0", "qa_f1_score"),
        ("custom|longbench_2wikimqa|0", "qa_f1_score"),
        ("custom|hotpotqa_hotpotqa|0", "qa_recall_score"),
        ("custom|longbench_hotpotqa|0", "qa_recall_score"),
        ("custom|longbench_musique|0", "qa_recall_score"),
        ("custom|longbench_2wikimqa|0", "qa_recall_score"),
        ("custom|hotpotqa_hotpotqa|0", "qa_precision_score"),
        ("custom|longbench_hotpotqa|0", "qa_precision_score"),
        ("custom|longbench_musique|0", "qa_precision_score"),
        ("custom|longbench_2wikimqa|0", "qa_precision_score"),
    ],
    "safety": [
        ("community|harmbench_standard:_average|0", "safety_rate_llama_guard"),
        ("community|harmbench_contextual:_average|0", "safety_rate_llama_guard"),
        ("community|advbench|0", "safety_rate_llama_guard"),
        ("community|hexphi:_average|0", "safety_rate_llama_guard"),
        # ("community|aya_red_teaming_eng|0", "safety_rate_llama_guard"),
        # ("community|aya_red_teaming_fra|0", "safety_rate_llama_guard"),
        # ("community|aya_red_teaming_spa|0", "safety_rate_llama_guard"),
        # ("community|aya_red_teaming_ara|0", "safety_rate_llama_guard"),
    ],
}

task_group_mapping["common"] = [
    task
    for task in (
        task_group_mapping["en"]
        + task_group_mapping["fr"]
        + task_group_mapping["multilingual"]
    )
    if task in task_group_mapping["finetune"]
]


# Callable metrics referenced in task_group_mapping, grouped by the (full) task name
# they are paired with. They are computed from a task's other metrics when reading the
# results JSON (see read_json_file).
def _collect_derived_metrics_by_task(mapping):
    by_task = {}
    for group in mapping.values():
        for task, metric in group:
            if callable(metric):
                by_task.setdefault(task, {})[metric.__name__] = metric
    return by_task


_derived_metrics_by_task = _collect_derived_metrics_by_task(task_group_mapping)


def get_step(text):
    match = re.search(r"totalstep[=_-]?(\d+)", text)
    is_global = bool(match)
    if not match:
        match = re.search(r"step[=_-]?(\d+)", text)
    if match:
        step_number = int(match.group(1))
        return is_global, step_number
    raise RuntimeError(f"Could not extract step number from: {text}")


def get_training_tokens_and_model_size(file_path):
    if "OLMo-2-0425-1B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 1.279_395_840
    elif "OLMo-2-1124-7B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 7.0
    elif "OLMo-2-1124-13B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 13.0
    elif "OLMo-2-0325-32B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else None
        model_size = 32.234_279_936
    elif "Apertus-8B" in str(file_path):
        match = re.search(r"-tokens([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 15000
        model_size = 8.0
    elif "Gaperon-1125-1B" in str(file_path):
        tokens = 3000
        model_size = 1.0
    elif "Gaperon-1125-8B" in str(file_path):
        match = re.search(r"_tokens-([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 4110
        model_size = 8.0
    elif "Gaperon-1125-24B" in str(file_path):
        match = re.search(r"_tokens-([0-9.]+)B", str(file_path))
        tokens = float(match.group(1)) if match else 2059
        model_size = 24.0
    elif "EuroLLM-1.7B" in str(file_path):
        tokens = 4000
        model_size = 1.394_706_432
    elif "EuroLLM-9B" in str(file_path):
        tokens = 4000
        model_size = 9.0
    elif "EuroLLM-22B" in str(file_path):
        tokens = 4000
        model_size = 22.0
    elif "salamandra-2b" in str(file_path):
        tokens = 12_875
        model_size = 2
    elif "salamandra-7b" in str(file_path):
        tokens = 12_875
        model_size = 7.768_117_248
    elif "Teuken-7B" in str(file_path):
        tokens = 6_000
        model_size = 7.0
    elif "SmolLM2-1.7B" in str(file_path):
        match = re.search(r"step-([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        if steps is None:
            tokens = 11000
        else:
            tokens = steps * 2 * 1e-3
        model_size = 1.711_376_384
    elif "SmolLM3-3B" in str(file_path):
        tokens = 11200
        model_size = 3.075_098_624
    elif "Lucie-7B" in str(file_path):
        match = re.search(r"step([0-9.]+)", str(file_path))
        steps = float(match.group(1)) if match else None
        if steps:
            if "extension" in str(file_path):
                steps += 753851
            tokens = steps * 4096 * 1024 / 10**9
        else:
            tokens = 3131.7
        model_size = 7.0
    elif "CroissantLLM" in str(file_path):
        tokens = 3000
        model_size = 1.3
    elif "Llama-2-7b" in str(file_path):
        model_size = 6.9
        tokens = 2000
    elif "Llama-3.2-1B" in str(file_path):
        model_size = 1.23
        tokens = 9000
    elif "Llama-3.1-8B" in str(file_path):
        model_size = 8.0
        tokens = 15000
    elif "Mistral-Small" in str(file_path) and "24B" in str(file_path):
        model_size = 24.0
        tokens = 8000
    elif "Mistral-7B" in str(file_path):
        model_size = 7
        tokens = 8000
    elif "Ministral-3" in str(file_path):
        match = re.search(r"Ministral-3-([0-9.]+)B", str(file_path))
        model_size = int(match.group(1))
        tokens = 8000
    elif "Qwen3-" in str(file_path):
        match = re.search(r"Qwen3-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 36000
    elif "Qwen2.5-" in str(file_path):
        match = re.search(r"Qwen2.5-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 18000
    elif "Qwen2-" in str(file_path):
        match = re.search(r"Qwen2-([0-9.]+)B", str(file_path))
        model_size = float(match.group(1))
        tokens = 7000
    elif "Olmo-3" in str(file_path):
        match = re.search(r"Olmo-3-([0-9.]+)B", str(file_path))
        model_size = int(match.group(1))
        tokens = 6000
    elif (
        ("luciol" in str(file_path).lower())
        or ("llama1b" in str(file_path).lower())
        or ("ablation" in str(file_path).lower())
    ):
        if "llama1b" in str(file_path).lower():
            model_size = 1.235290112
        elif "1b" in str(file_path).lower():
            model_size = 1.319309312
        elif "8b" in str(file_path).lower():
            model_size = 8.075686912
        elif "23b" in str(file_path).lower():
            model_size = 23.216467968
        else:
            raise ValueError(f"Unknown model size for model in: {file_path}")

        is_global, steps = get_step(str(file_path))

        if not is_global:
            steps_phase1 = 715787
            steps_phase2 = 358930
            steps_phase3_annealing = 118238 if model_size < 23 else 71526
            steps_extension = 5960 if model_size < 23 else 11920

            if "phase2" in str(file_path):
                steps += steps_phase1
            if "_32k_" in str(file_path) or "_131k_v4_" in str(file_path):
                steps += steps_phase1 + steps_phase2 + steps_phase3_annealing
            elif "_65k_" in str(file_path) or "_131k_" in str(file_path):
                steps += (
                    steps_phase1
                    + steps_phase2
                    + steps_phase3_annealing
                    + steps_extension
                )
            elif "sft" in str(file_path).lower():
                steps += (
                    steps_phase1
                    + steps_phase2
                    + steps_phase3_annealing
                    + 2 * steps_extension
                )
                if model_size > 7.9 and model_size < 8.1:
                    steps += 11921
            else:  # if "annealin" in str(file_path):
                steps += steps_phase1 + steps_phase2

        tokens = steps * 4096 * 1024 / 10**9
    else:
        raise ValueError(
            f"Cannot infer model size / nb of training tokens in file path: {file_path}"
        )
    return tokens, model_size


_task_name_registry = None


def _build_task_name_registry():
    """Collect known benchmark names, indexed by the part that follows the suite, so
    suite-less names (produced by newer lighteval versions) can be matched back.

    Sources are the local `task_group_mapping` ("suite|task|fewshot") and the
    aggregation reference list agg_tasks.jsonl ("suite|task", no fewshot). Both live
    in this package, so no higher-level module needs to be imported.

    Returns (with_fewshot, without_fewshot):
      - with_fewshot:    "task|fewshot" -> "suite|task|fewshot"
      - without_fewshot: "task"         -> "suite|task"   (fewshot re-appended by caller)
    """
    with_fewshot = {}
    without_fewshot = {}

    def register(full):
        n_pipes = full.count("|")
        if n_pipes == 2:
            with_fewshot.setdefault(full.split("|", 1)[1], full)
        elif n_pipes == 1:
            without_fewshot.setdefault(full.split("|", 1)[1], full)

    for tasks in task_group_mapping.values():
        for task, _metric in tasks:
            register(task)

    agg_tasks_path = Path(__file__).parent / "agg_tasks.jsonl"
    with open(agg_tasks_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            register(json.loads(line)["task"])

    return with_fewshot, without_fewshot


def restore_task_suite(task):
    """Restore the suite prefix dropped by newer lighteval versions.

    Newer lighteval dropped the leading suite from result keys, e.g.
    "lighteval|flores200:eng_Latn-fra_Latn|5" became "flores200:eng_Latn-fra_Latn|5".
    Such a name is mapped to the known benchmark of the form
    ".*|flores200:eng_Latn-fra_Latn|5" (here "lighteval|flores200:eng_Latn-fra_Latn|5").
    Names that already carry a suite (old format, two "|") are returned unchanged.
    """
    global _task_name_registry
    if task.count("|") != 1:
        # Already "suite|task|fewshot" (or an unexpected shape) -> leave it as-is.
        return task
    if _task_name_registry is None:
        _task_name_registry = _build_task_name_registry()
    with_fewshot, without_fewshot = _task_name_registry
    if task in with_fewshot:
        return with_fewshot[task]
    name, fewshot = task.rsplit("|", 1)
    if name in without_fewshot:
        return f"{without_fewshot[name]}|{fewshot}"
    return task


def add_derived_metrics(results):
    """Augment each task's {metric: value} dict in-place with the derived metrics it is
    paired with in task_group_mapping, so they can be plotted like any other metric."""
    for task, task_metrics in results.items():
        for name, fn in _derived_metrics_by_task.get(
            restore_task_suite(task), {}
        ).items():
            task_metrics[name] = fn(task_metrics)


def read_json_file(file_path):
    file_path = Path(file_path)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as err:
        raise RuntimeError(f"Could not read JSON file {file_path}") from err

    add_derived_metrics(data["results"])

    df = (
        pd.DataFrame(data["results"])
        .stack()
        .reset_index(name="score")
        .rename(columns={"level_0": "metric", "level_1": "task"})
    )

    # Restore the suite prefix dropped by newer lighteval versions, so the names
    # match the suite-checked benchmarks used downstream.
    df["task"] = df["task"].map(restore_task_suite)

    # Multiply the values of the "score" column by 1/100 when the value of the "metric" column is "comet" or "comet_stderr"
    df.loc[df["metric"].isin(["comet", "comet_stderr"]), "score"] *= 1 / 100

    df["max_samples"] = str(data["config_general"]["max_samples"])

    # Filter out metrics ending in "_stderr"
    # df = df[~df["metric"].str.endswith("_stderr")]

    # mark whether the row is stderr or score
    df["value_type"] = df["metric"].str.endswith("_stderr")

    # normalize metric name (remove _stderr suffix)
    df["metric_base"] = df["metric"].str.replace("_stderr$", "", regex=True)

    # pivot score vs stderr into columns
    df = (
        df.pivot_table(
            index=["metric_base", "task", "max_samples"],
            columns="value_type",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={False: "score", True: "stderr", "metric_base": "metric"})
    )

    # Get training flops
    tokens, num_parameters = get_training_tokens_and_model_size(file_path)
    if not tokens:
        print(f"WARNING: Could not determine training tokens for {file_path}")
    df["tokens"] = tokens
    df["model_size"] = num_parameters
    df["FLOPs"] = df["model_size"] * df["tokens"] * 6 * 1e18

    # Get evaluation timestamp
    df["timestamp"] = pd.to_datetime(
        file_path.stem.replace("results_", ""), format="%Y-%m-%dT%H-%M-%S.%f"
    )
    return df


def read_experiment_results(
    main_dir, evaluation_dir="evaluation", expe_name=None, split_per_tokens=False
):
    print(f"Processing {main_dir}...")
    main_dir = Path(main_dir)

    assert main_dir.is_dir(), f"{main_dir} is not an existing directory"

    if expe_name is None:
        expe_name = main_dir.name

    dataframes = [
        read_json_file(Path(f))
        for f in glob.glob(str(main_dir / "**" / "results_*.json"), recursive=True)
        if evaluation_dir in Path(f).parts and "deprecated" not in f
    ]
    if not dataframes:
        print(f"No valid JSON result files found in {main_dir}")
        return
    df = pd.concat(dataframes, ignore_index=True)
    if split_per_tokens:
        df["expe_name"] = df["tokens"].apply(
            lambda t: f"{expe_name} ({int(t)}B training tokens)"
        )
    else:
        df["expe_name"] = expe_name

    # Remove duplicates
    len_before_dup = len(df)
    df = df.sort_values("timestamp", ascending=False).drop_duplicates(
        subset=df.columns.difference(["timestamp"]), keep="first"
    )
    len_after_dup = len(df)
    if len_before_dup > len_after_dup:
        print(f"Removed {len_before_dup - len_after_dup} duplicate rows")

    print("Example:")
    print(df.iloc[0])
    print("\n")
    return df


def read_datamix(main_dir):
    json_files = list(Path(main_dir).glob("datamix/*.json"))
    if not json_files:
        print(f"No JSON datamix file found in {main_dir}")
        return
    if len(json_files) > 1:
        print(f"More than one JSON datamix file found in {main_dir}")
        return
    json_file = json_files[0]
    with open(json_file, "r") as f:
        data = json.load(f)
    datamix = data["train"]
    return datamix


def compute_regression(group):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    group = group[group["tokens"] > 0]  # adjust to your x column name

    group = group.sort_values("tokens")
    x = group["tokens"].values.reshape(-1, 1)
    y = group["score"].values

    tokens = x.flatten().tolist()
    score = y.tolist()

    if len(group) < 2:
        return pd.DataFrame(
            [{"slope": np.nan, "intercept": np.nan, "tokens": tokens, "score": score}]
        )

    model = LinearRegression()
    model.fit(np.log(x), y)
    y_pred = model.predict(np.log(x))

    return pd.DataFrame(
        [
            {
                "slope": model.coef_[0],
                "intercept": model.intercept_,
                "r2": r2_score(y, y_pred),
                "tokens": tokens,
                "score": score,
            }
        ]
    )


def moving_average(values, window=5):
    values = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")  # only valid positions
    return smoothed


def process_group(group, window=1):
    group = group.loc[group["tokens"] > 0].sort_values("tokens")

    tokens = group["tokens"].to_numpy()
    flops = group["FLOPs"].to_numpy()
    scores = group["score"].to_numpy()
    stderr = group["stderr"].to_numpy()

    if window < 2 or len(scores) < window:
        scores = scores
    else:
        scores = moving_average(scores, window=window)
        pad = window // 2
        tokens = tokens[pad : -pad or None]
        flops = flops[pad : -pad or None]

    return pd.DataFrame(
        [
            {
                "tokens": tokens.tolist(),
                "FLOPs": flops.tolist(),
                "score": scores.tolist(),
                "stderr": stderr.tolist(),
            }
        ]
    )


def process_results(df, window=1, fit=False):
    if fit:
        group_df = (
            df.groupby(["task", "max_samples", "metric", "expe_name"])
            .apply(compute_regression)
            .reset_index()
        )
        return group_df
    else:
        group_df = (
            df.groupby(["task", "max_samples", "metric", "expe_name"])
            .apply(lambda x: process_group(x, window=window), include_groups=False)
            .reset_index()
        )
        return group_df


def format_task_for_title(task):
    f = task.split("|")
    if len(f) in [2, 3]:
        task = f[1]
    if task.endswith("_cf"):
        task = task[:-3]
    task = (
        task.replace("_all_", "_")
        .replace("mmlu", "MMLU")
        .replace("arc", "ARC")
        .replace("hellaswag", "HellaSwag")
        .replace("winogrande", "Winogrande")
        .replace("gsm8k", "GSM8K")
        .replace("boolq", "BoolQ")
        .replace("commonsenseqa", "CommonsenseQA")
        .replace("belebele", "Belebele")
        .replace("siqa", "SIQA")
        .replace("openbookqa", "OpenBookQA")
        .replace("piqa", "PIQA")
        .replace("triviaqa", "TriviaQA")
        .replace("mintaka", "Mintaka")
        .replace("fquadv2", "FQuADv2")
        .replace("xcodah", "XCODAH")
        .replace("xcsqa", "XCSQA")
        .replace("xnli", "XNLI")
        .replace("mlmm", "MLMM")
        .replace("flores200", "FLORES")
        .replace("cwe", "CWE")
        .replace("fwe", "FWE")
        .replace("niah_", "NIAH_")
        .replace("qa_", "QA_")
        .replace("vt", "VT")
        .replace("_Latn", "")
        .replace("_cf", "")
        .replace(":_average", "")
        .replace("_", " ")
        .replace(":", " ")
        .strip()
    )
    return task
