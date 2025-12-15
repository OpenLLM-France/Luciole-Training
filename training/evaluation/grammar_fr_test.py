import random
import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig

# Table pour matcher tes noms à ceux du dataset HF 
split_name = {
    "grammar": "Grammar",
    "vocab": "Vocabulary",
    "reading": "Reading"
}

# Fonction de prompt pour le dataset french-bench-grammar-vocab-reading
def prompt_french_bench_grammar_vocab_reading_fr(line, task_name: str = None):
    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"
    question = line["question"]
    choices = [line["answerA"], line["answerB"], line["answerC"], line["answerD"]]

    # lettre réponse (ex: 'A'), index dans LETTER_INDICES
    answer_letter = line["answer"].strip().upper()
    gold_index = LETTER_INDICES.index(answer_letter)

    query = f"{instruction}Question : {question}\n"
    query += "".join([f"{letter}. {choice}\n" for letter, choice in zip(LETTER_INDICES, choices)])
    query += "Réponse: "

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=instruction,
    )

# Sous-ensembles disponibles dans le dataset
SAMPLE_SUBSETS = ["grammar", "vocab", "reading"]

# Création d’une task par sous-ensemble avec les bons splits
french_bench_grammar_vocab_reading_fr_task = [
    LightevalTaskConfig(
        name=f"french_bench_grammar_vocab_reading_fr_{subset}",
        suite=["community"],
        prompt_function=prompt_french_bench_grammar_vocab_reading_fr,
        hf_repo="manu/french-bench-grammar-vocab-reading",
        hf_subset="default",
        hf_avail_splits=[split_name[subset]],       
        evaluation_splits=[split_name[subset]],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metric=[Metrics.loglikelihood_acc],
        stop_sequence=[""],
        trust_dataset=True,
        version=0,
    )
    for subset in SAMPLE_SUBSETS
]

# Liste des tâches exportées
TASKS_TABLE = french_bench_grammar_vocab_reading_fr_task