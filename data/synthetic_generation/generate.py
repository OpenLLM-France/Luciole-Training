import datasets 
from distilabel.models import TransformersLLM
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
import os

with Pipeline() as pipeline: # 
    TextGeneration( # 
        llm=vLLM(
            model="Qwen/Qwen3-0.6B",
            # generation_kwargs={"temperature": 0.7, "max_new_tokens": 512},
        ),
    )

if __name__ == "__main__":
    with open('prompt/fr.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()

    dataset = datasets.load_dataset(
        os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/raw_datasets/fineweb2/data/fra_Latn/train")
        ) 
    dataset = dataset.map(
        lambda x: {"instruction": prompt.replace('<extrait>', x["text"])},
        remove_columns=dataset["train"].column_names,
    )
    distiset = pipeline.run(dataset=dataset["train"].select(range(2)))
    distiset.save_to_disk(
        "test-dataset",
        save_card=True,
        save_pipeline_config=True,
        save_pipeline_log=True
    )
    