# Model Card for Luciole-1.1-1B-Base-2603

<!-- inspired from the following template:
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md?plain=1
-->

* [Model Description](#model-description)
* [Uses](#uses)
* [Bias, Risks, and Limitations](#disclaimer)
* [Example Code in Python](#example-code-in-python)
  * [Load the model](#load-the-model)
  * [Sentence completion](#sentence-completion)
  * [Load a checkpoint](#load-a-checkpoint)
* [Training Details](#training-details)
  * [Training Data](#training-data)
  * [Training Procedure](#training-procedure)
    * [Neural Network Architecture](#neural-network-architecture)
    * [Training Hyperparameters](#training-hyperparameters)
      1. [Main Pre-training](#1-main-pre-training)
      2. [Context Length Extension](#2-context-extension)
      3. [Annealing](#3-annealing)
  * [Training Logs and Learning Curves](#training-logs-and-learning-curves)
<!-- * [Evaluation](#evaluation) -->
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Model Description

Luciole-1.1-1B-Base is a pretrained 1B parameter causal language model developed by [LINAGORA](https://labs.linagora.com/) and [OpenLLM-France](https://github.com/OpenLLM-France) and released under an [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

Luciole-1.1-1B-Base was trained on 5 trillion tokens of multilingual data, including English (), French (), German (), Spanish (), Italian (), Portuguese (),  Arabic (),  Dutch (), a small subset of regional languages including regional languages of the French metropolitan area, French variants (Walloon), and French creoles from around the world (),*
and parallel data from a selection of languages (2.5%),
as well as several programming languages (14.7%).

Training and data preparation code can be found in the [Luciole-Training]() repository.  


The technical report is coming soon. 

*Languages selected from the [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset: Basque, Breton, Catalan, Corsican, Franco-Provençal, Guadeloupean Creole French, Guianese Creole French, Occitan, Picard, Réunion Creole French, Saint Lucian Creole French, Seselwa Creole French, Tahitian, Walloon

## Uses

### Direct use
Luciole-1.1-1B-Base is a foundation language model trained solely to predict the most probable next word in a sequence. It is designed as the first brick in a more complex training pipeline that would include multitask training on diverse instructions or focused fine-tuning on select downstream tasks, as well as possible alignment for human preferences.

### Downstream use 
Due to its multilingual training, Luciole-1.1-1B-Base can be fine-tuned for downstream tasks centered on the generation of multilingual text, with a special focus on French and English. 

### Out-of-Scope Use
Luciole-1.1-1B-Base is not intended to generate text directly for end use cases. It must be fine-tuned first. Its pretraining is optimized for multilingual performance, especially in French and English, and might perform less well on other languages without additional training. While trained on code data, it is not optimized for code generation tasks.

## Bias, Risks, and Limitations

Like other foundation models, Luciole-1.1-1B-Base is trained on large amounts of web data. Additionally, due to the scarcity of French textual non-web data published under open licenses, much of our French data comes from older works in the public domain that carry biases from other time periods. While we made efforts to reduce toxic and offensive content in the [Luciole Training Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset), Luciole-1.1-1B-Base may still generate such content. Filtering of the Luciole Training Dataset is an ongoing project to which we welcome contributions. 

### Recommendations
To limit the generation of undesirable content, it is advised to fine-tune Luciole-1.1-1B-Base through instruction and preference tuning (DPO, RLHF, etc.).

## Example Code in Python

### Load the model

Load the model (quantized version on GPU if possible, for efficient inference):
```python
import transformers

model_name = "OpenLLM-France/Luciole-1.1-1B-Base"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
    device_map="auto",
    load_in_4bit=True       # For efficient inference, if quantization is supported by the GPU card
)
```
### Sentence completion

$works for me but I get warning messages$

Wrap the model in a text generation pipeline, and specify some generation parameters:
```
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

generation_kwargs = dict(
    num_return_sequences=1,               # Number of variants to generate.
    return_full_text= False,              # Do not include the prompt in the generated text.
    do_sample=True,
    temperature=1.0, top_p=1, top_k=None, # Sampling parameters.
    max_new_tokens=200,                   # Maximum length for the output text (in number of tokens).
)
```

Try 1-shot question answering:
```python
prompt = """\
Quelle est la capitale de l'Espagne ? Madrid\n\
Quelle est la capitale de la France ?\
"""
completions = pipeline(prompt, **generation_kwargs)
for completion in completions:
    print(prompt + " […]" + completion['generated_text'])
```
This will print something like:
```
Quelle est la capitale de l'Espagne ? Madrid
Quelle est la capitale de la France ? […] Paris
Quelle est la capitale de l'Italie? Rome
Quelle est la capitale de la Grande-Bretagne? Londres
Quelle est la capitale de la Suisse? Berne
Quelle est la capitale du Portugal? Lisbonne
Quelle est la capitale de l'Algérie? Alger
...
```

If running on GPU (`cuda` device), you will need at least 6GB of VRAM to run inference using 4bit quantization (16GB of VRAM without 4bit quantization).

### Intermediate checkpoints

Checkpoints at several training steps are available under revision tags,
every 5000 steps during the first 30000 steps, and then every 10000 steps.

Intermediate checkpoints can be loaded using the `revision` parameter:
```python
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
    revision="step0753851",
    ...
)
```
where `revision` can be one of:
* "[`step0005000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0005000)", "[`step0010000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0010000)", "[`step0015000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0015000)", "[`step0020000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0020000)": every 5000 steps for the first pre-training steps (with a context length of 4096).
* "[`step0025000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0025000)", "[`step0050000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0050000)", "[`step0075000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0075000)", "[`step0100000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0100000)", ..., "[`step0750000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0750000)": every 25000 steps from 25k to 750k steps.
* "[`step0753851`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/step0753851)": last pre-training step before context length extension and annealing.
* "[`extension_step0000250`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/extension_step0000250)", "[`extension_step0000500`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/extension_step0000500)", "[`extension_step0000750`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/extension_step0000750)", "[`extension_step0001000`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/extension_step0001000)", "[`extension_step0001220`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/extension_step0001220)": several checkpoints during context length extension (with a context length of 32000).

## Training Details

### Training Data

The training dataset used for the pretraining of Luciole-1.1-1B-Base is available
at [OpenLLM-France/Luciole-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset).
<!-- and described in ["The Lucie Training Dataset" (2024/12)](https://arxiv.org/abs/xxxx.xxxxx). -->

The initial composition of the training data is as follows:

![Initial Data Composition](figures/pie_dataset_composition.png)

Pretraining consisted of three principal phases of training with a context length of 4,096 tokens. The token breakdowns for the three phases are as follows:

1. 3.5 trillion tokens of diverse data
2. 1 trillion tokens introducing higher quality data and increasing math and code proportions
3. 0.5 trillion tokens introducing some instruction-style and reasoning data

Pretraining was followed by two short mid-training phases to extend the context length to 131,000 tokens:

4. 25 billion tokens to extend context length from 4,096 to 32,000 tokens 
5. 25 billion tokens to extend context length from 32,000 to 65,000 tokens

This yields the following distributions.

![Training Data Composition](figures/pie_dataset_composition_training.png)

### Training Procedure 

Luciole-1.1-1B-Base is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

It was pre-trained on 128 - 256 H100 80GB GPUs (32 - 64 nodes) for about 41,962 GPU hours (253 hours) on the [Jean Zay supercomputer](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html).

The training code is available at [https://github.com/OpenLLM-France/Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training). Training used version 2.3.1 of NVIDIA's [NeMo framework](https://github.com/NVIDIA-NeMo/NeMo).


<!-- Optimizer checkpoints are available at [OpenLLM-France/Lucie-7B-optimizer-states](https://huggingface.co/OpenLLM-France/Lucie-7B-optimizer-states). -->

#### Neural Network Architecture

The architecture of Luciole-1.1-1B-Base is a custom adaptation of the [Nemotron3-4B](https://github.com/NVIDIA-NeMo/NeMo/blob/0b1be8d1165f49ee2ef1e74f72f2ff07350f6798/nemo/collections/llm/recipes/nemotron3_4b.py) recipe.
It has exactly 1.3 billion free parameters,
with the following hyperparameters:
| **Hyperparameter**        | **Value** |
|---------------------------|---------|
| Vocabulary size (\# tokens)| 128,000 |
| \# transformer blocks     |      24 |
| \# attention heads        |      32 |
| \# key-value heads        |       8 |
| Hidden size               |   2048 |
| Feed-Forward hidden size  |  8192 |
| Activation                |  `relu2`|


The "theta" parameter of Rotary Positional Embedding (RoPE) was increased during the context extension phases training process. Its values are indicated in the tables with training hyperparameters below.

#### Training Hyperparameters

The training consisted of three main phases:
1. Main pre-training on 3.1T tokens, with a context length of 4096,
2. Context extension on 5B tokens, with a context length of 32000,
3. Annealing on 5B tokens of high quality data composed of a mixture of new data and data seen during training.
<!-- perhaps cite the dataset for annealing  -->

The details of each phase are given below.

##### 1. Main Pre-training

Training hyperparameters in torch/Megatron-DeepSpeed were as follows:
| **Hyperparameter**     | **Value**  |
|------------------------|------------|
| Total \# samples| 762 144 586 (3.1T tokens) |
| Total \# steps  | 715,786 + 382,455 + 118,237 |
| RoPE theta             | 10,000    |
| Context length         | 4,096      |
| Initial Batch size     | 256        |
| Final Batch size       | 1 024      |
| Batch size rampup      | by steps of 64 over 10M samples |
| Learning rate schedule | warmup (2M samples) + cosine annealing |
| Maximum Learning rate  | 3e-4       |
| Final Learning rate    | 3e-5       |
| Weight decay           | 0.1        |
| Dropout                | _          |
| Gradient clipping      | 1          |
| Initializer range      | 0.009        |
| Optimizer              | `AdamW` (β₁=0.9, β₂=0.95, ε=1e-5)    |
| Precision              | `bfloat16` |
| Tensor Parallelism (with 512 GPUs)   | 4           |
| Pipeline Parallelism (with 512 GPUs) | 4           |
| Data Parallelism (with 512 GPUs)     | 32          |

#### 2. Context Length Extension

Training hyperparameters are the same as above, with the following changes:
| **Hyperparameter**     | **Value**  |
|------------------------|------------|
| Total \# samples| 156 250 (5B tokens) |
| Total \# steps  | 1 220      |
| RoPE theta             | 20 000 000 |
| Context length         | 32 000     |
| Batch size             | 128        |
| Learning rate          | 2e-5       |
| Learning rate schedule | constant   |
| Tensor Parallelism (with 128 GPUs)   | 4     |
| Pipeline Parallelism (with 128 GPUs) | 4     |
| Data Parallelism (with 128 GPUs)     | 8     |

#### 3. Annealing

Training hyperparameters are the same as for context length extension, with the following changes:
| **Hyperparameter**     | **Value**  |
|------------------------|------------|
| Total \# samples| 156 250 (5B tokens) |
| Total \# steps  | 1 220      |
| Learning rate schedule | linear annealing |
| Maximum Learning rate  | 3e-5       |
| Final Learning rate    | 0          |

### Training Logs and Learning Curves

#### Training loss

Training logs can be found in Tensorboard format in:
* [`metadata/training_logs/`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/main/metadata/training_logs)
<br> ├── [`1_pretraining.zip`](metadata/training_logs/1_pretraining.zip) training logs for the first pre-training phases,
in a zip file. Each file in the zip corresponds to a job of at most 20H of training (parallelized over 512 GPUs).
<br> ├── [`2_extension/`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/main/metadata/training_logs/2_extension) folder containing the training log <br> └── [`3_annealing/`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/main/metadata/training_logs/3_annealing) folder containing the training log for the annealing phase, which also took around 13H of training (parallelized over 128 GPUs).

The convergence curves of the three pre-training phases are the following:

![figures/convergence-curve-pretraining.png](figures/convergence-curve-pretraining.png)

Data corresponding to these plots were extracted from tensorboard logs and are available in the following CSV files:
* [`metadata/training_logs/`](https://huggingface.co/OpenLLM-France/Lucie-7B/tree/main/metadata/training_logs)
<br> ├── [`1_pretraining.csv`](metadata/training_logs/1_pretraining.csv)
<br> ├── [`2_extension.csv`](metadata/training_logs/2_extension.csv)
<br> └── [`3_annealing.csv`](metadata/training_logs/3_annealing.csv)

#### Evaluations

Multiple evaluations were conducted during Luciole-1.1-1B-Base's training to assess its performance on standard benchmarks,
primarily in French and English, as well as in Spanish, German, and Italian.

Evaluation results on benchmark datasets of checkpoints of Luciole-1.1-1B-Base throughout the training process are available at
[metadata/evaluation_learning_curve_lucie.csv](metadata/evaluation_learning_curve_lucie.csv).
Evaluation results of baseline models on the same benchmark datasets are available at
[metadata/evaluation_baselines.csv](metadata/evaluation_baselines.csv).

Main results are summarized in the following figures:

### French
![figures/learning-curve-evaluation-french-bench.png](figures/learning-curve-evaluation-french-bench.png)

### English
![figures/learning-curve-evaluation-benchmarks-in-english.png](figures/learning-curve-evaluation-benchmarks-in-english.png)

### other
![figures/learning-curve-evaluation-multilingual-arc-benchmark.png](figures/learning-curve-evaluation-multilingual-arc-benchmark.png)

### Needle in a Haystack

#### Pretraining
![figures/needle-in-a-haystack/Lucie-7B-main.png](figures/needle-in-a-haystack/Lucie-7B-main.png) 

#### Context Length Extension
![figures/needle-in-a-haystack/Lucie-7B-extension.png](figures/needle-in-a-haystack/Lucie-7B-extension.png) 

#### Annealing
![figures/needle-in-a-haystack/Lucie-7B-annealing.png](figures/needle-in-a-haystack/Lucie-7B-annealing.png) 




## Citation

✍ Paper coming soon!


## Acknowledgements

### BPI France

We gratefully acknowledge BPI France for funding the OpenLLM France project under the call "Communs numériques pour l’intelligence artificielle générative" ("Digital commons for generative artificial intelligence") and the project numbers DOS0250771 and DOS0250773.

Training of Luciole-1.1-1B-Base was made possible by computing AI and storage resources by GENCI at IDRIS thanks to the grant 2024-GC011015444 on the supercomputer Jean Zay’s H100 partition. We gratefully acknowledge support from GENCI and IDRIS and from Stephane Requena (GENCI) and Pierre-François Lavallée (IDRIS) in particular. 

Luciole-1.1-1B-Base was created by members of [LINAGORA](https://labs.linagora.com/) for the OpenLLM-France project, including in alphabetical order:  

Audran Bert  
Akshay Chaturvedi  
Olivier Gouvert  
Julie Hunter  
Jean-Pierre Lorré  
Jérôme Louradour  
Charlotte Noel   
Kate Thompson   

We thank the support teams from IDRIS and NVIDIA for their technical guidance throughout the project, especially:  

Meriem Bendris (NVIDIA)  
Martin Comminges (IDRIS)  
Rémi Lacroix (IDRIS)     
Myriam Peyrounette (IDRIS)  
Hayk Shoukourian (NVIDIA)  
Oleg Sudakov (NVIDIA)  

We are also greatful to the partners of the [OpenLLM-France](https://www.openllm-france.fr/) consortium for their valuable input, with particular thanks to (in alphabetical order):  

Pascal Alix (Sorbonne),  
Clément Bénesse (Opsci),    
Bertrand Cabot (IDRIS),  
Christophe Cerisara (LORIA),    
Liam Duignan (CEA),   
Olivier Ferret (CEA),    
Emile Hazard (OpSci),  
Léo Hunout (IDRIS),  
Gabriel Lauzzana (LORIA),      
Michel-Marie Maudet (LINAGORA),  
Celia Zolynski (Sorbonne)

We would also like to thank Djamé Seddah and the GAPERON team for sharing their insights with us.

Finally, we thank the entire OpenLLM-France community, whose members have helped in diverse ways. 

## Contact

contact@openllm-france.fr
