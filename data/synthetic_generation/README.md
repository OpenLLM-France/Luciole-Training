# Annotate Fineweb dataset

## Installation

Create a new env:

```
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda create -n distilabel-env python=3.12
conda activate distilabel-env
pip install -r requirements.txt
```

If you do not want to use vLLM you can install: `pip install -U distilabel[hf-transformers]` for example (see [distilabel doc](https://distilabel.argilla.io/latest/sections/getting_started/installation/#llms))

## Generate annotations

```
python generate.py --help
usage: generate.py [-h] [--model_name MODEL_NAME] [--data_language {en,fra_Latn,esp_Latn,ita_Latn}] [--nsamples NSAMPLES] [--gpus GPUS] [--prompt_name PROMPT_NAME] [--output_dir OUTPUT_DIR]
                   [--use_cache] [--disable_thinking] [--vllm]

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model you want to use. It can be on HF or local.
  --data_language {en,fra_Latn,esp_Latn,ita_Latn}
                        Dataset language. "en" corresponds to fineweb-edu. Otherwise, from fineweb-2.
  --nsamples NSAMPLES   Number of samples you want to annotate. Usually, we need 400k samples to train a fasttext classfier.
  --gpus GPUS           Number of gpus to use. It will use tensor parallelism, then data parallelism.
  --prompt_name PROMPT_NAME
                        Name of the prompt you want to use. Prompts are defined in "prompt/". You can add new ones. Use the <text> to insert the web page extract (first 2000 characters will be
                        used).
  --output_dir OUTPUT_DIR
                        Output directory. Name of the dataset is generated automatically.
  --use_cache           Activate if you want to use cache. The process may be stuck when activated...
  --disable_thinking    Disable the thinking process for qwen model.
  --vllm                Use vLLM.
```

You can edit `run_generation.slurm` to run it on JZ.

## Plot results

Use `plot_generation_stats.py` and `compare_expe.py` if you want to plot some stats.

## Train a fasttext classifier

Now that you have data you can train your fasttext classifier!

```
python train_fasttext.py --help
usage: train_fasttext.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--label {educational_score,is_toxic,is_ad,topic}] [--epoch EPOCH] [--lr LR] [--ngrams NGRAMS]
                         [--normalize_text] [--from_parquet]

Train a fastText classifier on educational data.

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to the input dataset.
  --output_path OUTPUT_PATH
                        Path to save the output model and data.
  --label {educational_score,is_toxic,is_ad,topic}
                        Label to use for classification.
  --epoch EPOCH
  --lr LR
  --ngrams NGRAMS
  --normalize_text      Whether to normalize the text.
  --from_parquet        read from parquet files.
  ```

  You can edit `run_fasttext.slurm` to run it on JZ.
