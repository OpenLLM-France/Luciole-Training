# Tokenizer Training

Train, evaluate, and publish custom BPE tokenizers for multilingual LLMs.

## Scripts

| Script | Description |
|--------|-------------|
| `tokenizer_train.py` | Train a BPE tokenizer using HuggingFace tokenizers library |
| `tokenizer_eval.py` | Evaluate tokenizer compression across languages and domains |
| `tokenizer_quicktest.py` | Quick tokenization test on sample text |

## Workflow

### 1. Prepare training data

```bash
cd prepare_data/
python dump_subsampled_parquets.py --help
```

This subsamples datasets proportionally to create a balanced tokenizer training corpus.

### 2. Train a tokenizer

```bash
python tokenizer_train.py \
    --vocab_size 128000 \
    --alphabet alphabet.tsv \
    --output_dir trained/my_tokenizer
```

Key parameters:
- `--vocab_size`: Target vocabulary size (e.g. 65000, 128000)
- `--alphabet`: TSV file defining the base character set
- Training data is configured in `data.py`

### 3. Evaluate

```bash
python tokenizer_eval.py --tokenizer_path trained/my_tokenizer
```

Evaluates compression ratio across:
- Multiple languages (English, French, German, Spanish, Italian, Arabic, Basque)
- Code (Python, C++, TeX, JavaScript)
- Parallel corpora (Europarl)

### 4. Create instruction variant

```bash
python tokenizer_make_instruct.py --tokenizer_path trained/my_tokenizer
```

Adds chat/instruction special tokens to the base tokenizer.

## Results

See `chronicles/README.md` for training data distributions and compression performance results.
