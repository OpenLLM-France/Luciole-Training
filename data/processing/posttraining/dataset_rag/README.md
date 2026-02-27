# dataset_rag

Pipeline to generate an SFT-ready RAG dataset from **HotpotQA** and **TAT-QA**.
The goal is to teach the LLM to cite its sources as specified in the system prompt using textual and tabular data, to avoid being influenced by chunk position, and to refuse answering when retrieved chunks are not relevant.

It needs a `.env` file (see `.env.example`) with `LLM_API_KEY=...`.

## Scripts

- `augment_hotpotqa.py`: generates answers with reasoning including accurate citations, and unanswerable examples from HotpotQA.
- `convert_tatqa_to_augment_format.py`: converts TAT-QA to the internal JSONL format.
- `augment_tatqa.py`: generates reasoning traces on converted TAT-QA.
- `evaluate_and_filter.py` / `evaluate_tatqa.py`: evaluation + filtering.
- `formatting_sft.py`: final conversion to SFT format (`prompt_completion` or `chat`).

## Supported languages

HotpotQA augmentation and SFT formatting support `en` and `fr` .
Pass `--language fr`. French is based on a translation of HotpotQA made by DeepSeek-R1-Distill-Qwen-32B. Because of too many translation errors, we chose not to use it yet and will look at MIRACL.

## Quick pipeline

```bash
# 1) HotpotQA
python3 augment_hotpotqa.py --output hotpotqa_augmented --limit 200
python3 evaluate_and_filter.py --input hotpotqa_augmented.jsonl --output hotpotqa_augmented_evaluated.jsonl --filter
python3 formatting_sft.py --input hotpotqa_augmented_evaluated_filtered.jsonl --format chat --language en

# 2) TAT-QA
python3 convert_tatqa_to_augment_format.py --input tatqa.json --output tatqa_converted.jsonl
python3 augment_tatqa.py --dataset tatqa_converted.jsonl --output tatqa_augmented
python3 evaluate_tatqa.py --input tatqa_augmented.jsonl --output tatqa_augmented_evaluated.jsonl --filter
python3 formatting_sft.py --input tatqa_augmented_evaluated_filtered.jsonl --format chat 
```
