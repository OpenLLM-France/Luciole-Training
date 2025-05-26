# Tokenizer Training Chronicles

Several tokenizers were trained, with different sizes (65k and 128k), and including or not Arabic data in the training set. 

There were three sources of training data:
1. **FineWeb-Edu**: A large English dataset, mainly focused on educational content.
2. **FineWeb-2**: A multilingual dataset, including French, Arabic, German, Dutch, Italian, Spanish, and Portuguese.
3. **StarCoder**: A large code dataset.

Each source was randomly subsampled to create a balanced training set across languages, with a focus on French, English and Arabic.
The following table summarizes all the training data used:
| language   | source      | # docs   | # words   | # chars   |
|:-----------|:------------|:---------|:----------|:----------|
| English    | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.11 M   | 860.09 M  | 5.30 B    |
| French     | [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)   | 1.61 M   | 874.40 M  | 5.42 B    |
| Arabic     | FineWeb-2   | 1.86 M   | 909.34 M  | 5.36 B    |
| German     | FineWeb-2   | 364.14 K | 173.39 M  | 1.23 B    |
| Dutch      | FineWeb-2   | 386.89 K | 179.77 M  | 1.13 B    |
| Italian    | FineWeb-2   | 339.42 K | 174.08 M  | 1.12 B    |
| Spanish    | FineWeb-2   | 331.88 K | 178.50 M  | 1.08 B    |
| Portuguese | FineWeb-2   | 360.12 K | 174.91 M  | 1.08 B    |
| Code       | [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata)   | 1.53 M   | 572.41 M  | 5.82 B    |

The dataset distribution, when including Arabic data, is as follows:
![Training Data Distribution](training_data.png)

Compression performances are the following:
![Tokenizer Training Chronicles](compression_performances.png)

Ablation studies were performed to assess the impact of different training data sources and sizes on the learning curves, when training on up to 20B tokens of French data (from FineWeb-2 dataset).
The following results show that the size of the tokenizer and the inclusion of Arabic data have little impact on performance, at least at the beginning of the training.
![Ablation results](ablation_results_fr.png)