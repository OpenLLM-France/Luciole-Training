# Data card for The Luciole Training Dataset 

* [Dataset Description](#dataset-description)
  * [Curation Rationale](#curation-rationale)
    * [Web Data Opt-Outs](#web-data-opt-outs)
    * [Personal and Sensitive Information (PII)](#personal-and-sensitive-information-pii)
  * [Bias, Risks, and Limitations](#bias-risks-and-limitations)
    * [Recommendations](#recommendations)
  * [Sample Metadata](#sample-metadata)
  * [Dataset Composition](#dataset-composition)
* [Downloading the Data](#downloading-the-data)
  * [Sample Use in Python](#sample-use-in-python)
  * [Accessing the English Web Data](#accessing-the-english-web-data)
* [Details on Data Sources](#details-on-data-sources)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)



## Dataset Description

The Luciole Training Dataset is a curated collection of multilingual text data designed for language model pretraining. The data are culled from a variety of sources including: web data, video subtitles, academic papers,
digital books, newspapers, and magazines, some of which were processed by Optical Character Recognition (OCR). The dataset also contains samples of diverse programming languages and some instruction-style and reasoning data.

The Luciole Training Dataset was created by the consortium of the [OpenLLM France](https://openllm-france.fr/) project funded by [BPI France](https://www.bpifrance.fr/) as a part of the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

It was used to pretrain the Luciole family of models, including [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1.1-1B-Base), [Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base) and [Luciole-23B-Base](https://huggingface.co/OpenLLM-France/Luciole-23B-Base), foundation LLMs with strong capabilities in French and English. 

Due to storage constraints, the English web data from the Luciole Training Dataset is published elsewhere (see [Accessing the English Web Data](#accessing-the-english-web-data) below for instructions on how to access this data).

The full dataset contains around 4.65 trillion tokens of multilingual data, including English (53.4%), French (16.3%), German (5.6%), Spanish (4.9%), Italian (2.8%), Portuguese (1.9%), Dutch (1.4%), Arabic (0.7%), and a small subset of regional languages including regional languages of the French metropolitan area, French variants, and French creoles from around the world (0.4%). The latter were selected from the [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset and include Basque, Breton, Catalan, Corsican, Franco-Provençal, Guadeloupean Creole French, Guianese Creole French, Occitan, Picard, Réunion Creole French, Saint Lucian Creole French, Seselwa Creole French, Tahitian, and Walloon.

The dataset also contains parallel data for a selection of languages (0.7%), as well as several programming languages (11.3%) and mathematical data (4.7%; included in the total English data above).

Final language proportions used to train the Luciole models, after up and down-sampling, can be found on the respective model cards.

* License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
* Code repository: [Luciole-Training](https://github.com/OpenLLM-France/Luciole-Training/tree/main/data)
* Paper: coming soon

### Curation Rationale

The Luciole Training Dataset contains only corpora under open licenses. It was created in part to facilitate the training of large language models in strict conformance to open-source requirements and European laws on AI development and intellectual property. 

An equally important motivation was to offer a resource designed to train models that cater to use cases in France. For this reason, the dataset contains a high quantity of French data but also significant amounts of data for languages commonly spoken in Metropolitan France or around its border. In particular, more than one-third of the corpus contains multilingual data with around half of the multilingual data coming from French. 

By sharing our resources openly, we aim to further research on, and development of, multilingual language models.  

#### Web Data Opt-Outs

Robots.txt rules were applied retrospectively to all web-crawled datasets (FineWeb, DCLM, FineMath, InfiWebMath, and MegaMath).
To do this, we processed robots.txt files from the [CommonCrawl dump CC-MAIN-2025-26](https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-26/index.html) and retained only the most recent robots.txt file for each website.

A URL was considered valid if it either explicitly allowed crawling by CCBot or contained a malformed robots.txt file (e.g., HTML content). Websites that did not appear in the CC-MAIN-2025-26 Common Crawl dump were excluded.

#### Personal and Sensitive Information (PII)

We follow the same approach as FineWeb-Edu to remove emails and IP addresses from the dataset.
Email addresses are detected and replaced with placeholders such as "email@example.com" or "firstname.lastname@example.com". Similarly, IP addresses are automatically identified and replaced with the tag "<IP_ADDRESS>".

In addition, phone numbers are detected and anonymized using the Google phonenumbers library, which provides robust parsing and validation for international phone formats. All detected phone numbers are replaced with the token "<PHONE_NUMBER>". This detection covers both international numbers and several country-specific formats, including French, Canadian, Belgian, German, Spanish, Italian, Portuguese, and Dutch phone numbers.

### Bias, Risks, and Limitations
While we have made strong efforts to only include only open corpora, it is possible that individual documents in those corpora are copyrighted. Similarly, it is possible that some personal information in those corpora has slipped through PII filters. If you find your copyrighted work in the Luciole Training Dataset or mention of your personal details therein, we invite you to contact us at contact@openllm-france.fr.

Despite efforts to filter toxicity in web data, improving filtering methods is an ongoing project, and it is extremely likely that toxic and offensive documents remain in web data. Another likely source of biases comes from older data in the public domain. Historical documents can carry biases related to, for example, gender, skin color, ethnicity, and religion that are not socially acceptable. 

A further limitation of this dataset is that it does not distinguish between variants of different languages. Quebequois French and Metropolitan French, to give just one example, are both classified as "French". In future work, we hope to focus more on regional linguistic diversity.

#### Recommendations
Due to harmful biases potentially conveyed by some documents in the Luciole Training Dataset, models pretrained on this data should undergo careful fine-tuning and alignment before being used for non-research purposes. 

### Sample Metadata

In addition to the `text` field, which provides the content of the sample, each training sample in the corpus contains the following metadata when available:

* [`source`]: an identifier for the source of the text sample (e.g., Wikipedia, FineWeb2, Gutenberg, …).
* [`id`]: an identifier that is unique among documents from the same source.
* [`language`]: the language of the text sample:
  - the ISO 639-1 or ISO ...-3 code for a given natural language ("en", "fr", "de", "es", "it", …),
  - the name of a programming language ("python", …),
  - a list of ISO 639-1 codes separated by "-" for data containing parallel translations ("fr-en", "de-fr", "es-en", "it-en", …).
* [`messages`] (optional): if applicable, the text formatted as a conversation following the Hugging Face chat format.
* [`metadata`] (optional): additional metadata about the text sample, in JSON format. This may include information such as the source subset, rights, URL, date, etc.

<!-- ### Dataset Composition -->
<!-- Olivier -->

## Downloading the Data

### Sample Use in Python

### Load the dataset

Load and iterate over the full dataset using the `datasets` library:
```python
from datasets import load_dataset

dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", split="train", streaming=True)

for sample in dataset:
   
   text = sample["text"]

   # … do something with the text
```

### Iterate over a subset

Several configurations are available to select a subset of the dataset.

The list of possible configurations can be obtained programmatically:
```python
from datasets import load_dataset_builder

config_names = list(load_dataset_builder("OpenLLM-BPI/Luciole-Training-Dataset").builder_configs)

print(config_names)
```
```plaintext
['default', '_robots_txt', 'Aya', 'Claire', 'CommonCorpus', 'CommonPile', 'CroissantAligned', 'Culturax-fr', 'Dolma3Longmino', 'Europarl', 'Eurovoc', 'Finemath-3plus', 'Finemath-4plus', 'Fineweb2-acf', 'Fineweb2-ar', 'Fineweb2-br', 'Fineweb2-ca', 'Fineweb2-co', 'Fineweb2-crs', 'Fineweb2-de', 'Fineweb2-es', 'Fineweb2-eu', 'Fineweb2-fr', 'Fineweb2-fr-3plus', 'Fineweb2-frp', 'Fineweb2-gcf', 'Fineweb2-gcr', 'Fineweb2-it', 'Fineweb2-nl', 'Fineweb2-oc', 'Fineweb2-pcd', 'Fineweb2-pt', 'Fineweb2-rcf', 'Fineweb2-ty', 'Fineweb2-wa', 'Fineweb2HQ-ar', 'Fineweb2HQ-de', 'Fineweb2HQ-es', 'Fineweb2HQ-fr', 'Fineweb2HQ-it', 'Fineweb2HQ-nl', 'Fineweb2HQ-pt', 'Gallica', 'Gutenberg', 'HAL', 'HPLT2-fr', 'Infiwebmath-3plus', 'Infiwebmath-4plus', 'INSEE', 'MathPile', 'MegamathWeb', 'NemotronPosttraining', 'OpenCodeReasoning', 'OpenMathInstruct', 'OpenThoughts', 'Opendata', 'Paradocs', 'Parlement', 'PleiasSynth', 'Scholar', 'StackEdu', 'StarcoderData', 'StarcoderOlmomix', 'SynthFineweb2', 'SyntheticWikipediaQA', 'Theses', 'Vikidia', 'Wikimedia', 'Youtube', 'acf', 'ar', 'br', 'ca', 'co', 'crs', 'de', 'en', 'es', 'eu', 'fr', 'frp', 'gcf', 'gcr', 'it', 'nl', 'oc', 'pcd', 'pt', 'rcf', 'ty', 'wa', 'de-fr', 'en-de', 'en-es', 'en-fr', 'en-it', 'en-nl', 'en-pt', 'es-pt']
```

Below are some examples of how to load data from different sources and in different languages.

Load data in French:
```python
from datasets import load_dataset

kwargs = dict(split="train", streaming=True)

dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", "fr", **kwargs)
```
Load data where French and English are aligned:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", "en-fr", **kwargs)
```

Load data from Wikimedia:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", "Wikimedia", **kwargs)
```

Load the Fineweb2-fr dataset:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", "Fineweb2-fr", **kwargs)
```

Load the subset Fineweb2-fr-3plus from the Pile dataset:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", "Fineweb2-fr-3plus", **kwargs)
```

Note that you can also access configurations that are not explicitly specified by exploring the [data hierarchy](data_hierarchy.txt).

For instance, to access the French subset of Wikimedia:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", split="train", streaming=True, data_dir="data/wikimedia/*/fr")
```

Or to load Python data:
```python
dataset = load_dataset("OpenLLM-BPI/Luciole-Training-Dataset", split="train", streaming=True, data_dir="data/**/python")
```

### Accessing the English Web Data

Due to storage limitations on the Hugging Face repository, we could not directly host three subsets of the Luciole-Training-Dataset: FineWeb-edu, DCLM-dolmino, and Fineweb-HQ.

These subsets can instead be downloaded from our external server using the following commands:
```bash
TOKEN=$(curl -s https://dl.labs.linagora.com/api/login -H "Content-Type: application/json" -d '{}')

curl -H "X-Auth: $TOKEN" "https://dl.labs.linagora.com/api/raw/datasets/OpenLLM-France/Luciole-Training-Dataset/?algo=zip" -o Luciole-Training-Dataset.zip
```

## Details on Data Sources

#### Aya Dataset
* <u>Source</u>: [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset). Licence: Apache 2.0.
* <u>Description</u>: "The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere Labs. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators. This dataset can be used to train, finetune, and evaluate multilingual LLMs" (Aya Dataset [data card](https://huggingface.co/datasets/CohereLabs/aya_dataset)).
* <u>Citation</u>: Shivalika Singh, Freddie Vargus, and Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzemiński, Hakimeh Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee and Sara Hooker (2024). Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.   [arXiv:2402.06619](https://arxiv.org/abs/2402.06619)
   


#### Claire (French and English)
* <u>Sources</u>:
  * French dataset: [OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1).
  * English dataset: [OpenLLM-France/Claire-Dialogue-English-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1).
* <u>Extracted from</u>: see the datacards for the [French](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1) and [English](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1) datasets.
* <u>Description</u>: The Claire datasets are composed of transcripts of spoken conversations -- including parliamentary proceedings, interviews, debates, meetings, and free conversations -- as well as some written conversations from theater plays and written chats. The dataset is designed to help downstream performance of models fine-tuned for tasks requiring the comprehension of spontaneous spoken conversation, such as meeting summarization. Each dialogue is split into speech turns, and each speech turn is labeled with the name of the speaker or a unique identifier. See the composition details for the <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_claire-french_pie.png">French dataset</a> and the <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_claire-english_pie.png">English dataset</a> for a high-level view of the distribution of different types of documents in each dataset.
* <u>Citation</u>: Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré (2023). The Claire French Dialogue Dataset. [arXiv:2311.16840](https://arxiv.org/abs/2311.16840).


#### Common Corpus
<!-- Julie -->
* <u>Source</u>: [PleIAs/common_corpus](https://huggingface.co/datasets/PleIAs/common_corpus). License: Public Domain or mixed open licenses.
* <u>Description</u>: "The data assembled in Common Corpus are either uncopyrighted or under permissible licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the high-resource European languages to some low-resource languages rarely represented in pre-training datasets. In addition, it includes a large portion of code data" (Langlais et al, (2026)).
<!-- Subsets -->
<!-- <u>Pre-processing</u>: -->
* <u>Citation</u>: Pierre-Carl Langlais, Pavel Chizhov, Catherine Arnett, Carlos Hinostroza, Mattia Nee, Eliot Jones, Irène Girard, David Mach, Anastasia Stasenko, Ivan Yamshchikov (2026). Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training. ICLR 2026.

#### Common Pile (v0.1)
* <u>Source</u>: [common-pile/common-pile-v01-filtered-data](https://huggingface.co/collections/common-pile/common-pile-v01-filtered-data). License: Mixed open licenses (see document details for each subset).
* <u>Description</u>: The Common Pile v0.1 is a curated "eight terabyte collection of openly licensed text designed for LLM pretraining. The Common Pile comprises content from 30 sources that span diverse domains including research papers, code, books, encyclopedias, educational materials, audio transcripts, and more" (Kandpal et al., 2025).
<!-- Subsets -->
<!-- <u>Pre-processing</u>: -->
* <u>Citation</u>: Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella Biderman, Baber Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper, Aviya Skowron, John Kirchenbauer, Shayne Longpre, Lintang Sutawika, Alon Albalak, Zhenlin Xu, Guilherme Penedo, Loubna Ben Allal, Elie Bakouch, John David Pressman, Honglu Fan, Dashiell Stander, Guangyu Song, Aaron Gokaslan, Tom Goldstein, Brian R. Bartoldson, Bhavya Kailkhura, and Tyler Murray (2025). [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)

#### Croissant Aligned
* <u>Source</u>: [OpenLLM-France/Translation-Instruct](https://huggingface.co/datasets/OpenLLM-France/Translation-Instruct). License: CC-BY-SA 4.0.
* <u>Original source</u>: [croissantllm/croissant_dataset_no_web_data](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data/tree/main/aligned_36b) (subset: `aligned_36b`). License: not specified.
* <u>Extracted from</u>: 
  * Translation pairs: [OPUS](https://opus.nlpl.eu/) (99.6% of the data in CroissantAligned). Pairs extracted from OPUS are labeled as "UnbabelFrEn". 
  * Thesis abstracts: French thesis abstract pairs. License: [ETALAB-Licence-Ouverte-v2.0](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).
  * Song lyrics: [lacoccinelle](https://www.lacoccinelle.net). 
* <u>Description</u>: CroissantAligned contains samples of parallel French/English (or English/French) data. Data extracted from OPUS takes the form of sentences pairs, where one sentence is in French and the other is in English. OPUS pairs were passed through a custom pipeline designed to select the highest quality translation examples. Selected pairs are labeled "UnbabelFrEn" in the CroissantAligned dataset. The thesis abstract subset contains thesis abstracts paired with translations written by the thesis authors. The song lyrics are translated by contributors to www.lacoccinelle.net. Parallel data are used to boost the multilingual capabilities of models trained on them ([Faysse et al.,2024](https://arxiv.org/pdf/2402.00786)).
* <u>Citation</u>: Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F.T. Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo (2024). "CroissantLLM: A Truly Bilingual French-English Language Model," [arXiv:2402.00786](https://arxiv.org/abs/2402.00786).

#### CulturaX
* <u>Source</u>: [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) Licence: [mC4 license](https://huggingface.co/datasets/allenai/c4#license), [OSCAR license](https://huggingface.co/datasets/uonlp/CulturaX).
* <u>Description</u>: A combination of mC4 and OSCAR corpora; a "substantial multilingual dataset with 6.3 trillion tokens in 167 languages, tailored for large language model (LLM) development. Our dataset undergoes meticulous cleaning and deduplication through a rigorous pipeline of multiple stages to accomplish the best quality for model training, including language identification, URL-based filtering, metric-based cleaning, document refinement, and data deduplication" (CulturaX [data card](https://huggingface.co/datasets/uonlp/CulturaX)).
* <u>Citation</u>: Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt,
      Ryan A.Rossi, and Thien Huu Nguyen (2024). CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages. In Calzolari et al, (eds.), Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). pp. 4226-4237. [paper](https://aclanthology.org/2024.lrec-main.377)

#### DCLM Dolmino (via external server)
* <u>Source</u>: [allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124), DCLM subset. Licence: ODC-BY.
* <u>Description</u>: A subset of data from [DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). "DCLM-Baseline was created by applying a series of cleaning, filtering, and deduplication steps to the raw Common Crawl data (DCLM-Pool). The key steps include: Heuristic cleaning and filtering (reproduction of RefinedWeb), Deduplication using a Bloom filter, Model-based filtering using a fastText classifier trained on instruction-formatted data" (DCLM Baseline 1.0 [data card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0))
* <u>Citation</u>: 
  * Dolmino Mix: Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, and Matt Jordan, et al. (2024). 2 OLMo 2 Furious. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656).
  * DCLM: Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar (2024). DataComp-LM: In search of the next generation of training sets for language models. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)

#### Dolma3 Longmino
* <u>Source</u>: [allenai/dolma3_longmino_mix-100B-1125](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125). License: ODC-By.
* <u>Description</u>: The full Longmino dataset contains over 22 million long documents from the olmOCR pool of science PDFs. These documents are filtered and synthetically augmented by injecting certain aggregation tasks at regular intervals. (For more details, see the [Olmo3 paper](https://arxiv.org/pdf/2512.13961).) The Dolma 3 Longmino Mix (100B) is a selection of documents from the larger Dolmino pool that was used during the third stage of training for the Olmo 3 32B model.
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld,  Faeze Brahman,  Finbarr Timbers,  Hamish Ivison,  et al. (2025). Olmo 3. [arXiv:2512.13961](https://arxiv.org/pdf/2512.13961).

#### Europarl and EuroparlAligned. 
* <u>Sources</u>: 
  * Monolingual data: [Europarl v10](https://www.statmt.org/europarl/v10/training-monolingual/). License: [Open](https://www.statmt.org/europarl/).
  * Parallel data: [OpenLLM-France/Translation-Instruct](https://huggingface.co/datasets/OpenLLM-France/Translation-Instruct)
* <u>Original sources</u>:
  * `fr-en`, `es-en`, `it-en` parallel data: [Europarl v7](https://www.statmt.org/europarl/v7/). License: [Open](https://www.statmt.org/europarl/).
  * `de-fr` parallel data: [Europarl v10](https://www.statmt.org/europarl/v10/training-monolingual/). License: [Open](https://www.statmt.org/europarl/).
* <u>Description</u>: "The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek. The goal of the extraction and processing was to generate sentence aligned text for statistical machine translation systems" ([www.statmt.org](https://www.statmt.org/europarl/)).
* <u>Citation</u>: Philipp Koehn (2005). "Europarl: A Parallel Corpus for Statistical Machine Translation," MT Summit. 

#### Eurovoc
* <u>Source</u>:   [EuropeanParliament/Eurovoc](https://huggingface.co/datasets/EuropeanParliament/Eurovoc). License: [EUPL 1.1](https://huggingface.co/datasets/EuropeanParliament/Eurovoc).
* <u>Extracted from</u>: [Cellar](https://op.europa.eu/en/web/cellar). License: [CC BY-4.0](https://op.europa.eu/en/web/about-us/legal-notices/publications-office-of-the-european-union-copyright).
* <u>Description</u>: A collection of mutlilingual documents from the data repository of the Publications Office of the European Union annotated with Eurovoc labels. The corpus contains legal, policy-related, historical and organizational information about the EU. Dataset containing text retrieved through OCR.
<!-- * <u>Pre-processing</u>: -->
* <u>Citations</u>:
  * Ilias Chalkidis, Emmanouil Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos (2019). "[Extreme Multi-Label Legal Text Classification: A Case Study in EU Legislation](https://arxiv.org/pdf/1905.10892)," Proceedings of the Natural Legal Language Processing Workshop 2019, pages 78–87, Minneapolis, Minnesota. Association for Computational Linguistics.
  * Ilias Chalkidis,  Manos Fergadiotis, Prodromos Malakasiotis and Ion Androutsopoulos (2019). "[Large-Scale Multi-Label Text Classification on EU Legislation](https://arxiv.org/pdf/1906.02192)," Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers).
  * Andrei-Marius Avram, Vasile Pais, and Dan Ioan Tufis (2021). "[PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors](https://arxiv.org/pdf/2108.01139)," Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 92–101, Held Online. INCOMA Ltd.
  * Zein Shaheen, Gerhard Wohlgenannt and Erwin Filtz (2020). "Large scale legal text classification using transformer models," [arXiv:2010.12871](https://arxiv.org/abs/2010.12871v1).

#### FineMath and InfiMM-WebMath
* <u>Source</u>: [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath). License: ODC-BY.
* <u>Description</u>: "FineMath consists of 34B tokens (FineMath-3+) and 54B tokens (FineMath-3+ with InfiMM-WebMath-3+) of mathematical educational content filtered from CommonCrawl. To curate this dataset, we trained a mathematical content classifier using annotations generated by LLama-3.1-70B-Instruct. We used the classifier to retain only the most educational mathematics content, focusing on clear explanations and step-by-step problem solving rather than advanced academic papers" (FineMath [data card](https://huggingface.co/datasets/HuggingFaceTB/finemath)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín,  Vaibhav Srivastav,  Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra and Thomas Wolf (2025). SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. [arXiv:2502.02737](https://arxiv.org/abs/2502.02737). 

#### FineWeb2
* <u>Source</u>: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). License: ODC-BY.
* <u>Description</u>: FineWeb2 extends the original [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset by adding pretraining data for over 1000 languages. "The data was sourced from 96 CommonCrawl snapshots, spanning the summer of 2013 to April 2024, and processed using datatrove, our large scale data processing library. This carefully deduplicated and filtered dataset comprises roughly 20 terabytes, across 5 billion documents, with over 3 trillion words" (FineWeb2 [data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)). 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Guilherme Penedo,  Hynek Kydlíček,  Vinko Sabolčec,  Bettina Messmer,  Negar Foroutan,  Amir Hossein Kargaran,  Colin Raffel,  Martin Jaggi, Leandro Von Werra and Thomas Wolf (2025). FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language. [arXiv:2506.20920](https://arxiv.org/abs/2506.20920).


#### FineWeb HQ (via external server)
* <u>Source</u>: [epfml/FineWeb-HQ](https://huggingface.co/datasets/epfml/FineWeb-HQ). License: ODC-BY.
* <u>Description</u>: "FineWeb-HQ is a high-quality, model-filtered pretraining dataset derived as a subset of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). FineWeb-HQ was created by selecting the top 10% of FineWeb documents based on a deep learning classifier trained to identify structured and knowledge-rich samples. This classifier uses XLM-RoBERTa embeddings to score documents."
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Bettina Messmer, Vinko Sabolčec and Martin Jaggi (2025). Enhancing Multilingual LLM Pretraining with Model-Based Data Selection. [arXiv:2502.10361](https://arxiv.org/abs/2502.10361).

#### FineWeb 2 HQ
* <u>Source</u>: [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ). License: ODC-BY.
* <u>Description</u>: "FineWeb2-HQ is a high-quality, model-filtered pretraining dataset derived as a subset of [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), spanning 20 languages. It enables around 6x faster pretraining compared to the base dataset. FineWeb2-HQ was created by selecting the top 10% quality documents of FineWeb2 in each language, based on scores assigned by a deep learning classifier trained to identify structured and knowledge-rich samples using XLM-RoBERTa embeddings."
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Bettina Messmer, Vinko Sabolčec and Martin Jaggi (2025). Enhancing Multilingual LLM Pretraining with Model-Based Data Selection. [arXiv:2502.10361](https://arxiv.org/abs/2502.10361).

#### FineWebEdu (via external server)
* <u>Source</u>: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
* <u>Extracted from</u>: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
* <u>Description</u>: A 1.3 trillion token selection from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), which contains 15 trillion tokens of curated data from 96 Common Crawl dumps. Content in FineWebEdu has been selected by a custom designed classifier for its high-quality, educational content. Most recent crawl: 2024-10 (see <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_finewebedu-english_histogram.png">composition details</a> for information about the crawls included in this dataset.)
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," [	arXiv:2406.17557](https://arxiv.org/abs/2406.17557).

#### Gallica
* <u>Sources</u>:
  * Monographies: [PleIAs/French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books). License: Public domain.
  * Press: [PleIAs/French-PD-Newspapers](https://huggingface.co/datasets/PleIAs/French-PD-Newspapers). License: Public domain.
* <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
* <u>Description</u>: A large collection of French monographies, newspapers and periodicals in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)). Dataset containing text retrieved through OCR.
<!-- * <u>Pre-processing</u>: -->


#### Gutenberg
* <u>Source</u>: Corpus compiled by OpenLLM partners.
* <u>Extracted from</u>: 
  * [aleph.gutenberg.org](http://aleph.gutenberg.org/) via [Project Gutenberg](https://www.gutenberg.org/). License: [Open](https://www.gutenberg.org/policy/terms_of_use.html).
  * [pgcorpus](https://github.com/pgcorpus/gutenberg). License: [CC BY-4.0](https://zenodo.org/records/2422561).
* <u>Description</u>: A collection of free eBooks, manually prepared by human annotators. 
* <u>Pre-processing</u>:
  * <u>Filtering</u>: The dataset was filtered based on the author date of death, so that only texts from authors who died more than 70 years ago are included (80 years for French authors). See [code details here](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1136). This filtering was done to ensure that the texts are in the public domain.
  * <u>Text cleaning</u>: Headers and footers containing information about Project Gutenberg were removed (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/text.py#L93)).

#### HAL
<!-- Julie -->
* <u>Source</u>: Corpus processed by OpenLLM partners. <!--  and published separately as [](). License: -->
* <u>Original source</u>:  based on [almanach/halvest](https://huggingface.co/datasets/almanach/halvest). License: [HAL license](https://doc.hal.science/en/legal-aspects/).
* <u>Extracted from</u>: [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of scientific papers and manuscripts distributed through the open science platform HAL. Dataset containing text retrieved through OCR.
<!-- * <u>Pre-processing</u>: -->
  
<!-- * <u>Citation</u>: -->


#### HPLT 2
* <u>Source</u>: [HPLT/HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned). Licence: CC-0 1.0.
* <u>Description</u>: A cleaned, "large-scale collection of web-crawled documents in 191 world languages, produced by the HPLT project. The source of the data is mostly Internet Archive with some additions from Common Crawl" (HPLT 2 [data card](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Laurie Burchell, Ona de Gibert, Nikolay Arefyev, Mikko Aulamo, Marta Bañón, Pinzhen Chen, Mariia Fedorova, Liane Guillou, Barry Haddow, Jan Hajic, Jindrich Helcl, Erik Henriksson, Mateusz Klimaszewski, Ville Komulainen, Andrey Kutuzov, Joona Kytöniemi, Veronika Laippala, Petter Mæhlum, Bhavitvya Malik, Farrokh Mehryary, Vladislav Mikhailov, Nikita Moghe, Amanda Myntti, Dayyán O’Brien, Stephan Oepen, Proyag Pal, Jousia Piha, Sampo Pyysalo, Gema Ramírez-Sánchez, David Samuel, Pavel Stepachev, Jörg Tiedemann, Duan Variš, Tereza Vojtechová and Jaume Zaragoza-Bernabeu (2025). An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT). Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 17452-17485. [paper](https://aclanthology.org/2025.acl-long.854/)


#### INSEE
<!-- Julie -->
* <u>Source</u>: Corpus processed by OpenLLM partners. <!--  and published separately as [](). License: -->
* <u>Extracted from</u>: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
<!-- * <u>Citation</u>: -->

#### MathPile (Commercial)
* <u>Source</u>: [OpenLLM-France/Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset).
* <u>Original source</u>: [GAIR/MathPile_Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial). License: [CC BY-SA 4.0](https://huggingface.co/datasets/GAIR/MathPile_Commercial).
* <u>Description</u>: A preprocessed collection of documents focused on math, including Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and web pages from Common Crawl. The content targets a range of levels, from kindergarten through postgraduate level. MathPile_Commercial was obtained by removing documents from MathPile that do not allow commercial use.
* <u>Citation</u>: Zengzhi Wang, Rui Xia and Pengfei Liu (2023). "Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math," [	arXiv:2312.17120](https://export.arxiv.org/abs/2312.17120).


#### MegaMath Web
* <u>Source</u>: [LLM360/MegaMath](https://huggingface.co/datasets/LLM360/MegaMath). Licence: ODC-BY.
* <u>Description</u>: MegaMath is "an open math pretraining dataset curated from diverse, math-focused sources, with over 300B tokens". MegaMath Web includes "re-extracted mathematical documents from Common Crawl with math-oriented HTML optimizations, fasttext-based filtering and deduplication, all for acquiring higher-quality data on the Internet" (MegaMath [data card](https://huggingface.co/datasets/LLM360/MegaMath)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Fan Zhou, Zengzhi Wang,  Nikhil Ranjan,  Zhoujun  Cheng, Liping Tang,  Guowei He,  Zhengzhong Liu, and Eric P. Xing (2025). MegaMath: Pushing the Limits of Open Math Corpora. [arXiv:2504.02807](https://arxiv.org/abs/2504.02807).

#### Nemotron Post-Training v2
* <u>Source</u>: [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2). Licence: CC-BY 4.0.
* <u>Description</u>: A collection of instruction-style, supervised-fine tuning data in math, code, STEM (science-technology-engineering-math), and general chat. This version contains instructions in French, Spanish, Italian, German, and Japanese.
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 
  * Dhruv Nathawani, Shuoyang Ding,  Vitaly Lavrukhin,  Igor Gitman,  Somshubra Majumdar,  Evelina Bakhturina,  Boris Ginsburg,  and Jane Polak Scowcroft (2025). Nemotron-Post-Training-Dataset-v2. [Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).
  * NVIDIA (2025). NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model. [arXiv:2508.14444](https://arxiv.org/abs/2508.14444).

#### Open Code Reasoning
* <u>Source</u>: [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning). Licence: CC-BY 4.0.
* <u>Original sources:</u> 
  * CodeForces problems:  [CodeForces](http://codeforces.com).
  * Question collections: [TACO](https://huggingface.co/datasets/BAAI/TACO), [APPS](https://huggingface.co/datasets/codeparrot/apps), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), and [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces).
* <u>Description</u>: OpenCodeReasoning "comprises 735,255 samples in Python across 28,319 unique competitive programming questions. OpenCodeReasoning is designed for supervised fine-tuning (SFT)" (OpenCodeReasoning [data card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg (2025). OpenCodeReasoning: Advancing Data Distillation for Competitive Coding. [arXiv:2504.01943](https://arxiv.org/abs/2504.01943).

#### OpenData
* <u>Source</u>: [Nicolas-BZRD/DILA_OPENDATA_FR_2023](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main) (balo, dole, inca, kali, legi and sarde subsets). License: [ODC-BY](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main).
* <u>Extracted from</u>: [OpenData](https://echanges.dila.gouv.fr/OPENDATA/) (Data collection date: October, 2023).
* <u>Description</u>: "The French Government Open Data (DILA) Dataset is a collection of text data extracted from various sources provided by the French government, specifically the Direction de l'information légale et administrative (DILA). This dataset contains a wide range of legal, administrative, and legislative documents. The data has been organized into several categories for easy access and analysis" (from the [dataset card](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main)).
<!-- * <u>Citation</u>: No paper found. -->


#### Open Math Instruct (v1)
* <u>Source</u>: [nvidia/OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1). Licence: NVIDIA.
* <u>Description</u>: "OpenMathInstruct-1 is a math instruction tuning dataset with 1.8M problem-solution pairs generated using permissively licensed Mixtral-8x7B model. The problems are from GSM8K and MATH training subsets and the solutions are synthetically generated by allowing Mixtral model to use a mix of text reasoning and code blocks executed by Python interpreter" (OpenMathInstruct [data card](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia and Igor Gitman (2024). OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset. [arXiv:2402.10176](https://arxiv.org/abs/2402.10176).

#### Open Thoughts
* <u>Source</u>: [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M). Licence: Apache 2.0.
* <u>Description</u>: "This dataset comprises 1.2 million questions across math, code, and science domains, with reasoning traces annotated from QwQ-32B. OpenThoughts3-1.2M is the result of over 1,000+ rigorous experiments on each stage in the reasoning dataset construction pipeline" (OpenThoughts3 [blog](https://www.openthoughts.ai/blog/ot3)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt (2025). OpenThoughts: Data Recipes for Reasoning Models. [arXiv:2506.04178](https://arxiv.org/abs/2506.04178).


#### Paradocs
* <u>Source</u>: [jhu-clsp/paradocs](https://huggingface.co/datasets/jhu-clsp/paradocs). Licence: Apache 2.0.
* <u>Description</u>: "ParaDocs is a publicly available dataset that produces parallel annotations for the document-level metadata of three large publicly available corpora (ParaCrawl, Europal, and News Commentary) in many languages" (ParaDocs [data card](https://huggingface.co/datasets/jhu-clsp/paradocs)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Rachel Wicks, Matt Post, and Philipp Koehn (2024). Recovering document annotations for sentence-level bitext. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9876–9890. Association for Computational Linguistics. [paper](https://aclanthology.org/2024.findings-acl.589/).

#### Parlement
* <u>Source</u>: [OpenLLM-France/Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset). Subsets: AmendementsParlement, DiscoursPublics, InterventionsParlement, QuestionsEcritesParlement.
* <u>Extracted from</u>:  
  * DiscoursPublics: [Vie Publique](https://www.vie-publique.fr/collection-discours-publics). License: [ETALAB-Licence-Ouverte-v2.0](https://www.vie-publique.fr/mentions-legales).
  * Other subsets: [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4). License: [CC BY-SA](https://www.regardscitoyens.org/mentions-legales/).
* <u>Description</u>: AmendementsParlement is a collection of proposed amendments by the French parliament. DiscoursPublics is a collection of public speeches from the principal public actors in France including speeches from the French President starting from 1974 and from the Prime Minister and members of the government starting from 1980. InterventionsParlement is transcripts of remarks made during French parlementary debates.  QuestionsEcritesParlement is a collection of long written questions, read during a session at the French National Assembly. 

#### Pleias SYNTH
* <u>Source</u>: [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH). Licence: CDLA-permissive 2.0.
* <u>Description</u>: SYNTH is a synthetic dataset created on the basis of seed data from Wikipedia, Wikipedia:Vital, Wikibooks and hand-crafted data. These seed data are used to generate a variety of queries and responses, including negative queries, which make up the resulting SYNTH data. 
<!-- * <u>Pre-processing</u>: -->
<!--* <u>Citation</u>: -->

#### Scholar 
* <u>Source</u>: [kurakurai/scholar](https://huggingface.co/datasets/kurakurai/scholar) Licence: ODC-BY.
* <u>Description</u>: "This dataset was created to address the lack of high-quality scientific datasets in French. It is based on Baccalauréat and Classes Préparatoires (CPGE) exam questions and their detailed solutions, covering a wide range of subjects, primarily mathematics, physics and chemistry and computer science. The dataset includes 30.3K annotated samples designed to support both educational and research applications in French-language NLP" (Scholar [data card](https://huggingface.co/datasets/kurakurai/scholar)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Maxence Lasbordes and Sinoué Gad (2025). Luth: Efficient French Specialization for Small Language Models and Cross-Lingual Transfer. [arXiv:2510.05846](https://arxiv.org/abs/2510.05846).

#### StarCoder Data
* <u>Source</u>: [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata). Licence: Mixed Open Licenses.
* <u>Description</u>: StarCoder "contains 783GB of code in 86 programming languages, and includes 54GB GitHub Issues + 13GB Jupyter notebooks in scripts and text-code pairs, and 32GB of GitHub commits, which is approximately 250 Billion tokens" (StarCoder [data card](https://huggingface.co/datasets/bigcode/starcoderdata)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Raymond Li, Loubna Ben allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia LI, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Joel Lamy-Poirier, Joao Monteiro, Nicolas Gontier, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Ben Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason T Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Urvashi Bhattacharyya, Wenhao Yu, Sasha Luccioni, Paulo Villegas, Fedor Zhdanov, Tony Lee, Nadav Timor, Jennifer Ding, Claire S Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Carolyn Jane,erson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro Von Werra, Harm de Vries (2023). StarCoder: May the Source Be With You!. In Transactions on Machine Learning Research, pages 2835-8856.  [paper](https://openreview.net/forum?id=KoFOg41haE)

#### Starcoder Olmomix
* <u>Source</u>: [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124). Licence: ODC-BY.
* <u>Description</u>: A filtered subset of [StarCoder Data](#starcoder-data). Documents are filtered to remove documents with fewer than 2 stars on GitHub, with only binary format or numerical content, or repeated sequences of 32 or more n-grams. 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Dolmino Mix: Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, and Matt Jordan, et al. (2024). 2 OLMo 2 Furious. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656).

#### StackEdu 
* <u>Source</u>: [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu). License: Mixed open licenses.
* <u>Extracted from</u>: [GitHub](https://github.com/) via [GHarchive](https://www.gharchive.org/) via the [Software Heritage](https://www.softwareheritage.org/) archive. Mixed licenses for source.
* <u>Description</u>: Stack-Edu is a 125B token dataset of educational code filtered from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2). "The Stack v2 contains over 3B files in 600+ programming and markup languages. The dataset was created as part of the [BigCode Project](https://www.bigcode-project.org/), an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets" (Stack [data card](https://huggingface.co/datasets/bigcode/the-stack-v2)).
* <u>Citation</u>: 
  * StackEdu: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall,,rés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf (2025). SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model [arXiv:2502.02737](https://arxiv.org/abs/2502.02737).
  * Stack v2: Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane,erson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, Harm de Vries (2024). StarCoder 2 and The Stack v2: The Next Generation. [arXiv:2402.19173](https://arxiv.org/abs/2402.19173).



#### Synth FineWeb 2
* <u>Source</u>: Original subset of the Luciole Training Corpus.
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->


#### Synth Wikipedia
* <u>Source</u>: Original subset of the Luciole Training Corpus.
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->


#### Theses
* <u>Source</u>: [OpenLLM-France/Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset).
* <u>Extracted from</u>: [theses.fr](https://theses.fr/?domaine=theses) (License: [Licence Ouverte / Open Licence version 2.0](https://www.data.gouv.fr/fr/datasets/theses-soutenues-en-france-depuis-1985/)) and  [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of doctoral theses published in France. Dataset containing text retrieved through OCR.

<!-- * <u>Citation</u>: No paper found. -->


#### Vikidia
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Wikimedia
* <u>Source</u>: [OpenLLM-France/wikimedia](https://huggingface.co/datasets/OpenLLM-France/wikimedia)
* <u>Extracted from</u>: [Wikimedia dumps](https://dumps.wikimedia.org/other/enterprise_html/runs/). License: [GFDL/CC BY-SA](https://dumps.wikimedia.org/legal.html).
* <u>Description</u>: A curated collection of Wikimedia pages in markdown format, compiled from various Wikimedia projects across multiple languages, including: Wikipedia, Wikibooks, Wikinews, Wikiquote, Wikisource, Wikiversity, Wikivoyage, Wiktionary.
<!-- * <u>Pre-processing</u>: TODO -->
<!-- * <u>Citation</u>: No paper found. -->

#### YouTube
<!-- Julie -->
* <u>Source</u>: [OpenLLM-France/Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset).
* <u>Extracted from</u>: [YouTube](https://www.youtube.com/). <!-- License: TODO? -->
* <u>Description</u>: French subtitles from videos published with permissive licenses on YouTube. <!-- TODO -->



## Citation

✍ Paper coming soon!


## Acknowledgements

We gratefully acknowledge BPI France for funding the OpenLLM France project under the call "Communs numériques pour l’intelligence artificielle générative" ("Digital commons for generative artificial intelligence") and the project numbers DOS0250771 and DOS0250773.

Processing and storage of the Luciole Training Dataset was made possible by computing AI and storage resources by GENCI at IDRIS thanks to the grant 2024-GC011015444 on the supercomputer Jean Zay. We gratefully acknowledge support from GENCI and IDRIS and from Stephane Requena (GENCI) and Pierre-François Lavallée (IDRIS) in particular. 

The Luciole Training Dataset was created by members of [LINAGORA](https://labs.linagora.com/) and [OpenLLM-France](https://openllm-france.fr/), including, in alphabetical order:  
 
Akshay Chaturvedi (LINAGORA)   
Jérôme Deshayes (CEA)  
Liam Duignan (CEA)    
Olivier Ferret (CEA)  
Olivier Gouvert (LINAGORA)  
Julie Hunter (LINAGORA)  
Jean-Pierre Lorré (LINAGORA)  
Jérôme Louradour (LINAGORA)     
Kate Thompson (LINAGORA)   

We thank the support team from IDRIS for their technical guidance throughout the project, especially:  Martin Comminges, Rémi Lacroix, and Myriam Peyrounette.    

We are also greatful to the partners of the [OpenLLM-France](https://www.openllm-france.fr/) consortium for their valuable input, with particular thanks to (in alphabetical order):  

Pascal Alix (Sorbonne),   
Gabriel Lauzzana (LORIA),      
Michel-Marie Maudet (LINAGORA),  
Celia Zolynski (Sorbonne)

We would also like to thank the numerous open data projects that have guided us in, or directly contributed to, the creation of this dataset. This includes in particular: the [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) from [Pleias](https://pleias.fr/), the [Common Pile](https://huggingface.co/common-pile), the [Nemotron post-training datasets](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) from [Nvidia](https://www.nvidia.com/en-eu/) and numerous projects from [Hugging Face](https://huggingface.co/) and [Allen AI](https://allenai.org/).

Finally, we thank the entire OpenLLM-France community, whose members have helped in diverse ways. 

## Contact

contact@openllm-france.fr


