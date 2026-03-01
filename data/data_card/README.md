# The Luciole Training Dataset 

The Luciole Training Dataset is a curated collection of multilingual text data culled from a variety of sources including: web data, video subtitles, academic papers,
digital books, newspapers, and magazines, some of which were processed by Optical Character Recognition (OCR). It also contains samples of diverse programming languages and some instruction-style and reasoning data.

The Luciole Training Dataset was created by the consortium of the [OpenLLM France](https://openllm-france.fr/) project funded by [BPI France](https://www.bpifrance.fr/) as a part of the [France 2030](https://www.info.gouv.fr/grand-dossier/france-2030) program.

It was used to pretrain the Luciole family of models, including [Luciole-1B-Base](https://huggingface.co/OpenLLM-France/Luciole-1.1-1B-Base), [Luciole-8B-Base](https://huggingface.co/OpenLLM-France/Luciole-8B-Base) and [Luciole-23B-Base](https://huggingface.co/OpenLLM-France/Luciole-23B-Base), foundation LLMs with strong capabilities in French and English. Code for data preparation can be found in the repository [Luciole-Training](). The corpus is published under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) licence.

Due to storage constraints, the English web data from the Luciole Training Dataset is published elsewhere (see [Accessing the English web data](#accessing-the-english-web-data) below for instructions on how to access this data).

The full dataset contains around 4.65 trillion tokens of multilingual data, including English (53.4%), French (16.3%), German (5.6%), Spanish (4.9%), Italian (2.8%), Portuguese (1.9%), Dutch (1.4%), Arabic (0.7%), and a small subset of regional languages including regional languages of the French metropolitan area, French variants, and French creoles from around the world (0.4%). The latter were selected from the [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) dataset and include Basque, Breton, Catalan, Corsican, Franco-Provençal, Guadeloupean Creole French, Guianese Creole French, Occitan, Picard, Réunion Creole French, Saint Lucian Creole French, Seselwa Creole French, Tahitian, and Walloon.

The dataset also contains parallel data for a selection of languages (0.7%), as well as several programming languages (11.3%) and mathematical data (4.7%; included in the total English data above).

Final language proportions used to train the Luciole models, after up and down-sampling, can be found on the respective model cards.

The technical report is coming soon. 


## Dataset Description

### Direct Use

The Luciole Training Dataset is a collection of textual documents designed for language model pretraining.

### Principal Features

Dataset Creation
Curation Rationale
{{ curation_rationale_section | default("[More Information Needed]", true)}}

Personal and Sensitive Information
{{ personal_and_sensitive_information | default("[More Information Needed]", true)}}

Bias, Risks, and Limitations
{{ bias_risks_limitations | default("[More Information Needed]", true)}}

Recommendations
{{ bias_recommendations | default("Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.", true)}}

### Sample Metadata

### Dataset Composition
{{ dataset_structure | default("[More Information Needed]", true)}}

## Downloading the data

### Sample Use in Python


### English web data via ftp

## Details on Data Sources

#### Aya Dataset
* <u>Source</u>: [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset). Licence: Apache 2.0
* <u>Description</u>: "The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community via Aya Annotation Platform from Cohere Labs. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators. This dataset can be used to train, finetune, and evaluate multilingual LLMs" (from the [data card](https://huggingface.co/datasets/CohereLabs/aya_dataset)).
* <u>Citation</u>: Shivalika Singh, Freddie Vargus, and Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzemiński, Hakimeh Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee and Sara Hooker (2024). Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.   [arXiv:2402.06619](https://arxiv.org/abs/2402.06619)
   


#### Claire (French and English)
* <u>Sources</u>:
  * French dataset: [OpenLLM-France/Claire-Dialogue-French-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1).
  * English dataset: [OpenLLM-France/Claire-Dialogue-English-0.1](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1). License: [CC BY-NC-SA 4.0](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1).
* <u>Extracted from</u>: see the datacards for the [French](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1) and [English](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-English-0.1) datasets.
* <u>Description</u>: The Claire datasets are composed of transcripts of spoken conversations -- including parliamentary proceedings, interviews, debates, meetings, and free conversations -- as well as some written conversations from theater plays and written chats. The dataset is designed to help downstream performance of models fine-tuned for tasks requiring the comprehension of spontaneous spoken conversation, such as meeting summarization. Each dialogue is split into speech turns, and each speech turn is labeled with the name of the speaker or a unique identifier. See the composition details for the <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_claire-french_pie.png">French dataset</a> and the <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_claire-english_pie.png">English dataset</a> for a high-level view of the distribution of different types of documents in each dataset.
* <u>Citation</u>: Julie Hunter, Jérôme Louradour, Virgile Rennard, Ismaïl Harrando, Guokan Shang, Jean-Pierre Lorré (2023). The Claire French Dialogue Dataset. [arXiv:2311.16840](https://arxiv.org/abs/2311.16840).


#### Common Corpus
* <u>Source</u>:

#### Common Pile
#### Pile (Uncopyrighted)
* <u>Source</u>: [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted). License: [Other](https://huggingface.co/datasets/monology/pile-uncopyrighted).
* <u>Extracted from</u>: [FreeLaw](https://free.law/), [StackExchange](https://stackexchange.com/), [USPTO Backgrounds](https://bulkdata.uspto.gov/), [DM Mathematics](https://github.com/google-deepmind/mathematics_dataset), [Ubuntu IRC](https://irclogs.ubuntu.com/), [PhilPapers](https://philpapers.org/), NIH ExPorter from [The Pile](https://huggingface.co/datasets/EleutherAI/pile). License: [MIT](https://arxiv.org/pdf/2201.07311).
* <u>Description</u> (from the [Datasheet](https://arxiv.org/abs/2201.07311)):
  * FreeLaw: "The Free Law Project is US registered non-profit that provide access to millions of legal opinions and analytical tools for academic studies in the legal realm."
  * StackExchange: "The StackExchange dataset is a dump of anonymized user-contributed content on the Stack Exchange network, a popular collection of websites centered around user-contributed questions and answers."
  * USPTO Backgrounds: "The USPTO Backgrounds dataset is a set of background sections from patents granted by the United States Patent and Trademark Office, derived from its published bulk archives."
  * DM Mathematics: "The DeepMind Mathematics dataset consists of a collection of mathematical problems such as algebra, arithmetic, calculus, number theory, and probability, formatted as natural language prompts [Saxton et al., 2019](https://arxiv.org/abs/1904.01557)."
  * Ubuntu IRC: "The Ubuntu IRC dataset is derived from the publicly available chatlogs of all Ubunturelated channels on the Freenode IRC chat server."
  * PhilPapers: a dataset of open access philosophy publications from an international database maintained by the Center for Digital Philosophy at the University of Western Ontario.
  * NIH ExPORTER: "The NIH Grant abstracts provides a bulk-data repository for awarded applications through the ExPORTER4 service covering the fiscal years 1985-present."
* <u>Pre-processing (v1.2 only)</u>:
  * <u>Filtering of PhilPapers</u>: Papers were removed if their language, detected using [Stanza](https://github.com/stanfordnlp/stanza), was not classified as English, French, German, Spanish or Italian.
  * <u>Filtering and text cleaning of Ubuntu IRC</u>: Texts from some channels were excluded to avoid data from languages other than English, French, German, Spanish or Italian and certain encoding errors were fixed (see [code details here](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/text.py#L190)).
* <u>Citations</u>:
  * Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy (2020). "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," [	arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
  * Stella Biderman, Kieran Bicheno, Leo Gao (2022). "Datasheet for the Pile," [arXiv:2201.07311](https://arxiv.org/abs/2201.07311).

#### Croissant Aligned
* <u>Source</u> [OpenLLM-France/Translation-Instruct](https://huggingface.co/datasets/OpenLLM-France/Translation-Instruct)
* <u>Original source</u>: [croissantllm/croissant_dataset_no_web_data](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data/tree/main/aligned_36b) (subset: `aligned_36b`). License: not specified.
* <u>Extracted from</u>: 
  * Translation pairs: [OPUS](https://opus.nlpl.eu/) (99.6% of the data in CroissantAligned). Pairs extracted from OPUS are labeled as "UnbabelFrEn". 
  * Thesis abstracts: French thesis abstract pairs. License: [ETALAB-Licence-Ouverte-v2.0](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).
  * Song lyrics: [lacoccinelle](https://www.lacoccinelle.net). 
* <u>Description</u>: CroissantAligned contains samples of parallel French/English (or English/French) data. Data extracted from OPUS takes the form of sentences pairs, where one sentence is in French and the other is in English. OPUS pairs were passed through a custom pipeline designed to select the highest quality translation examples. Selected pairs are labeled "UnbabelFrEn" in the CroissantAligned dataset. The thesis abstract subset contains thesis abstracts paired with translations written by the thesis authors. The song lyrics are translated by contributors to www.lacoccinelle.net. Parallel data are used to boost the multilingual capabilities of models trained on them ([Faysse et al.,2024](https://arxiv.org/pdf/2402.00786)).
* <u>Pre-processing</u>:
  * <u>Language separation and tagging</u>: The original text field of [the Croissant dataset](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data) contains a sentence or passage in French or English immediately followed by its translation without any indication of which passage is in which language. The first step was thus to split each text into separate, monolingual passages and tag each passage with the appropriate language code, identified automatically using the [langid library](https://pypi.org/project/langid/) (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/data.py#L1407)). In the Lucie Training Dataset, the `extra` metadata field for CroissantAligned contains separate keys, `text_fr` for French and `text_en` for English, that stores the texts separately.
  * <u>Random combination of texts prefixed by language</u>: To create the text values, each monolingual text was repaired with its translation, but random separators and various methods of prefixing the text with the language (name or code) were added.
  This was done as a precaution to prevent models trained on this data from switching languages when generating text and can be seen as a very basic instruction to translate the source (first) text into the target (second) text (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/data.py#L1458)).
* <u>Citation</u>: Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F.T. Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo (2024). "CroissantLLM: A Truly Bilingual French-English Language Model," [arXiv:2402.00786](https://arxiv.org/abs/2402.00786).

#### CulturaX
* <u>Source</u>: Licence: 
* <u>Description</u>: 
* <u>Citation</u>: 

#### RedPajama (v2)
* <u>Source</u>: [togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2). License: [Apache 2.0](https://github.com/togethercomputer/RedPajama-Data) (data preparation code), Not specified (data) but see [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).
* <u>Extracted from</u>: [Common Crawl](https://commoncrawl.org/).
* <u>Description</u>: "RedPajama-V2 is an open dataset for training large language models. The dataset includes over 100B text documents coming from 84 CommonCrawl snapshots and processed using the [CCNet](https://github.com/facebookresearch/cc_net) pipeline. Out of these, there are 30B documents in the corpus that additionally come with quality signals, and 20B documents that are deduplicated" (from [GitHub](https://github.com/togethercomputer/RedPajama-Data)). Most recent crawl for French data in the Lucie Training Dataset v1.1: 2023-14. (For more details on the time periods covered by crawls in this dataset see the composition details for <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_redpajama-french_histogram.png">French</a>, <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_redpajama-german_histogram.png">German</a>, <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_redpajama-italian_histogram.png">Italian</a> and <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_redpajama-spanish_histogram.png">Spanish</a>.)
* <u>Pre-processing and deduplication</u>: 
  * <u> Url filtering: </u>
    * <u>Removing duplicate urls</u>: urls were removed if their base domain overlapped with a dataset already in the Lucie Training Dataset (e.g., "theses.fr") in order to increase diversity of content (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/webdata_processing/base.py#L154)).
    * <u>Filtering certain toxic content</u>: urls from a list of blacklisted content were removed (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/webdata_processing/base.py#L177)).
    * <u>Filtering by robots.txt files</u>: we collect robots.txt and remove all documents for which CCBot is disallowed or for which we failed to collect information as of July 2024 in an effort to select data free from opt-out evidence according to the 4th article of the copyright European directive (2019).
  * <u>Filtering</u>: A series of filters were applied using [quality signals](https://github.com/togethercomputer/RedPajama-Data?tab=readme-ov-file#quality-annotations)  already available in the dataset. This includes (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/d9cccb7bfac37b8c8285f9c04aa67d907ce475f0/webdata_processing/base.py#L36)):
    * CCnet perplexity below 10 or above 1000 
    * C4 filtering (including removal of documents that contain toxic words)
    * Gopher filtering and repetition removal
    * Redpajama document deduplication
  * <u>Removal of personally identifying information (PII)</u>: email addresses and ip addresses were replaced with random addresses (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/webdata_processing/base.py#L301)).
  * <u>MinHash deduplication</u> was performed on each snapshot and language independantly as proposed in FineWeb. For minhash configuration [see code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/webdata_processing/minhash.py#L63). 

  The [Datatrove](https://github.com/huggingface/datatrove) library was used to perform both filtering and deduplication stages.

* <u>Citation</u>: Together Computer (2023). "RedPajama-Data-v2: an Open Dataset with 30 Trillion Tokens for Training Large Language Models," [GitHub](https://github.com/togethercomputer/RedPajama-Data).


#### DCLM Dolmino (via ftp)

#### Dolma3 Longmino



#### Europarl and EuroparlAligned
* <u>Sources</u>: 
  * `fr-en`, `es-en`, `it-en` parallel data: [Europarl v7](https://www.statmt.org/europarl/v7/). License: [Open](https://www.statmt.org/europarl/).
  * `fr`, `en`, `de`, `es` monolingual data and `de-fr` parallel data: [Europarl v10](https://www.statmt.org/europarl/v10/training-monolingual/). License: [Open](https://www.statmt.org/europarl/).
* <u>Description</u>: "The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian), Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene), Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek. The goal of the extraction and processing was to generate sentence aligned text for statistical machine translation systems" ([www.statmt.org](https://www.statmt.org/europarl/)).
* <u>Pre-processing</u>:
  * <u>Random combination of aligned texts prefixed by language</u>: The same process as used for the [CroissantAligned](#croissantaligned) dataset was applied to the EuroparlAligned dataset (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/data.py#L1350)).
  In the Lucie Training Dataset, the `extra` field in the metadata for EuroparlAligned provides texts in the two languages under the sub-fields `text_1` and `text_2`, and the corresponding language codes under `lang_1` and `lang_2`.
* <u>Citation</u>: Philipp Koehn (2005). "Europarl: A Parallel Corpus for Statistical Machine Translation," MT Summit. 

#### Eurovoc
* <u>Source</u>:   [EuropeanParliament/Eurovoc](https://huggingface.co/datasets/EuropeanParliament/Eurovoc). License: [EUPL 1.1](https://huggingface.co/datasets/EuropeanParliament/Eurovoc).
* <u>Extracted from</u>: [Cellar](https://op.europa.eu/en/web/cellar). License: [CC BY-4.0](https://op.europa.eu/en/web/about-us/legal-notices/publications-office-of-the-european-union-copyright).
* <u>Description</u>: A collection of mutlilingual documents from the data repository of the Publications Office of the European Union annotated with Eurovoc labels. The corpus contains legal, policy-related, historical and organizational information about the EU. Dataset containing text retrieved through OCR.
* <u>Pre-processing</u>:
  * <u>Filtering</u>:
  To filter out documents with excessive OCR errors, the dataset was refined by discarding texts with a perplexity higher than 1500,
  measured using a CCNET model on the target language (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1590)).
  The code to compute CCNET perplexity, parallelizing on parquet files, is [available here](https://github.com/OpenLLM-France/Lucie-dataset-filtering).
  * <u>Text cleaning</u>:
  Mentions of Credit Institutions Directives (CID) that appears in the raw texts such as `(cid:146)` were removed.
* <u>Citations</u>:
  * Ilias Chalkidis, Emmanouil Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos (2019). "[Extreme Multi-Label Legal Text Classification: A Case Study in EU Legislation](https://arxiv.org/pdf/1905.10892)," Proceedings of the Natural Legal Language Processing Workshop 2019, pages 78–87, Minneapolis, Minnesota. Association for Computational Linguistics.
  * Ilias Chalkidis,  Manos Fergadiotis, Prodromos Malakasiotis and Ion Androutsopoulos (2019). "[Large-Scale Multi-Label Text Classification on EU Legislation](https://arxiv.org/pdf/1906.02192)," Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers).
  * Andrei-Marius Avram, Vasile Pais, and Dan Ioan Tufis (2021). "[PyEuroVoc: A Tool for Multilingual Legal Document Classification with EuroVoc Descriptors](https://arxiv.org/pdf/2108.01139)," Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 92–101, Held Online. INCOMA Ltd.
  * Zein Shaheen, Gerhard Wohlgenannt and Erwin Filtz (2020). "Large scale legal text classification using transformer models," [arXiv:2010.12871](https://arxiv.org/abs/2010.12871v1).

#### FineMath

#### FineWeb 2

#### FineWeb HQ (via ftp)

#### FineWeb 2 HQ

#### FineWebEdu (via ftp)
* <u>Source</u>: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
* <u>Extracted from</u>: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). License: [ODC-BY](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
* <u>Description</u>: A 1.3 trillion token selection from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), which contains 15 trillion tokens of curated data from 96 Common Crawl dumps. Content in FineWebEdu has been selected by a custom designed classifier for its high-quality, educational content. Most recent crawl: 2024-10 (see <a href="https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset/blob/main/figures/fig_distribution_finewebedu-english_histogram.png">composition details</a> for information about the crawls included in this dataset.)
* <u>Pre-processing</u>: 
  * <u>Removing duplicate urls</u>: urls were removed if their base domain overlapped with a dataset already in the Lucie Training Dataset (e.g., "philpapers.org") in order to increase diversity of content (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/text.py#L843))
  * <u>Filtering by robots.txt files</u>: we collect robots.txt and remove all documents for which CCBot is disallowed or for which we failed to collect information as of July 2024 in an effort to select data free from opt-out evidence according to the 4th article of the copyright European directive (2019).
* <u>Citation</u>: Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," [	arXiv:2406.17557](https://arxiv.org/abs/2406.17557).

#### GallicaMonographies (just Gallica on the repo)
* <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Books](https://huggingface.co/datasets/PleIAs/French-PD-Books). License: Public domain.
* <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
* <u>Description</u>: A large collection of French monographies in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)). Dataset containing text retrieved through OCR.
* <u>Pre-processing</u>:
  * <u>Text cleaning for v1.1</u>:
  To filter out documents with excessive OCR errors, the dataset was split into chunks and chunks were kept if the source language was detected as French by [FastText](https://github.com/facebookresearch/fastText) with a confidence score of 0.65 or above, and the perplexity score, as measured using a CCNET model in French, was between 10 and 1000.
  The code to compute CCNET perplexity, parallelizing on parquet files, is [available here](https://github.com/OpenLLM-France/Lucie-dataset-filtering).
  * <u>Filtering for v1.2</u>: Using OCR scores provided in the metadata of the source corpus, documents with an OCR score of less than 90 out of 100 were filtered out.

#### GallicaPress
* <u>Source</u>: Corpus contributed by OpenLLM partners. A version is also published here: [PleIAs/French-PD-Newspapers](https://huggingface.co/datasets/PleIAs/French-PD-Newspapers). License: Public domain.
* <u>Extracted from</u>: [Gallicagram](https://shiny.ens-paris-saclay.fr/app/gallicagram).
* <u>Description</u>: A large collection of French newspapers and periodicals in the public domain made available through the French National Library ([Gallica](https://gallica.bnf.fr/accueil/fr/content/accueil-fr?mode=desktop)). Dataset containing text retrieved through OCR.
* <u>Pre-processing</u>:
  * <u>Text cleaning for v1.1</u>:
  To filter out documents with excessive OCR errors, the dataset was split into chunks and chunks were kept if the source language was detected as French by [FastText](https://github.com/facebookresearch/fastText) with a confidence score of 0.65 or above, and the perplexity score, as measured using a CCNET model in French, was between 10 and 1000 (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1840)).
  The code to compute CCNET perplexity, parallelizing on parquet files, is [available here](https://github.com/OpenLLM-France/Lucie-dataset-filtering).
  * <u>Filtering for v1.2</u>: Using OCR scores provided in the metadata of the source corpus, documents with an OCR score of less than 90 out of 100 were filtered out.

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
* <u>Source</u>: [bigscience-data/roots_fr_hal_archives_ouvertes](https://huggingface.co/datasets/bigscience-data/roots_fr_hal_archives_ouvertes). License: Roots dataset.
* <u>Extracted from</u>: [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of scientific papers and manuscripts distributed through the open science platform HAL. Dataset containing text retrieved through OCR.
* <u>Pre-processing</u>:
  * <u>Filtering</u>:
  To filter out documents with excessive OCR errors, the dataset was refined by discarding texts with a perplexity higher than 930,
  measured using a CCNET model in French (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1929)).
  The code to compute CCNET perplexity, parallelizing on parquet files, is [available here](https://github.com/OpenLLM-France/Lucie-dataset-filtering).
* <u>Citation</u>: Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, Jörg Frohberg, Mario Šaško, Quentin Lhoest, Angelina McMillan-Major, Gerard Dupont, Stella Biderman, Anna Rogers, Loubna Ben allal, Francesco De Toni, Giada Pistilli, Olivier Nguyen, Somaieh Nikpoor, Maraim Masoud, Pierre Colombo, Javier de la Rosa, Paulo Villegas, Tristan Thrush, Shayne Longpre, Sebastian Nagel, Leon Weber, Manuel Muñoz, Jian Zhu, Daniel Van Strien, Zaid Alyafeai, Khalid Almubarak, Minh Chien Vu, Itziar Gonzalez-Dios, Aitor Soroa, Kyle Lo, Manan Dey, Pedro Ortiz Suarez, Aaron Gokaslan, Shamik Bose, David Adelani, Long Phan, Hieu Tran, Ian Yu, Suhas Pai, Jenny Chim, Violette Lepercq, Suzana Ilic, Margaret Mitchell, Sasha Alexandra Luccioni, Yacine Jernite (2022). "[The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ce9e92e3de2372a4b93353eb7f3dc0bd-Abstract-Datasets_and_Benchmarks.html)," Advances in Neural Information Processing Systems (NeurIPS), 35, 31809-31826.


#### HPLT 2

#### InfiWebMath


#### INSEE


#### MathPile (Commercial)
* <u>Source</u>: [GAIR/MathPile_Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial). License: [CC BY-SA 4.0](https://huggingface.co/datasets/GAIR/MathPile_Commercial).
* <u>Extracted from</u>: [MathPile](https://huggingface.co/datasets/GAIR/MathPile). License: [CC BY-SA-NC 4.0](https://huggingface.co/datasets/GAIR/MathPile).
* <u>Description</u>: A preprocessed collection of documents focused on math, including Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and web pages from Common Crawl. The content targets a range of levels, from kindergarten through postgraduate level. MathPile_Commercial was obtained by removing documents from MathPile that do not allow commercial use.
* <u>Pre-processing</u>:
  * <u>Formatting</u>: Converted the content of StackExchange questions and answers to match the {"text": value} format, using the following formula:
  ```python
  text = sample["question"]["Body"] + "\n\n".join([answer["Body"] for answer in sample["answers"]])
  ```
* <u>Citation</u>: Zengzhi Wang, Rui Xia and Pengfei Liu (2023). "Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math," [	arXiv:2312.17120](https://export.arxiv.org/abs/2312.17120).


#### MegaMath Web

#### Nemotron Post-Training v2

#### Open Code Reasoning

#### OpenData
* <u>Source</u>: [Nicolas-BZRD/DILA_OPENDATA_FR_2023](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main) (balo, dole, inca, kali, legi and sarde subsets). License: [ODC-BY](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main).
* <u>Extracted from</u>: [OpenData](https://echanges.dila.gouv.fr/OPENDATA/) (Data collection date: October, 2023).
* <u>Description</u>: "The French Government Open Data (DILA) Dataset is a collection of text data extracted from various sources provided by the French government, specifically the Direction de l'information légale et administrative (DILA). This dataset contains a wide range of legal, administrative, and legislative documents. The data has been organized into several categories for easy access and analysis" (from the [dataset card](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main)).
<!-- * <u>Citation</u>: No paper found. -->


#### Open Math Instruct


#### Open Thoughts


#### Paradocs

#### Parlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4). License: [CC BY-SA](https://www.regardscitoyens.org/mentions-legales/).
* <u>Description</u>: A collection of proposed amendments by the French parliament. Documents contain the text of the proposed amendment, the name of the associated law as well as information on who voted on the amendment and what was decided.

#### DiscoursPublics
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>: [Vie Publique](https://www.vie-publique.fr/collection-discours-publics). License: [ETALAB-Licence-Ouverte-v2.0](https://www.vie-publique.fr/mentions-legales).
* <u>Description</u>: A collection of public speeches from the principal public actors in France including speeches from the French President starting from 1974 and from the Prime Minister and members of the government starting from 1980.
* <u>Pre-processing</u>:
  * <u>Text cleaning</u>: the mention of the source url and the number of views were removed from the text.

  #### InterventionsParlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4). License: [CC BY-SA](https://www.regardscitoyens.org/mentions-legales/). 
* <u>Description</u>: Transcripts of remarks made during French parlementary debates. Each text contains a continuous remark by a single speaker. 







#### QuestionsEcritesParlement
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>:  [Regards citoyens](https://www.regardscitoyens.org/#&panel1-4). License: [CC BY-SA](https://www.regardscitoyens.org/mentions-legales/).
* <u>Description</u>: Collection of long written questions, read during a session at the French National Assembly. Questions are asked by a member of the French parliament and addressed to a minister (who is given two months to respond).

#### Pleias SYNTH


#### Scholar 


#### Starcoder Data


#### Starcoder Olmomix

#### TheStack (v1.2)
* <u>Source</u>: [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup). License: [Other](https://huggingface.co/datasets/bigcode/the-stack-dedup) (mixture of copyleft licenses).
* <u>Extracted from</u>: [GitHub](https://github.com/) via [GHarchive](https://www.gharchive.org/). Mixed licenses for source.
* <u>Description</u>: "The Stack contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the [BigCode Project](https://www.bigcode-project.org/), an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets. This is the near-deduplicated version with 3TB data" (from the [dataset card](https://huggingface.co/datasets/bigcode/the-stack-dedup)).
* <u>Citation</u>: Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra and Harm de Vries (2022). "The Stack: 3 TB of permissively licensed source code," [arxiv:2211.15533](https://arxiv.org/abs/2211.15533).


#### Synth FineWeb 2

#### Synth Wikipedia

#### Theses
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>: [theses.fr](https://theses.fr/?domaine=theses) (License: [Licence Ouverte / Open Licence version 2.0](https://www.data.gouv.fr/fr/datasets/theses-soutenues-en-france-depuis-1985/)) and  [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of doctoral theses published in France. Dataset containing text retrieved through OCR.
* <u>Pre-processing</u>:
  * <u>Text cleaning</u>:
    * Title pages about HAL, pages containing a significant fraction of control characters, and duplicate lines were removed (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/cdec8fd6369385455829ab39c2f04bcb1a8a475a/tokenization/text.py#L277)).
    * Because the results of OCR on tables and graphics can give rise to garbage text, the text was cleaned by removing the most suspicious chunks.
    In particular, a chunk was removed if it was not detected as being written in French, English, Spanish, German or Italian, or if the perplexity of a CCNet Language Model on the chunk was higher than 2000 (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1946)).
    The code to compute CCNET perplexity, parallelizing on parquet files, is [available here](https://github.com/OpenLLM-France/Lucie-dataset-filtering).
  * <u>Filtering</u>: Texts with fewer than 1000 words or 10000 characters were removed (see [code details](https://github.com/OpenLLM-France/Lucie-Training/blob/7f1f7efa1288f709662a9067bf2c3db856b850f8/tokenization/data.py#L1975)).

<!-- * <u>Citation</u>: No paper found. -->


#### Vikidia

#### Wikipedia, Wikisource, Wiktionary
* <u>Source</u>: Corpus contributed by LINAGORA Labs (OpenLLM-France).
  Also published here:
  * [OpenLLM-France/wikipedia](https://huggingface.co/datasets/OpenLLM-France/wikipedia)
  * [OpenLLM-France/wikisource](https://huggingface.co/datasets/OpenLLM-France/wikisource)
  * [OpenLLM-France/wiktionary](https://huggingface.co/datasets/OpenLLM-France/wiktionary)
* <u>Extracted from</u>: [Wikimedia dumps](https://dumps.wikimedia.org/other/enterprise_html/runs/). License: [GFDL/CC BY-SA](https://dumps.wikimedia.org/legal.html).
<!-- * <u>Description</u>: TODO -->
<!-- * <u>Pre-processing</u>: TODO -->
<!-- * <u>Citation</u>: No paper found. -->

#### YouTube
* <u>Source</u>: Corpus contributed by LINAGORA Labs and [LeVoiceLab](https://www.levoicelab.org/).
* <u>Extracted from</u>: [YouTube](https://www.youtube.com/). <!-- License: TODO? -->
* <u>Description</u>: French subtitles from videos published with permissive licenses on YouTube. <!-- TODO -->
* <u>Extraction pipeline description</u>:
  * **Searching for YouTube videos likely in French:** Based on searches generated automatically from random sequences of words extracted from a corpus of French journalistic articles (initially obtained through a web-crawling tool applied to publicly accessible news and media sites such as Huffington Post, 20 Minutes, Le Parisien, Actu, Numerama, Slate, etc.).  
  Selection of videos with subtitles labeled as "French," excluding those marked as "automatically generated."  
  *At this stage: 52,778 videos selected, corresponding to 10,654 hours of audio.*  
  * **Selection of videos whose subtitle language classification confirms French with a certain confidence index:**  
  *At this stage: 51,934 videos selected, corresponding to 10,425 hours of audio.*  
  * **Selection of videos whose subtitles contain uppercase, lowercase, and punctuation marks:**  
  This step filters out automatically generated subtitles created with speech recognition tools.  
  *At this stage: 45,488 videos selected, corresponding to 8,904 hours of audio.*  
  * **Extraction of audio tracks from the selected videos.**  
  * **Automatic formatting of transcripts obtained from subtitles:** Removal of emojis, sound event annotations in brackets (like "[Music]") and extra text such as "subtitled by XXX."  (on last seconds of the video).
  * **Selection of videos where an automatic speech recognition tool correctly transcribes the first 30 seconds with a minimum recall and precision rate:**  
  *At this stage: 37,513 videos selected, corresponding to 7,541 hours of audio.*  
  * **Realignment of the transcript:** Ensuring accurate timestamps in the transcriptions based on the subtitles and excluding audios where alignment fails.  
  *At this stage: 36,618 videos selected, corresponding to 6,729 hours of audio.* 




## Citation

✍ Paper coming soon!


## Acknowledgements

We gratefully acknowledge BPI France for funding the OpenLLM France project under the call "Communs numériques pour l’intelligence artificielle générative" ("Digital commons for generative artificial intelligence") and the project numbers DOS0250771 and DOS0250773.

Processing and storage of the Luciole Training Dataset was made possible by computing AI and storage resources by GENCI at IDRIS thanks to the grant 2024-GC011015444 on the supercomputer Jean Zay. We gratefully acknowledge support from GENCI and IDRIS and from Stephane Requena (GENCI) and Pierre-François Lavallée (IDRIS) in particular. 

The Luciole Training Dataset was created by members of [LINAGORA](https://labs.linagora.com/) and [OpenLLM-France](https://openllm-france.fr/), including, in alphabetical order:  
 
Akshay Chaturvedi (LINAGORA)   
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


