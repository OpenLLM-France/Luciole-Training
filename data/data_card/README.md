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
<!-- Olivier -->


#### Personal and Sensitive Information (PII)
<!-- Olivier -->


### Bias, Risks, and Limitations
While we have made strong efforts to only include only open corpora, it is possible that individual documents in those corpora are copyrighted. Similarly, it is possible that some personal information in those corpora has slipped through PII filters. If you find your copyrighted work in the Luciole Training Dataset or mention of your personal details therein, we invite you to contact us at contact@openllm-france.fr.

Despite efforts to filter toxicity in web data, improving filtering methods is an ongoing project, and it is extremely likely that toxic and offensive documents remain in web data. Another likely source of biases comes from older data in the public domain. Historical documents can carry biases related to, for example, gender, skin color, ethnicity, and religion that are not socially acceptable. 

A further limitation of this dataset is that it does not distinguish between variants of different languages. Quebequois French and Metropolitan French, to give just one example, are both classified as "French". In future work, we hope to focus more on regional linguistic diversity.

#### Recommendations
Due to harmful biases potentially conveyed by some documents in the Luciole Training Dataset, models pretrained on this data should undergo careful fine-tuning and alignment before being used for non-research purposes. 

### Sample Metadata
<!-- Olivier -->

### Dataset Composition
<!-- Olivier -->

## Downloading the Data

### Sample Use in Python
<!-- Olivier -->

### Accessing the English Web Data
<!-- Olivier -->

## Details on Data Sources

#### Aya Dataset
* <u>Source</u>: [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset). Licence: Apache 2.0
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
* <u>Source</u>:
<!-- <u>Pre-processing</u>: -->
* <u>Citation</u>:

#### Common Pile (v0.1)
* <u>Source</u>: [common-pile/common-pile-v01-filtered-data](https://huggingface.co/collections/common-pile/common-pile-v01-filtered-data)
* <u>Description</u>: The Common Pile v0.1 is a curated "eight terabyte collection of openly licensed text designed for LLM pretraining. The Common Pile comprises content from 30 sources that span diverse domains including research papers, code, books, encyclopedias, educational materials, audio transcripts, and more" (Kandpal et al., 2025).

<!-- Subsets -->
<!-- <u>Pre-processing</u>: -->
* <u>Citation</u>: Nikhil Kandpal and Brian Lester and Colin Raffel and Sebastian Majstorovic and Stella Biderman and Baber Abbasi and Luca Soldaini and Enrico Shippole and A. Feder Cooper and Aviya Skowron and John Kirchenbauer and Shayne Longpre and Lintang Sutawika and Alon Albalak and Zhenlin Xu and Guilherme Penedo and Loubna Ben Allal and Elie Bakouch and John David Pressman and Honglu Fan and Dashiell Stander and Guangyu Song and Aaron Gokaslan and Tom Goldstein and Brian R. Bartoldson and Bhavya Kailkhura and Tyler Murray (2025). [arXiv:2506.05209](https://arxiv.org/abs/2506.05209)

#### Croissant Aligned
* <u>Source</u>: [OpenLLM-France/Translation-Instruct](https://huggingface.co/datasets/OpenLLM-France/Translation-Instruct)
* <u>Original source</u>: [croissantllm/croissant_dataset_no_web_data](https://huggingface.co/datasets/croissantllm/croissant_dataset_no_web_data/tree/main/aligned_36b) (subset: `aligned_36b`). License: not specified.
* <u>Extracted from</u>: 
  * Translation pairs: [OPUS](https://opus.nlpl.eu/) (99.6% of the data in CroissantAligned). Pairs extracted from OPUS are labeled as "UnbabelFrEn". 
  * Thesis abstracts: French thesis abstract pairs. License: [ETALAB-Licence-Ouverte-v2.0](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).
  * Song lyrics: [lacoccinelle](https://www.lacoccinelle.net). 
* <u>Description</u>: CroissantAligned contains samples of parallel French/English (or English/French) data. Data extracted from OPUS takes the form of sentences pairs, where one sentence is in French and the other is in English. OPUS pairs were passed through a custom pipeline designed to select the highest quality translation examples. Selected pairs are labeled "UnbabelFrEn" in the CroissantAligned dataset. The thesis abstract subset contains thesis abstracts paired with translations written by the thesis authors. The song lyrics are translated by contributors to www.lacoccinelle.net. Parallel data are used to boost the multilingual capabilities of models trained on them ([Faysse et al.,2024](https://arxiv.org/pdf/2402.00786)).
* <u>Citation</u>: Manuel Faysse, Patrick Fernandes, Nuno M. Guerreiro, António Loison, Duarte M. Alves, Caio Corro, Nicolas Boizard, João Alves, Ricardo Rei, Pedro H. Martins, Antoni Bigata Casademunt, François Yvon, André F.T. Martins, Gautier Viaud, Céline Hudelot, Pierre Colombo (2024). "CroissantLLM: A Truly Bilingual French-English Language Model," [arXiv:2402.00786](https://arxiv.org/abs/2402.00786).

#### CulturaX
* <u>Source</u>: Licence: 
* <u>Description</u>: 
* <u>Citation</u>: 

#### DCLM Dolmino (via ftp)
* <u>Source</u>: Licence: 
* <u>Description</u>: 
* <u>Citation</u>: 

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

#### FineMath (??? and InfiMM-WebMath)
* <u>Source</u>: [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath). License: ODC-BY.
* <u>Description</u>: "FineMath consists of 34B tokens (FineMath-3+) and 54B tokens (FineMath-3+ with InfiMM-WebMath-3+) of mathematical educational content filtered from CommonCrawl. To curate this dataset, we trained a mathematical content classifier using annotations generated by LLama-3.1-70B-Instruct. We used the classifier to retain only the most educational mathematics content, focusing on clear explanations and step-by-step problem solving rather than advanced academic papers" (FineMath [data card](https://huggingface.co/datasets/HuggingFaceTB/finemath)).
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín,  Vaibhav Srivastav,  Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra and Thomas Wolf (2025). SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. [arXiv:2502.02737](https://arxiv.org/abs/2502.02737). 

#### FineWeb2
* <u>Source</u>: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). License: ODC-BY.
* <u>Description</u>: FineWeb2 extends the original [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset by adding pretraining data for over 1000 languages. "The data was sourced from 96 CommonCrawl snapshots, spanning the summer of 2013 to April 2024, and processed using datatrove, our large scale data processing library. This carefully deduplicated and filtered dataset comprises roughly 20 terabytes, across 5 billion documents, with over 3 trillion words" (FineWeb2 [data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)). 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Guilherme Penedo,  Hynek Kydlíček,  Vinko Sabolčec,  Bettina Messmer,  Negar Foroutan,  Amir Hossein Kargaran,  Colin Raffel,  Martin Jaggi, Leandro Von Werra and Thomas Wolf (2025). FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language. [arXiv:2506.20920](https://arxiv.org/abs/2506.20920).


#### FineWeb HQ (via ftp)
* <u>Source</u>: [epfml/FineWeb-HQ](https://huggingface.co/datasets/epfml/FineWeb-HQ). License: ODC-BY.
* <u>Description</u>: "FineWeb-HQ is a high-quality, model-filtered pretraining dataset derived as a subset of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). FineWeb-HQ was created by selecting the top 10% of FineWeb documents based on a deep learning classifier trained to identify structured and knowledge-rich samples. This classifier uses XLM-RoBERTa embeddings to score documents."
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Bettina Messmer, Vinko Sabolčec and Martin Jaggi (2025). Enhancing Multilingual LLM Pretraining with Model-Based Data Selection. [arXiv:2502.10361](https://arxiv.org/abs/2502.10361).

#### FineWeb 2 HQ
* <u>Source</u>: [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ). License: ODC-BY.
* <u>Description</u>: "FineWeb2-HQ is a high-quality, model-filtered pretraining dataset derived as a subset of [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), spanning 20 languages. It enables around 6x faster pretraining compared to the base dataset. FineWeb2-HQ was created by selecting the top 10% quality documents of FineWeb2 in each language, based on scores assigned by a deep learning classifier trained to identify structured and knowledge-rich samples using XLM-RoBERTa embeddings."
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: Bettina Messmer, Vinko Sabolčec and Martin Jaggi (2025). Enhancing Multilingual LLM Pretraining with Model-Based Data Selection. [arXiv:2502.10361](https://arxiv.org/abs/2502.10361).

#### FineWebEdu (via ftp)
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
* <u>Source</u>: Corpus processed by OpenLLM partners and published separately as []().
* <u>Original source</u>:  based on [almanach/halvest](https://huggingface.co/datasets/almanach/halvest). License: [HAL license](https://doc.hal.science/en/legal-aspects/).
* <u>Extracted from</u>: [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of scientific papers and manuscripts distributed through the open science platform HAL. Dataset containing text retrieved through OCR.
<!-- * <u>Pre-processing</u>: -->
  
* <u>Citation</u>: 


#### HPLT 2
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 


#### INSEE
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### MathPile (Commercial)
* <u>Source</u>: [GAIR/MathPile_Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial). License: [CC BY-SA 4.0](https://huggingface.co/datasets/GAIR/MathPile_Commercial).
* <u>Extracted from</u>: [MathPile](https://huggingface.co/datasets/GAIR/MathPile). License: [CC BY-SA-NC 4.0](https://huggingface.co/datasets/GAIR/MathPile).
* <u>Description</u>: A preprocessed collection of documents focused on math, including Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, and web pages from Common Crawl. The content targets a range of levels, from kindergarten through postgraduate level. MathPile_Commercial was obtained by removing documents from MathPile that do not allow commercial use.
* <u>Citation</u>: Zengzhi Wang, Rui Xia and Pengfei Liu (2023). "Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math," [	arXiv:2312.17120](https://export.arxiv.org/abs/2312.17120).


#### MegaMath Web
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Nemotron Post-Training v2
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Open Code Reasoning
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### OpenData
* <u>Source</u>: [Nicolas-BZRD/DILA_OPENDATA_FR_2023](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main) (balo, dole, inca, kali, legi and sarde subsets). License: [ODC-BY](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main).
* <u>Extracted from</u>: [OpenData](https://echanges.dila.gouv.fr/OPENDATA/) (Data collection date: October, 2023).
* <u>Description</u>: "The French Government Open Data (DILA) Dataset is a collection of text data extracted from various sources provided by the French government, specifically the Direction de l'information légale et administrative (DILA). This dataset contains a wide range of legal, administrative, and legislative documents. The data has been organized into several categories for easy access and analysis" (from the [dataset card](https://huggingface.co/datasets/Nicolas-BZRD/DILA_OPENDATA_FR_2023/tree/main)).
<!-- * <u>Citation</u>: No paper found. -->


#### Open Math Instruct
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Open Thoughts
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 


#### Paradocs
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

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
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Scholar 
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Starcoder Data
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Starcoder Olmomix
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### TheStack (v1.2)
* <u>Source</u>: [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup). License: [Other](https://huggingface.co/datasets/bigcode/the-stack-dedup) (mixture of copyleft licenses).
* <u>Extracted from</u>: [GitHub](https://github.com/) via [GHarchive](https://www.gharchive.org/). Mixed licenses for source.
* <u>Description</u>: "The Stack contains over 6TB of permissively-licensed source code files covering 358 programming languages. The dataset was created as part of the [BigCode Project](https://www.bigcode-project.org/), an open scientific collaboration working on the responsible development of Large Language Models for Code (Code LLMs). The Stack serves as a pre-training dataset for Code LLMs, i.e., code-generating AI systems which enable the synthesis of programs from natural language descriptions as well as other from code snippets. This is the near-deduplicated version with 3TB data" (from the [dataset card](https://huggingface.co/datasets/bigcode/the-stack-dedup)).
* <u>Citation</u>: Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra and Harm de Vries (2022). "The Stack: 3 TB of permissively licensed source code," [arxiv:2211.15533](https://arxiv.org/abs/2211.15533).


#### Synth FineWeb 2
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Synth Wikipedia
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

#### Theses
* <u>Source</u>: Corpus contributed by OpenLLM partners.
* <u>Extracted from</u>: [theses.fr](https://theses.fr/?domaine=theses) (License: [Licence Ouverte / Open Licence version 2.0](https://www.data.gouv.fr/fr/datasets/theses-soutenues-en-france-depuis-1985/)) and  [HAL](https://hal.science/) ([Open access](https://about.hal.science/)).
* <u>Description</u>: A collection of doctoral theses published in France. Dataset containing text retrieved through OCR.

<!-- * <u>Citation</u>: No paper found. -->


#### Vikidia
* <u>Source</u>: Licence: 
* <u>Description</u>: 
<!-- * <u>Pre-processing</u>: -->
* <u>Citation</u>: 

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


