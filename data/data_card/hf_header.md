---
language:
  - ar
  - br
  - ca
  - co
  - de
  - en
  - es
  - eu
  - fr
  - frp
  - it
  - nl
  - oc
  - pcd
  - pt
  - acf
  - crs
  - gcf
  - gcr
  - rcf
  - ty
  - wa
multilinguality:
  - multilingual
license: cc-by-sa-4.0
pretty_name: Luciole Training Dataset
size_categories:
  - n>1T
task_categories:
  - text-generation
configs:
  - config_name: all
    data_files:
      - split: all
        path: data/**/*.parquet
      - split: ar
        path: data/**/ar/**/*.parquet
      - split: br
        path: data/**/br/**/*.parquet
      - split: ca
        path: data/**/ca/**/*.parquet
      - split: co
        path: data/**/co/**/*.parquet
      - split: de
        path: data/**/de/**/*.parquet
      - split: en
        path: data/**/en/**/*.parquet
      - split: es
        path: data/**/es/**/*.parquet
      - split: eu
        path: data/**/eu/**/*.parquet
      - split: fr
        path: data/**/fr/**/*.parquet
      - split: frp
        path: data/**/frp/**/*.parquet
      - split: it
        path: data/**/it/**/*.parquet
      - split: nl
        path: data/**/nl/**/*.parquet
      - split: oc
        path: data/**/oc/**/*.parquet
      - split: pcd
        path: data/**/pcd/**/*.parquet
      - split: pt
        path: data/**/pt/**/*.parquet
      - split: acf
        path: data/**/acf/**/*.parquet
      - split: crs
        path: data/**/crs/**/*.parquet
      - split: gcf
        path: data/**/gcf/**/*.parquet
      - split: gcr
        path: data/**/gcr/**/*.parquet
      - split: rcf
        path: data/**/rcf/**/*.parquet
      - split: ty
        path: data/**/ty/**/*.parquet
      - split: wa
        path: data/**/wa/**/*.parquet
      - split: de_fr
        path: data/**/de-fr/**/*.parquet
      - split: en_de
        path: data/**/en-de/**/*.parquet
      - split: en_es
        path: data/**/en-es/**/*.parquet
      - split: en_fr
        path: data/**/en-fr/**/*.parquet
      - split: en_it
        path: data/**/en-it/**/*.parquet
      - split: en_nl
        path: data/**/en-nl/**/*.parquet
      - split: en_pt
        path: data/**/en-pt/**/*.parquet
      - split: es_pt
        path: data/**/es-pt/**/*.parquet
      - split: code
        path:
          - data/stack_edu/**/*.parquet
          - data/starcoder_data/**/*.parquet
          - data/starcoder_olmomix/**/*.parquet
  - config_name: Aya
    data_files:
      - split: all
        path: data/aya/**/*.parquet
      - split: ar
        path: data/aya/ar/**/*.parquet
      - split: de
        path: data/aya/de/**/*.parquet
      - split: en
        path: data/aya/en/**/*.parquet
      - split: es
        path: data/aya/es/**/*.parquet
      - split: eu
        path: data/aya/eu/**/*.parquet
      - split: fr
        path: data/aya/fr/**/*.parquet
      - split: it
        path: data/aya/it/**/*.parquet
      - split: nl
        path: data/aya/nl/**/*.parquet
      - split: pt
        path: data/aya/pt/**/*.parquet
  - config_name: Claire
    data_files:
      - split: all
        path: data/claire/**/*.parquet
      - split: en
        path: data/claire/en/**/*.parquet
      - split: fr
        path: data/claire/fr/**/*.parquet
  - config_name: CommonCorpus-bnl-newspapers-1841-1879
    data_files:
      - split: all
        path: data/common_corpus/open-culture/bnl-newspapers-1841-1879/**/*.parquet
      - split: de
        path: data/common_corpus/open-culture/bnl-newspapers-1841-1879/de/**/*.parquet
      - split: fr
        path: data/common_corpus/open-culture/bnl-newspapers-1841-1879/fr/**/*.parquet
      - split: it
        path: data/common_corpus/open-culture/bnl-newspapers-1841-1879/it/**/*.parquet
      - split: nl
        path: data/common_corpus/open-culture/bnl-newspapers-1841-1879/nl/**/*.parquet
  - config_name: CommonCorpus-eurlex
    data_files:
      - split: all
        path: data/common_corpus/open-government/eurlex/**/*.parquet
      - split: de
        path: data/common_corpus/open-government/eurlex/de/**/*.parquet
      - split: en
        path: data/common_corpus/open-government/eurlex/en/**/*.parquet
      - split: es
        path: data/common_corpus/open-government/eurlex/es/**/*.parquet
      - split: fr
        path: data/common_corpus/open-government/eurlex/fr/**/*.parquet
      - split: it
        path: data/common_corpus/open-government/eurlex/it/**/*.parquet
      - split: nl
        path: data/common_corpus/open-government/eurlex/nl/**/*.parquet
      - split: pt
        path: data/common_corpus/open-government/eurlex/pt/**/*.parquet
  - config_name: CommonCorpus-gatt-library
    data_files:
      - split: all
        path: data/common_corpus/open-government/gatt-library/**/*.parquet
      - split: de
        path: data/common_corpus/open-government/gatt-library/de/**/*.parquet
      - split: en
        path: data/common_corpus/open-government/gatt-library/en/**/*.parquet
      - split: es
        path: data/common_corpus/open-government/gatt-library/es/**/*.parquet
      - split: fr
        path: data/common_corpus/open-government/gatt-library/fr/**/*.parquet
  - config_name: CommonCorpus-oecd
    data_files:
      - split: all
        path: data/common_corpus/open-government/oecd/**/*.parquet
      - split: de
        path: data/common_corpus/open-government/oecd/de/**/*.parquet
      - split: en
        path: data/common_corpus/open-government/oecd/en/**/*.parquet
      - split: es
        path: data/common_corpus/open-government/oecd/es/**/*.parquet
      - split: fr
        path: data/common_corpus/open-government/oecd/fr/**/*.parquet
      - split: it
        path: data/common_corpus/open-government/oecd/it/**/*.parquet
      - split: nl
        path: data/common_corpus/open-government/oecd/nl/**/*.parquet
  - config_name: CommonCorpus-tedeutenders
    data_files:
      - split: all
        path: data/common_corpus/open-government/tedeutenders/**/*.parquet
      - split: ca
        path: data/common_corpus/open-government/tedeutenders/ca/**/*.parquet
      - split: de
        path: data/common_corpus/open-government/tedeutenders/de/**/*.parquet
      - split: en
        path: data/common_corpus/open-government/tedeutenders/en/**/*.parquet
      - split: es
        path: data/common_corpus/open-government/tedeutenders/es/**/*.parquet
      - split: fr
        path: data/common_corpus/open-government/tedeutenders/fr/**/*.parquet
      - split: it
        path: data/common_corpus/open-government/tedeutenders/it/**/*.parquet
      - split: nl
        path: data/common_corpus/open-government/tedeutenders/nl/**/*.parquet
      - split: pt
        path: data/common_corpus/open-government/tedeutenders/pt/**/*.parquet
  - config_name: CommonCorpus-wto
    data_files:
      - split: all
        path: data/common_corpus/open-government/wto/**/*.parquet
      - split: ar
        path: data/common_corpus/open-government/wto/ar/**/*.parquet
      - split: de
        path: data/common_corpus/open-government/wto/de/**/*.parquet
      - split: en
        path: data/common_corpus/open-government/wto/en/**/*.parquet
      - split: es
        path: data/common_corpus/open-government/wto/es/**/*.parquet
      - split: fr
        path: data/common_corpus/open-government/wto/fr/**/*.parquet
      - split: it
        path: data/common_corpus/open-government/wto/it/**/*.parquet
      - split: nl
        path: data/common_corpus/open-government/wto/nl/**/*.parquet
      - split: pt
        path: data/common_corpus/open-government/wto/pt/**/*.parquet
  - config_name: CommonPile-arxiv_abstracts_filtered
    data_files:
      - split: all
        path: data/common_pile/arxiv_abstracts_filtered/**/*.parquet
      - split: en
        path: data/common_pile/arxiv_abstracts_filtered/en/**/*.parquet
  - config_name: CommonPile-arxiv_papers_filtered
    data_files:
      - split: all
        path: data/common_pile/arxiv_papers_filtered/**/*.parquet
      - split: en
        path: data/common_pile/arxiv_papers_filtered/en/**/*.parquet
  - config_name: CommonPile-biodiversity_heritage_library_filtered
    data_files:
      - split: all
        path: data/common_pile/biodiversity_heritage_library_filtered/**/*.parquet
      - split: en
        path: data/common_pile/biodiversity_heritage_library_filtered/en/**/*.parquet
  - config_name: CommonPile-caselaw_access_project_filtered
    data_files:
      - split: all
        path: data/common_pile/caselaw_access_project_filtered/**/*.parquet
      - split: en
        path: data/common_pile/caselaw_access_project_filtered/en/**/*.parquet
  - config_name: CommonPile-data_provenance_initiative_filtered
    data_files:
      - split: all
        path: data/common_pile/data_provenance_initiative_filtered/**/*.parquet
      - split: en
        path: data/common_pile/data_provenance_initiative_filtered/en/**/*.parquet
  - config_name: CommonPile-doab_filtered
    data_files:
      - split: all
        path: data/common_pile/doab_filtered/**/*.parquet
      - split: en
        path: data/common_pile/doab_filtered/en/**/*.parquet
  - config_name: CommonPile-foodista_filtered
    data_files:
      - split: all
        path: data/common_pile/foodista_filtered/**/*.parquet
      - split: en
        path: data/common_pile/foodista_filtered/en/**/*.parquet
  - config_name: CommonPile-github_archive_filtered
    data_files:
      - split: all
        path: data/common_pile/github_archive_filtered/**/*.parquet
      - split: en
        path: data/common_pile/github_archive_filtered/en/**/*.parquet
  - config_name: CommonPile-library_of_congress_filtered
    data_files:
      - split: all
        path: data/common_pile/library_of_congress_filtered/**/*.parquet
      - split: en
        path: data/common_pile/library_of_congress_filtered/en/**/*.parquet
  - config_name: CommonPile-libretexts_filtered
    data_files:
      - split: all
        path: data/common_pile/libretexts_filtered/**/*.parquet
      - split: en
        path: data/common_pile/libretexts_filtered/en/**/*.parquet
  - config_name: CommonPile-news_filtered
    data_files:
      - split: all
        path: data/common_pile/news_filtered/**/*.parquet
      - split: en
        path: data/common_pile/news_filtered/en/**/*.parquet
  - config_name: CommonPile-oercommons_filtered
    data_files:
      - split: all
        path: data/common_pile/oercommons_filtered/**/*.parquet
      - split: en
        path: data/common_pile/oercommons_filtered/en/**/*.parquet
  - config_name: CommonPile-peS2o_filtered
    data_files:
      - split: all
        path: data/common_pile/peS2o_filtered/**/*.parquet
      - split: en
        path: data/common_pile/peS2o_filtered/en/**/*.parquet
  - config_name: CommonPile-pre_1929_books_filtered
    data_files:
      - split: all
        path: data/common_pile/pre_1929_books_filtered/**/*.parquet
      - split: en
        path: data/common_pile/pre_1929_books_filtered/en/**/*.parquet
  - config_name: CommonPile-pressbooks_filtered
    data_files:
      - split: all
        path: data/common_pile/pressbooks_filtered/**/*.parquet
      - split: en
        path: data/common_pile/pressbooks_filtered/en/**/*.parquet
  - config_name: CommonPile-public_domain_review_filtered
    data_files:
      - split: all
        path: data/common_pile/public_domain_review_filtered/**/*.parquet
      - split: en
        path: data/common_pile/public_domain_review_filtered/en/**/*.parquet
  - config_name: CommonPile-pubmed_filtered
    data_files:
      - split: all
        path: data/common_pile/pubmed_filtered/**/*.parquet
      - split: en
        path: data/common_pile/pubmed_filtered/en/**/*.parquet
  - config_name: CommonPile-python_enhancement_proposals_filtered
    data_files:
      - split: all
        path: data/common_pile/python_enhancement_proposals_filtered/**/*.parquet
      - split: en
        path: data/common_pile/python_enhancement_proposals_filtered/en/**/*.parquet
  - config_name: CommonPile-regulations_filtered
    data_files:
      - split: all
        path: data/common_pile/regulations_filtered/**/*.parquet
      - split: en
        path: data/common_pile/regulations_filtered/en/**/*.parquet
  - config_name: CommonPile-stackexchange_filtered
    data_files:
      - split: all
        path: data/common_pile/stackexchange_filtered/**/*.parquet
      - split: en
        path: data/common_pile/stackexchange_filtered/en/**/*.parquet
  - config_name: CommonPile-ubuntu_irc_filtered
    data_files:
      - split: all
        path: data/common_pile/ubuntu_irc_filtered/**/*.parquet
      - split: en
        path: data/common_pile/ubuntu_irc_filtered/en/**/*.parquet
  - config_name: CommonPile-youtube_filtered
    data_files:
      - split: all
        path: data/common_pile/youtube_filtered/**/*.parquet
      - split: en
        path: data/common_pile/youtube_filtered/en/**/*.parquet
  - config_name: CroissantAligned
    data_files:
      - split: all
        path: data/croissant_aligned/**/*.parquet
      - split: en_fr
        path: data/croissant_aligned/en-fr/**/*.parquet
  - config_name: Culturax
    data_files:
      - split: all
        path: data/culturax/**/*.parquet
      - split: fr
        path: data/culturax/fr/**/*.parquet
  - config_name: Dolma3Longmino-lc_synth-cwe
    data_files:
      - split: all
        path: data/dolma3_longmino/lc_synth-cwe/**/*.parquet
      - split: en
        path: data/dolma3_longmino/lc_synth-cwe/en/**/*.parquet
  - config_name: Dolma3Longmino-lc_synth-rex
    data_files:
      - split: all
        path: data/dolma3_longmino/lc_synth-rex/**/*.parquet
      - split: en
        path: data/dolma3_longmino/lc_synth-rex/en/**/*.parquet
  - config_name: Dolma3Longmino-olmocr_science_pdfs
    data_files:
      - split: all
        path: data/dolma3_longmino/olmocr_science_pdfs/**/*.parquet
      - split: en
        path: data/dolma3_longmino/olmocr_science_pdfs/en/**/*.parquet
  - config_name: Europarl
    data_files:
      - split: all
        path: data/europarl/**/*.parquet
      - split: de
        path: data/europarl/de/**/*.parquet
      - split: de_fr
        path: data/europarl/de-fr/**/*.parquet
      - split: en
        path: data/europarl/en/**/*.parquet
      - split: en_de
        path: data/europarl/en-de/**/*.parquet
      - split: en_es
        path: data/europarl/en-es/**/*.parquet
      - split: en_fr
        path: data/europarl/en-fr/**/*.parquet
      - split: en_it
        path: data/europarl/en-it/**/*.parquet
      - split: en_pt
        path: data/europarl/en-pt/**/*.parquet
      - split: es
        path: data/europarl/es/**/*.parquet
      - split: es_pt
        path: data/europarl/es-pt/**/*.parquet
      - split: fr
        path: data/europarl/fr/**/*.parquet
      - split: it
        path: data/europarl/it/**/*.parquet
      - split: pt
        path: data/europarl/pt/**/*.parquet
  - config_name: Eurovoc
    data_files:
      - split: all
        path: data/eurovoc/**/*.parquet
      - split: ar
        path: data/eurovoc/ar/**/*.parquet
      - split: ca
        path: data/eurovoc/ca/**/*.parquet
      - split: de
        path: data/eurovoc/de/**/*.parquet
      - split: en
        path: data/eurovoc/en/**/*.parquet
      - split: es
        path: data/eurovoc/es/**/*.parquet
      - split: fr
        path: data/eurovoc/fr/**/*.parquet
      - split: it
        path: data/eurovoc/it/**/*.parquet
      - split: nl
        path: data/eurovoc/nl/**/*.parquet
      - split: pt
        path: data/eurovoc/pt/**/*.parquet
  - config_name: Finemath-3plus
    data_files:
      - split: all
        path:
          - data/finemath/en/score_3/**/*.parquet
          - data/finemath/en/score_4/**/*.parquet
          - data/finemath/en/score_5/**/*.parquet
      - split: en
        path:
          - data/finemath/en/score_3/**/*.parquet
          - data/finemath/en/score_4/**/*.parquet
          - data/finemath/en/score_5/**/*.parquet
  - config_name: Finemath-4plus
    data_files:
      - split: all
        path:
          - data/finemath/en/score_4/**/*.parquet
          - data/finemath/en/score_5/**/*.parquet
      - split: en
        path:
          - data/finemath/en/score_4/**/*.parquet
          - data/finemath/en/score_5/**/*.parquet
  - config_name: Fineweb2
    data_files:
      - split: all
        path: data/fineweb2/**/*.parquet
      - split: acf
        path: data/fineweb2/acf/**/*.parquet
      - split: ar
        path:
          - data/fineweb2/ar/score_4/**/*.parquet
          - data/fineweb2/ar/score_3/**/*.parquet
          - data/fineweb2/ar/score_2/**/*.parquet
          - data/fineweb2/ar/score_1/**/*.parquet
          - data/fineweb2/ar/score_0/**/*.parquet
      - split: br
        path: data/fineweb2/br/**/*.parquet
      - split: ca
        path: data/fineweb2/ca/**/*.parquet
      - split: co
        path: data/fineweb2/co/**/*.parquet
      - split: crs
        path: data/fineweb2/crs/**/*.parquet
      - split: de
        path:
          - data/fineweb2/de/score_4/**/*.parquet
          - data/fineweb2/de/score_3/**/*.parquet
          - data/fineweb2/de/score_2/**/*.parquet
          - data/fineweb2/de/score_1/**/*.parquet
          - data/fineweb2/de/score_0/**/*.parquet
      - split: es
        path:
          - data/fineweb2/es/score_4/**/*.parquet
          - data/fineweb2/es/score_3/**/*.parquet
          - data/fineweb2/es/score_2/**/*.parquet
          - data/fineweb2/es/score_1/**/*.parquet
          - data/fineweb2/es/score_0/**/*.parquet
      - split: eu
        path: data/fineweb2/eu/**/*.parquet
      - split: fr
        path:
          - data/fineweb2/fr/score_4/**/*.parquet
          - data/fineweb2/fr/score_3/**/*.parquet
          - data/fineweb2/fr/score_2/**/*.parquet
          - data/fineweb2/fr/score_1/**/*.parquet
          - data/fineweb2/fr/score_0/**/*.parquet
      - split: frp
        path: data/fineweb2/frp/**/*.parquet
      - split: gcf
        path: data/fineweb2/gcf/**/*.parquet
      - split: gcr
        path: data/fineweb2/gcr/**/*.parquet
      - split: it
        path:
          - data/fineweb2/it/score_4/**/*.parquet
          - data/fineweb2/it/score_3/**/*.parquet
          - data/fineweb2/it/score_2/**/*.parquet
          - data/fineweb2/it/score_1/**/*.parquet
          - data/fineweb2/it/score_0/**/*.parquet
      - split: nl
        path:
          - data/fineweb2/nl/score_4/**/*.parquet
          - data/fineweb2/nl/score_3/**/*.parquet
          - data/fineweb2/nl/score_2/**/*.parquet
          - data/fineweb2/nl/score_1/**/*.parquet
          - data/fineweb2/nl/score_0/**/*.parquet
      - split: oc
        path: data/fineweb2/oc/**/*.parquet
      - split: pcd
        path: data/fineweb2/pcd/**/*.parquet
      - split: pt
        path:
          - data/fineweb2/pt/score_4/**/*.parquet
          - data/fineweb2/pt/score_3/**/*.parquet
          - data/fineweb2/pt/score_2/**/*.parquet
          - data/fineweb2/pt/score_1/**/*.parquet
          - data/fineweb2/pt/score_0/**/*.parquet
      - split: rcf
        path: data/fineweb2/rcf/**/*.parquet
      - split: ty
        path: data/fineweb2/ty/**/*.parquet
      - split: wa
        path: data/fineweb2/wa/**/*.parquet
  - config_name: Fineweb2-3plus
    data_files:
      - split: all
        path:
          - data/fineweb2/fr/score_4/**/*.parquet
          - data/fineweb2/fr/score_3/**/*.parquet
      - split: fr
        path:
          - data/fineweb2/fr/score_4/**/*.parquet
          - data/fineweb2/fr/score_3/**/*.parquet
  - config_name: Fineweb2-HQ
    data_files:
      - split: all
        path: data/fineweb2_hq/**/*.parquet
      - split: ar
        path: data/fineweb2_hq/ar/**/*.parquet
      - split: de
        path: data/fineweb2_hq/de/**/*.parquet
      - split: es
        path: data/fineweb2_hq/es/**/*.parquet
      - split: fr
        path: data/fineweb2_hq/fr/**/*.parquet
      - split: it
        path: data/fineweb2_hq/it/**/*.parquet
      - split: nl
        path: data/fineweb2_hq/nl/**/*.parquet
      - split: pt
        path: data/fineweb2_hq/pt/**/*.parquet
  - config_name: Gallica-monographies
    data_files:
      - split: all
        path: data/gallica/monographies/**/*.parquet
      - split: fr
        path: data/gallica/monographies/fr/**/*.parquet
  - config_name: Gallica-press
    data_files:
      - split: all
        path: data/gallica/press/**/*.parquet
      - split: fr
        path: data/gallica/press/fr/**/*.parquet
  - config_name: Gutenberg
    data_files:
      - split: all
        path: data/gutenberg/**/*.parquet
      - split: de
        path: data/gutenberg/de/**/*.parquet
      - split: en
        path: data/gutenberg/en/**/*.parquet
      - split: es
        path: data/gutenberg/es/**/*.parquet
      - split: fr
        path: data/gutenberg/fr/**/*.parquet
      - split: it
        path: data/gutenberg/it/**/*.parquet
      - split: nl
        path: data/gutenberg/nl/**/*.parquet
      - split: pt
        path: data/gutenberg/pt/**/*.parquet
  - config_name: HAL
    data_files:
      - split: all
        path: data/hal/**/*.parquet
      - split: fr
        path: data/hal/fr/**/*.parquet
  - config_name: HPLT2
    data_files:
      - split: all
        path: data/hplt2/**/*.parquet
      - split: fr
        path: data/hplt2/fr/**/*.parquet
  - config_name: InfiwebMath-3plus
    data_files:
      - split: all
        path:
          - data/infiwebmath/en/score_3/**/*.parquet
          - data/infiwebmath/en/score_4/**/*.parquet
          - data/infiwebmath/en/score_5/**/*.parquet
      - split: en
        path:
          - data/infiwebmath/en/score_3/**/*.parquet
          - data/infiwebmath/en/score_4/**/*.parquet
          - data/infiwebmath/en/score_5/**/*.parquet
  - config_name: InfiwebMath-4plus
    data_files:
      - split: all
        path:
          - data/infiwebmath/en/score_4/**/*.parquet
          - data/infiwebmath/en/score_5/**/*.parquet
      - split: en
        path:
          - data/infiwebmath/en/score_4/**/*.parquet
          - data/infiwebmath/en/score_5/**/*.parquet
  - config_name: Insee
    data_files:
      - split: all
        path: data/insee/**/*.parquet
      - split: fr
        path: data/insee/fr/**/*.parquet
  - config_name: MathPile
    data_files:
      - split: all
        path: data/math_pile/**/*.parquet
      - split: en
        path: data/math_pile/en/**/*.parquet
  - config_name: MegamathWeb
    data_files:
      - split: all
        path: data/megamath-web/**/*.parquet
      - split: en
        path: data/megamath-web/en/**/*.parquet
  - config_name: NemotronPosttraining-chat
    data_files:
      - split: all
        path: data/nemotron_posttraining/chat/**/*.parquet
      - split: en
        path: data/nemotron_posttraining/chat/en/**/*.parquet
  - config_name: NemotronPosttraining-code
    data_files:
      - split: all
        path: data/nemotron_posttraining/code/**/*.parquet
      - split: en
        path: data/nemotron_posttraining/code/en/**/*.parquet
  - config_name: NemotronPosttraining-math
    data_files:
      - split: all
        path: data/nemotron_posttraining/math/**/*.parquet
      - split: en
        path: data/nemotron_posttraining/math/en/**/*.parquet
  - config_name: NemotronPosttraining-multilingual_w_thinking
    data_files:
      - split: all
        path: data/nemotron_posttraining/multilingual/w_thinking/**/*.parquet
      - split: de
        path: data/nemotron_posttraining/multilingual/w_thinking/de/**/*.parquet
      - split: es
        path: data/nemotron_posttraining/multilingual/w_thinking/es/**/*.parquet
      - split: fr
        path: data/nemotron_posttraining/multilingual/w_thinking/fr/**/*.parquet
      - split: it
        path: data/nemotron_posttraining/multilingual/w_thinking/it/**/*.parquet
  - config_name: NemotronPosttraining-multilingual_wo_thinking
    data_files:
      - split: all
        path: data/nemotron_posttraining/multilingual/wo_thinking/**/*.parquet
      - split: de
        path: data/nemotron_posttraining/multilingual/wo_thinking/de/**/*.parquet
      - split: es
        path: data/nemotron_posttraining/multilingual/wo_thinking/es/**/*.parquet
      - split: fr
        path: data/nemotron_posttraining/multilingual/wo_thinking/fr/**/*.parquet
      - split: it
        path: data/nemotron_posttraining/multilingual/wo_thinking/it/**/*.parquet
  - config_name: NemotronPosttraining-stem
    data_files:
      - split: all
        path: data/nemotron_posttraining/stem/**/*.parquet
      - split: en
        path: data/nemotron_posttraining/stem/en/**/*.parquet
  - config_name: OpenCodeReasoning
    data_files:
      - split: all
        path: data/open_code_reasoning/**/*.parquet
      - split: en
        path: data/open_code_reasoning/en/**/*.parquet
  - config_name: OpenThoughts-code
    data_files:
      - split: all
        path: data/open_thoughts/code/**/*.parquet
      - split: en
        path: data/open_thoughts/code/en/**/*.parquet
  - config_name: OpenThoughts-science
    data_files:
      - split: all
        path: data/open_thoughts/science/**/*.parquet
      - split: en
        path: data/open_thoughts/science/en/**/*.parquet
  - config_name: Opendata
    data_files:
      - split: all
        path: data/opendata/**/*.parquet
      - split: fr
        path: data/opendata/fr/**/*.parquet
  - config_name: Paradocs
    data_files:
      - split: all
        path: data/paradocs/**/*.parquet
      - split: en_de
        path: data/paradocs/en-de/**/*.parquet
      - split: en_es
        path: data/paradocs/en-es/**/*.parquet
      - split: en_fr
        path: data/paradocs/en-fr/**/*.parquet
      - split: en_it
        path: data/paradocs/en-it/**/*.parquet
      - split: en_nl
        path: data/paradocs/en-nl/**/*.parquet
      - split: en_pt
        path: data/paradocs/en-pt/**/*.parquet
  - config_name: Parlement-amendements_parlement
    data_files:
      - split: all
        path: data/parlement/amendements_parlement/**/*.parquet
      - split: fr
        path: data/parlement/amendements_parlement/fr/**/*.parquet
  - config_name: Parlement-discours_publics
    data_files:
      - split: all
        path: data/parlement/discours_publics/**/*.parquet
      - split: fr
        path: data/parlement/discours_publics/fr/**/*.parquet
  - config_name: Parlement-interventions_parlement
    data_files:
      - split: all
        path: data/parlement/interventions_parlement/**/*.parquet
      - split: fr
        path: data/parlement/interventions_parlement/fr/**/*.parquet
  - config_name: Parlement-questions_ecrites_parlement
    data_files:
      - split: all
        path: data/parlement/questions_ecrites_parlement/**/*.parquet
      - split: fr
        path: data/parlement/questions_ecrites_parlement/fr/**/*.parquet
  - config_name: PleiasSynth
    data_files:
      - split: all
        path: data/pleias_synth/**/*.parquet
      - split: ar
        path: data/pleias_synth/ar/**/*.parquet
      - split: ca
        path: data/pleias_synth/ca/**/*.parquet
      - split: de
        path: data/pleias_synth/de/**/*.parquet
      - split: en
        path: data/pleias_synth/en/**/*.parquet
      - split: es
        path: data/pleias_synth/es/**/*.parquet
      - split: eu
        path: data/pleias_synth/eu/**/*.parquet
      - split: fr
        path: data/pleias_synth/fr/**/*.parquet
      - split: it
        path: data/pleias_synth/it/**/*.parquet
      - split: nl
        path: data/pleias_synth/nl/**/*.parquet
      - split: oc
        path: data/pleias_synth/oc/**/*.parquet
      - split: pt
        path: data/pleias_synth/pt/**/*.parquet
  - config_name: Scholar
    data_files:
      - split: all
        path: data/scholar/**/*.parquet
      - split: fr
        path: data/scholar/fr/**/*.parquet
  - config_name: StackEdu
    data_files:
      - split: all
        path: data/stack_edu/**/*.parquet
      - split: code
        path: data/stack_edu/**/*.parquet
  - config_name: StarcoderData
    data_files:
      - split: all
        path: data/starcoder_data/**/*.parquet
      - split: code
        path: data/starcoder_data/**/*.parquet
  - config_name: StarcoderOlmomix
    data_files:
      - split: all
        path: data/starcoder_olmomix/**/*.parquet
      - split: code
        path: data/starcoder_olmomix/**/*.parquet
  - config_name: SynthFineweb2
    data_files:
      - split: all
        path: data/synth_fineweb2/**/*.parquet
      - split: fr
        path: data/synth_fineweb2/fr/**/*.parquet
  - config_name: SyntheticWikipediaQA
    data_files:
      - split: all
        path: data/synthetic_wikipedia_qa/**/*.parquet
      - split: fr
        path: data/synthetic_wikipedia_qa/fr/**/*.parquet
  - config_name: Theses
    data_files:
      - split: all
        path: data/theses/**/*.parquet
      - split: fr
        path: data/theses/fr/**/*.parquet
  - config_name: Vikidia
    data_files:
      - split: all
        path: data/vikidia/**/*.parquet
      - split: ar
        path: data/vikidia/ar/**/*.parquet
      - split: ca
        path: data/vikidia/ca/**/*.parquet
      - split: de
        path: data/vikidia/de/**/*.parquet
      - split: en
        path: data/vikidia/en/**/*.parquet
      - split: es
        path: data/vikidia/es/**/*.parquet
      - split: eu
        path: data/vikidia/eu/**/*.parquet
      - split: fr
        path: data/vikidia/fr/**/*.parquet
      - split: it
        path: data/vikidia/it/**/*.parquet
      - split: oc
        path: data/vikidia/oc/**/*.parquet
      - split: pt
        path: data/vikidia/pt/**/*.parquet
  - config_name: Wikimedia-wikibooks
    data_files:
      - split: all
        path: data/wikimedia/wikibooks/**/*.parquet
      - split: ar
        path: data/wikimedia/wikibooks/ar/**/*.parquet
      - split: ca
        path: data/wikimedia/wikibooks/ca/**/*.parquet
      - split: de
        path: data/wikimedia/wikibooks/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikibooks/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikibooks/es/**/*.parquet
      - split: eu
        path: data/wikimedia/wikibooks/eu/**/*.parquet
      - split: fr
        path: data/wikimedia/wikibooks/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikibooks/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikibooks/nl/**/*.parquet
      - split: oc
        path: data/wikimedia/wikibooks/oc/**/*.parquet
      - split: pt
        path: data/wikimedia/wikibooks/pt/**/*.parquet
  - config_name: Wikimedia-wikinews
    data_files:
      - split: all
        path: data/wikimedia/wikinews/**/*.parquet
      - split: ar
        path: data/wikimedia/wikinews/ar/**/*.parquet
      - split: ca
        path: data/wikimedia/wikinews/ca/**/*.parquet
      - split: de
        path: data/wikimedia/wikinews/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikinews/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikinews/es/**/*.parquet
      - split: fr
        path: data/wikimedia/wikinews/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikinews/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikinews/nl/**/*.parquet
      - split: pt
        path: data/wikimedia/wikinews/pt/**/*.parquet
  - config_name: Wikimedia-wikipedia
    data_files:
      - split: all
        path: data/wikimedia/wikipedia/**/*.parquet
      - split: ar
        path: data/wikimedia/wikipedia/ar/**/*.parquet
      - split: br
        path: data/wikimedia/wikipedia/br/**/*.parquet
      - split: ca
        path: data/wikimedia/wikipedia/ca/**/*.parquet
      - split: co
        path: data/wikimedia/wikipedia/co/**/*.parquet
      - split: de
        path: data/wikimedia/wikipedia/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikipedia/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikipedia/es/**/*.parquet
      - split: eu
        path: data/wikimedia/wikipedia/eu/**/*.parquet
      - split: fr
        path: data/wikimedia/wikipedia/fr/**/*.parquet
      - split: frp
        path: data/wikimedia/wikipedia/frp/**/*.parquet
      - split: it
        path: data/wikimedia/wikipedia/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikipedia/nl/**/*.parquet
      - split: oc
        path: data/wikimedia/wikipedia/oc/**/*.parquet
      - split: pcd
        path: data/wikimedia/wikipedia/pcd/**/*.parquet
      - split: pt
        path: data/wikimedia/wikipedia/pt/**/*.parquet
  - config_name: Wikimedia-wikiquote
    data_files:
      - split: all
        path: data/wikimedia/wikiquote/**/*.parquet
      - split: ar
        path: data/wikimedia/wikiquote/ar/**/*.parquet
      - split: br
        path: data/wikimedia/wikiquote/br/**/*.parquet
      - split: ca
        path: data/wikimedia/wikiquote/ca/**/*.parquet
      - split: de
        path: data/wikimedia/wikiquote/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikiquote/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikiquote/es/**/*.parquet
      - split: eu
        path: data/wikimedia/wikiquote/eu/**/*.parquet
      - split: fr
        path: data/wikimedia/wikiquote/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikiquote/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikiquote/nl/**/*.parquet
      - split: pt
        path: data/wikimedia/wikiquote/pt/**/*.parquet
  - config_name: Wikimedia-wikisource
    data_files:
      - split: all
        path: data/wikimedia/wikisource/**/*.parquet
      - split: ar
        path: data/wikimedia/wikisource/ar/**/*.parquet
      - split: br
        path: data/wikimedia/wikisource/br/**/*.parquet
      - split: ca
        path: data/wikimedia/wikisource/ca/**/*.parquet
      - split: de
        path: data/wikimedia/wikisource/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikisource/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikisource/es/**/*.parquet
      - split: eu
        path: data/wikimedia/wikisource/eu/**/*.parquet
      - split: fr
        path: data/wikimedia/wikisource/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikisource/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikisource/nl/**/*.parquet
      - split: pt
        path: data/wikimedia/wikisource/pt/**/*.parquet
  - config_name: Wikimedia-wikiversity
    data_files:
      - split: all
        path: data/wikimedia/wikiversity/**/*.parquet
      - split: ar
        path: data/wikimedia/wikiversity/ar/**/*.parquet
      - split: de
        path: data/wikimedia/wikiversity/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikiversity/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikiversity/es/**/*.parquet
      - split: fr
        path: data/wikimedia/wikiversity/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikiversity/it/**/*.parquet
      - split: pt
        path: data/wikimedia/wikiversity/pt/**/*.parquet
  - config_name: Wikimedia-wikivoyage
    data_files:
      - split: all
        path: data/wikimedia/wikivoyage/**/*.parquet
      - split: de
        path: data/wikimedia/wikivoyage/de/**/*.parquet
      - split: en
        path: data/wikimedia/wikivoyage/en/**/*.parquet
      - split: es
        path: data/wikimedia/wikivoyage/es/**/*.parquet
      - split: fr
        path: data/wikimedia/wikivoyage/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wikivoyage/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wikivoyage/nl/**/*.parquet
      - split: pt
        path: data/wikimedia/wikivoyage/pt/**/*.parquet
  - config_name: Wikimedia-wiktionary
    data_files:
      - split: all
        path: data/wikimedia/wiktionary/**/*.parquet
      - split: ar
        path: data/wikimedia/wiktionary/ar/**/*.parquet
      - split: br
        path: data/wikimedia/wiktionary/br/**/*.parquet
      - split: ca
        path: data/wikimedia/wiktionary/ca/**/*.parquet
      - split: co
        path: data/wikimedia/wiktionary/co/**/*.parquet
      - split: de
        path: data/wikimedia/wiktionary/de/**/*.parquet
      - split: en
        path: data/wikimedia/wiktionary/en/**/*.parquet
      - split: es
        path: data/wikimedia/wiktionary/es/**/*.parquet
      - split: eu
        path: data/wikimedia/wiktionary/eu/**/*.parquet
      - split: fr
        path: data/wikimedia/wiktionary/fr/**/*.parquet
      - split: it
        path: data/wikimedia/wiktionary/it/**/*.parquet
      - split: nl
        path: data/wikimedia/wiktionary/nl/**/*.parquet
      - split: oc
        path: data/wikimedia/wiktionary/oc/**/*.parquet
      - split: pt
        path: data/wikimedia/wiktionary/pt/**/*.parquet
  - config_name: Youtube
    data_files:
      - split: all
        path: data/youtube/**/*.parquet
      - split: fr
        path: data/youtube/fr/**/*.parquet
---
