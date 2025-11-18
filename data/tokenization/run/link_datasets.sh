#!/bin/bash

# Generic linking function
link_files() {
    local src_dir="$1"
    local dst_dir="$2"
    shift 2
    local patterns=("$@")

    mkdir -p "$dst_dir/stats"

    for ext in bin idx json; do
        if [ ${#patterns[@]} -eq 0 ]; then
            # Default patterns: match everything
            if [ "$ext" = "json" ]; then
                src_path="$src_dir/stats/*.${ext}"
                dst_path="$dst_dir/stats"
            else
                src_path="$src_dir/*.${ext}"
                dst_path="$dst_dir"
            fi

            for file in $src_path; do
                [ -e "$file" ] || { echo -e "\n>>> Error: $file not found\n"; continue; }
                echo -e "\nLinking $file to $dst_path"
                if ln -sf "$file" "$dst_path/$(basename "$file")"; then
                    echo -e "    Link successful"
                else
                    echo -e ">>> Error linking $file to $dst_path"
                fi
            done
        else
            for pattern in "${patterns[@]}"; do
                if [ "$ext" = "json" ]; then
                    src_path="$src_dir/stats/${pattern}.${ext}"
                    dst_path="$dst_dir/stats"
                else
                    src_path="$src_dir/${pattern}.${ext}"
                    dst_path="$dst_dir"
                fi

                for file in $src_path; do
                    [ -e "$file" ] || { echo -e "\n>>> Error: $file not found\n"; continue; }
                    echo -e "\nLinking $file to $dst_path"
                    if ln -sf "$file" "$dst_path/$(basename "$file")"; then
                        echo -e "    Link successful"
                    else
                        echo -e ">>> Error linking $file to $dst_path\n"
                    fi
                done
            done
        fi
    done
}

# --- Main tokens ---
main_patterns=(
    common-corpus_arabic-pd_ar*
    common-corpus_bnl-newspapers-1841-1879_fr*
    common-corpus_catalan-pd_ca*
    common-corpus_dutch-pd_nl*
    common-corpus_english-pd_en*
    common-corpus_eurlex_multi*
    common-corpus_gatt-library_multi*
    common-corpus_german-pd_de*
    common-corpus_german-science-pile_de*
    common-corpus_italian-pd_it*
    common-corpus_marianne-europe_multi*
    common-corpus_oecd_multi*
    common-corpus_open-science-pile_de*
    common-corpus_portuguese-pd_pt*
    common-corpus_spanish-pd-books_es*
    common-corpus_spanish-science-pile_es*
    common-corpus_tedeutenders_multi*
    common-corpus_wto_multi*
    common-pile_arxiv-abstracts_en*
    common-pile_arxiv-papers_en*
    common-pile_biodiversity-heritage-library_en*
    common-pile_caselaw-access-project_en*
    common-pile_data-provenance-initiative_en*
    common-pile_doab_en*
    common-pile_foodista_en*
    common-pile_github-archive_en*
    common-pile_library-of-congress_en*
    common-pile_libretexts_en*
    common-pile_news_en*
    common-pile_oercommons_en*
    common-pile_peS2o_en*
    common-pile_pre-1929-books_en*
    common-pile_pressbooks_en*
    common-pile_public-domain-review_en*
    common-pile_pubmed_en*
    common-pile_python-enhancement-proposals_en*
    common-pile_regulations_en*
    common-pile_stackexchange_en*
    common-pile_ubuntu-irc_en*
    common-pile_youtube_en*
    claire_open*
    croissant-aligned*
    math-pile*
    insee_fr*
    hal-cea_full-filtered_fr*
    gallica*
    gutenberg*
    opendata_fr*
    opene-edition_fr*
    parlement_fr*
    theses_fr*
    youtube_fr*
    wikimedia_nl*
    europarl_aligned*
    starcoder_olmomix*
    stack-edu_code*
    infiwebmath-filtered*
    finemath-filtered*
    open-code-reasoning_en*
    stack-math-qa-1600k_en*
    nemotron-posttraining*
    wikipedia*
    synthetic-fineweb_extract-knowledge-easy*
    synthetic-fineweb_extract-knowledge-medium*
    synthetic-fineweb_extract-knowledge-hard*
    fineweb2_hq*
    fineweb2_edu3+*
    fineweb2_edu4*
    open-math-instruct*
    aya*
    kurakurai_scholar*
    paradocs_geom*
    pleias*
)
link_files \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_training_v2" \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_lucie2" \
    "${main_patterns[@]}"

# --- Web non-english tokens ---
main_patterns=(
    culturax_ar*
    culturax_it*
    culturax_nl*
    culturax_pt*
    fineweb2_ar*
    fineweb2_ca*
    fineweb2_it*
    fineweb2_nl*
    fineweb2_pt*
)
link_files \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_training_web_wo_prefix" \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_lucie2" \
    "${main_patterns[@]}"

# --- Grouped tokens ---
link_files \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_training_grouped" \
    "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_lucie2"

# run stats
module purge
module load arch/a100 nemo/2.2.1

python merge_stats.py "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenized_data/tokens_lucie2" --add_output_path ./chronicles