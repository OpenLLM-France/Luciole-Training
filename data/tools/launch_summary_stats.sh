
python summary_stats.py \
    --reader_type jsonl \
    --data_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_datasets_ablation/fineweb_edu/output \
    --output_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_datasets_ablation/fineweb_edu \
    --language en \
    --sample_rate 0.2

python summary_stats.py \
    --reader_type jsonl \
    --data_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_datasets_ablation/starcoder/1_high_stars_count \
    --output_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/raw_datasets_ablation/starcoder \
    --language en \
    --sample_rate 0.2

