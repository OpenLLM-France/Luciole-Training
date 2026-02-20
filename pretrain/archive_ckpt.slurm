#!/bin/sh
#SBATCH --job-name=Archivage  # nom du travail
#SBATCH --partition=archive   # on utilise la partition "archive" ou "prepost" qui a accès au STORE
#SBATCH --ntasks=1            # on a un travail séquentiel
#SBATCH --hint=nomultithread  # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00       # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out    # fichier de sortie et d'erreur (%x = nom du travail, %j = numéro du travail)
#SBATCH --account=qgz@cpu

create_archive() {
    local dir_to_archive="$1"
    local output_path="$2"

    # Check arguments
    if [[ -z "$dir_to_archive" || -z "$output_path" ]]; then
        echo "Error: missing arguments"
        return 1
    fi

    # Check if the output path ends with .tar
    if [[ "$output_path" != *.tar ]]; then
        echo "Error: output_path must end with .tar"
        return 1
    fi

    # Check if the directory to archive exists
    if [[ ! -d "$dir_to_archive" ]]; then
        echo "Error: directory '$dir_to_archive' does not exist"
        return 1
    fi

    # Skip if archive already exists
    if [[ -f "$output_path" ]]; then
        echo "Archive already exists at $output_path — skipping."
        return 0
    fi

    # Optional: skip if unfinished archive exists
    if [[ -f "$output_path.unfinished" ]]; then
        echo "Unfinished archive exists at $output_path.unfinished — skipping."
        return 1
    fi

    # Create parent directory if needed
    mkdir -p "$(dirname "$output_path")" || return 1

    # Create archive with temporary name
    local tmp_archive="${output_path}.unfinished"
    tar -cvf "$tmp_archive" "$dir_to_archive"
    local tar_status=$?

    # Check if tar succeeded
    if [[ $tar_status -ne 0 ]]; then
        echo "Error: failed to create archive"
        rm -f "$tmp_archive"
        return 1
    fi

    # Move to final location
    mv "$tmp_archive" "$output_path" || return 1
    echo "Archive created at $output_path"
    return 0
}

dir_to_experiment="$1"
output_path="$2"

expename=$(basename "$dir_to_experiment")

# Find all checkpoint directories
for ckpt_folder in "$dir_to_experiment/$expename/checkpoints/"/*/; do
    ckpt_name=$(basename "$ckpt_folder")
    input="$dir_to_experiment/$expename/checkpoints/$ckpt_name"
    output="$output_path/$expename/checkpoints/$ckpt_name.tar"

    echo "Submitting job for $input to be archived at $output"
    create_archive "$input" "$output"
    echo ""
done