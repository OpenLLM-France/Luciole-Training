#!/bin/sh
#SBATCH --job-name=Archivage  # nom du travail
#SBATCH --partition=archive   # on utilise la partition "archive" ou "prepost" qui a accès au STORE
#SBATCH --ntasks=1            # on a un travail séquentiel
#SBATCH --hint=nomultithread  # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00       # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out    # fichier de sortie et d'erreur (%x = nom du travail, %j = numéro du travail)
#SBATCH --account=qgz@cpu
 
dir_to_archive="$1"
output_path="$2"

# Check if the output path ends with .tar
if [[ "$output_path" != *.tar ]]; then
    echo "Error: output_path must end with .tar"
    exit 1
fi

# Check if the directory to archive exists
if [[ ! -d "$dir_to_archive" ]]; then
    echo "Error: directory '$dir_to_archive' does not exist"
    exit 1
fi

# Create the parent directory for the archive if it doesn't exist
mkdir -p "$(dirname "$output_path")"

# Create archive with temporary name
tmp_archive="$output_path.unfinished"
tar -cvf "$tmp_archive" "$dir_to_archive"

# Check if tar succeeded
if [[ $? -ne 0 ]]; then
    echo "Error: failed to create archive"
    rm -f "$tmp_archive"
    exit 1
fi

# Move to final location
mv "$tmp_archive" "$output_path"
echo "Archive created at $output_path"