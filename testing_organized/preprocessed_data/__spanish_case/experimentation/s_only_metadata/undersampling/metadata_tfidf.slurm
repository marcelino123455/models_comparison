#!/bin/bash

# Nombre del job:
#SBATCH --job-name=M_tfidf
#SBATCH --account=investigacion1
#SBATCH --partition=gpu
#SBATCH --nodelist=g001


# Cantidad de CPUs cores a usar:
#SBATCH -c 1

#SBATCH --output=./metadata_us_pesudo/2/tf_idf.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe



cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__spanish_case/experimentation/s_only_metadata/undersampling

source /home/marcelino.maita/.venv/bin/activate

srun python3 metadata_tfidf.py