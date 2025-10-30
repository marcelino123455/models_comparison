#!/bin/bash

# Nombre del job:
#SBATCH --job-name=M_GPT
#SBATCH --account=investigacion1
#SBATCH --partition=gpu
#SBATCH --nodelist=g001


# Cantidad de CPUs cores a usar:
#SBATCH -c 1

#SBATCH --output=./outputs_numerical_categorical_metadata/2/gpt.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe



cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__pseudo_labeling/experimentation/only_metadata

source /home/marcelino.maita/.venv/bin/activate

srun python3 metadata_GPT.py