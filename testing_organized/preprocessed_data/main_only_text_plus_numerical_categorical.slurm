#!/bin/bash

# Nombre del job:
#SBATCH --job-name=t+n+c_F_DL
#SBATCH --account=investigacion1
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001


# Cantidad de CPUs cores a usar:
#SBATCH -c 1

#SBATCH --output=./outputs_text_plus_numerical_categorical/2/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe



cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data
source /home/marcelino.maita/.venv/bin/activate

srun python3 main_only_text_plus_numerical_categorical.py