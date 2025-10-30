#!/bin/bash

# Nombre del job:
#SBATCH --job-name=pytia_N
#SBATCH --account=investigacion1
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001


# Cantidad de CPUs cores a usar:
#SBATCH -c 1

#SBATCH --output=./outputs_text_pytia/27/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe



cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data
source /home/marcelino.maita/.venv/bin/activate

srun python3 main_text_pytia.py