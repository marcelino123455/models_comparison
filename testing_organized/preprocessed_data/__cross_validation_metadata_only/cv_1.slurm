#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=a_44
#SBATCH --partition=gpu
#SBATCH --nodelist=g002

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_cv/2/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__cross_validation_metadata_only
source /home/marcelino.maita/.venv/bin/activate

srun python3 cv_1.py