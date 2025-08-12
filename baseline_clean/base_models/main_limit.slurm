#!/bin/bash

# Nombre del job:
#SBATCH --job-name=models
#SBATCH --partition=gpu

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_limi/undersample_True_scaled_True_steaming_True_removestw_True_numeric_True_5000_tfidf_A/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean
source /home/marcelino.maita/.venv/bin/activate

srun python3 main_limit.py