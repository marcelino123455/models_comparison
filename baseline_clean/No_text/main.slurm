#!/bin/bash

# Nombre del job:
#SBATCH --job-name=No_text
#SBATCH --partition=gpu

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_b/undersample_True_scaled_True_steaming_True_removestw_True_numeric_True_5000_tfidf_B/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean/No_text
source /home/marcelino.maita/.venv/bin/activate

srun python3 main.py