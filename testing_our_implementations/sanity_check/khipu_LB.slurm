#!/bin/bash

# Nombre del job:
#SBATCH --job-name=LB_embb
#SBATCH --partition=gpu

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=checking_LB_embb.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_our_implementations/sanity_check
source /home/marcelino.maita/.venv/bin/activate

papermill \
  checking_LB_embb.ipynb \
  checking_LB_embb$(date +%Y%m%d_%H%M%S).ipynb