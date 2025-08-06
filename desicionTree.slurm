#!/bin/bash

# Nombre del job:
#SBATCH --job-name=DTjob
#SBATCH --partition=gpu

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=DT.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison
source /home/marcelino.maita/.venv/bin/activate

papermill \
  desicionTree.ipynb \
  ejecutado_DT_$(date +%Y%m%d_%H%M%S).ipynb