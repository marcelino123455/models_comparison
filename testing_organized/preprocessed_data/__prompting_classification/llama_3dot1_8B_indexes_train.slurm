#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=aaa_yannojefe
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=salida.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__prompting_classification
source /home/marcelino.maita/.venv/bin/activate

papermill \
  llama_3dot1_8B_indexes_train.ipynb \
  llama_3dot1_8B_indexes_train_$(date +%Y%m%d_%H%M%S).ipynb