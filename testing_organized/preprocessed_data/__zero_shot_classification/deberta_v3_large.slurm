#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=zs_vs_tf_
#SBATCH --partition=data-science
#SBATCH --nodelist=ds001

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=salida.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__zero_shot_classification
source /home/marcelino.maita/.venv/bin/activate

papermill \
  deberta_v3_large.ipynb \
  deberta_v3_large_indexes_$(date +%Y%m%d_%H%M%S).ipynb