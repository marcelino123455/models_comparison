#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=ablation_one_feature
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_ablation_one_feature/0/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/testing_organized/preprocessed_data/__ablation_study_tfidf
source /home/marcelino.maita/.venv/bin/activate

srun python3 ablation_study_only_one_feature.py