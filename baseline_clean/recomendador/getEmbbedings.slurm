#!/bin/bash

# Nombre del job:
#SBATCH --job-name=nvsnn
#SBATCH --partition=gpu

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean/recomendador



module purge
module load miniconda/3.0
conda activate my-env

export NLTK_DATA=/home/marcelino.maita/nltk_data

srun python3 getEmbbedings.py