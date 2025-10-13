#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=BertEVA_01
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_d/6/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean/No_text
source /home/marcelino.maita/.venv/bin/activate

srun python3 main_complete.py