#!/bin/bash

# Nombre del job:
#SBATCH --account=investigacion1
#SBATCH --job-name=hold_out_complete
#SBATCH --partition=gpu
#SBATCH --nodelist=g002

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./outputs_hold_out/0/consola.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean/No_text
source /home/marcelino.maita/.venv/bin/activate

srun python3 main_complete_hold_out.py