#!/bin/bash

# Nombre del job:
#SBATCH --job-name=gpt_embb
#SBATCH --partition=data-science

# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=date.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marcelino.maita@utec.edu.pe


cd /home/marcelino.maita/musica_ia_workshop/models_comparison/baseline_clean/complex_models/GenerateEmbbedings/Gpt_embbedings

source /home/marcelino.maita/.venv/bin/activate

papermill \
  generate_gptEmbb.ipynb \
  ejecutado_rn_$(date +%Y%m%d_%H%M%S).ipynb