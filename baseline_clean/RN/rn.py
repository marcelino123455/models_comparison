import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os


DATA_PATH ="../data"
path_lb_embb = os.path.join(DATA_PATH, "lb_npy.npy")
path_dataset = os.path.join(DATA_PATH, "spotify_dataset_sin_duplicados_4.csv")

# Para hacer pruebitas y luego mandarlo como sbatch
TESTING = True

if TESTING:
    NROWS = 1000
else:
    NROWS = 1000
    

df = pd.read_csv(path_dataset, nrows = NROWS)
df['Explicit_binary'] = df['Explicit'].map({'Yes': 1, 'No': 0})




X = np.load(path_lb_embb)

Y = df['Explicit_binary']
