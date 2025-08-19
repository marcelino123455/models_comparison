import string
from gensim.models import Word2Vec
import pandas as pd

from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import random
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# CONFIGURACIONES 

TESTING = True
COLUMN = 'lyrics'

if TESTING: 
    NROWS  = 100
else:
    NROWS = None  # Cargar todo el dataset



# Cargar el archivo CSV
ruta_csv = "../../data/ml-workshop.songs_lang.csv"

try:
    df = pd.read_csv(ruta_csv,nrows=NROWS)

    # Asegurar que existe la columna 'text'
    if COLUMN not in df.columns:
        raise ValueError(f"La columna '{COLUMN}' no se encuentra en el archivo CSV.")

    # Extraer las letras de las canciones a una lista
    todas_las_canciones = []

    for texto in tqdm(df[COLUMN].fillna("")):
        todas_las_canciones.append(texto)

    print("Letras extraídas con éxito.")
except Exception as e:
    print(f"Error al cargar o procesar el archivo CSV: {e}")

language = 'spanish'
stop_words = set(stopwords.words(language))

def preprocesar_texto(texto):
    texto = texto.lower()
    # texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    tokens = texto.split()  # En lugar de word_tokenize
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

corpus = [preprocesar_texto(letra) for letra in todas_las_canciones]


## Comenzando a sacar embbedings

# Reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from gensim.models.word2vec import Word2Vec
Word2Vec.seed = SEED

from gensim.models import Word2Vec

modelo_w2v = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2, workers=4, seed=SEED)


def vector_promedio(tokens, modelo):
    vectores = [modelo.wv[word] for word in tokens if word in modelo.wv]
    if len(vectores) == 0:
        return np.zeros(modelo.vector_size)
    return np.mean(vectores, axis=0)

vectores_canciones = np.array([vector_promedio(tokens, modelo_w2v) for tokens in corpus])


# Guardar en formato word2vec compatible con Gensim
with open("canciones_embeddings_spanish.txt", "w", encoding="utf-8") as f:
    f.write(f"{len(vectores_canciones)} {modelo_w2v.vector_size}\n")
    for idx, vec in enumerate(vectores_canciones):
        vector_str = ' '.join(map(str, vec))
        f.write(f"cancion_{idx} {vector_str}\n")