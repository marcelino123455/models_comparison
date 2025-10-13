import os
import re
import time
import json
import joblib
import psutil
import umap
import papermill
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from torch.utils.data import DataLoader, TensorDataset

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_STATE = 42


# Cleaning TEXT

class TextPreprocessor:
    """
    Text preprocessing class for lyrics data
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize the preprocessor

        Args:
            language: Language for stopwords (default: english)
        """
        self.language = language
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        import re

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()



        return text
    


    def tokenize_and_process(self, text, remove_stopwords = True,
                           apply_stemming = True):
        """
        Tokenize and process text

        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming

        Returns:
            List of processed tokens
        """
        if not text:
            return []

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Apply stemming if requested
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def preprocess(self, text: str, remove_stopwords: bool = True,
                   apply_stemming: bool = True) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            apply_stemming: Whether to apply stemming

        Returns:
            Preprocessed text as string
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_process(cleaned_text, remove_stopwords, apply_stemming)
        return ' '.join(tokens)





def load_LB_embbedings(csv_path, npy_path, sample_size=None):
    print("Loading Lb vectors from: ",csv_path)
    
    X_vec = np.load(npy_path)
    df = pd.read_csv(csv_path)
    df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)

    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)
        X = X_vec[sampled_idx]
    
    else:
        X = X_vec
    y = df['Explicit_binary']
    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def elimnate_y_index(y, path= "faltantes_according_token.json"):
    elminar_ = get_array(path)
    print("Se eliminarán los siguientes indices", elminar_)

    if elminar_[-1] > y.shape[0]:
        return y

    mask = np.ones(len(y), dtype=bool)
    mask[elminar_] = False
    y_filtrado = y[mask]
    
    print(f"Shape original y: {y.shape}")
    print(f"Shape filtrado  y: {y_filtrado.shape}")
    return  y_filtrado


def load_gpt_embbedings(csv_path, npy_path, sample_size=None):
    print("Loading Lb vectors from: ",csv_path)
    
    X_vec = np.load(npy_path)
    df = pd.read_csv(csv_path)
    df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)

    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)
        X = X_vec[sampled_idx]
    
    else:
        X = X_vec
    y = df['Explicit_binary']
    y = elimnate_y_index(y)
    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y



def load_TFIDF_embbedings(csv_path, sample_size=None, columns_tf_idfizable = ['text'], max_features = 5000, steaming= False, remove_stopwords = True):
    
    # Read dataset in csv
    df = pd.read_csv(csv_path)

    # print(df.columns)

    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)

 
    # Chose the columns to use TF-IDF
    df['combined_text'] = df[columns_tf_idfizable].fillna('').agg(' '.join, axis=1)
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    if steaming: 
        print("Preprocessing text...")
        # tqdm.pandas(desc="Text preprocessing")
        # df['processed_text'] = df['combined_text'].progress_apply(
        #     lambda x: preprocessor.preprocess(x, remove_stopwords=True, apply_stemming=steaming)
        # )
        df['processed_text'] = df['combined_text'].apply(
            lambda x: preprocessor.preprocess(x, remove_stopwords=True, apply_stemming=steaming)
        )
    else:
        df['processed_text'] = df['combined_text']

    df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)
    y = df['Explicit_binary']


    # Aplying TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.92,
        ngram_range=(1, 2)
    )


    X_tfidf = vectorizer.fit_transform(df['processed_text'])

    X = X_tfidf.toarray()
    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def undersample(X, y, random_state= RANDOM_STATE):
    print("="*50)
    print("Data antes del undersampling ...")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print("Apliying UNDERSAMPLE")
    df = pd.DataFrame(X).reset_index(drop=True)
    df['label'] = pd.Series(y).reset_index(drop=True)
    counts = df['label'].value_counts()
    n_min = counts.min()
    print(n_min)
    df_bal = df.groupby('label', group_keys=False) \
               .apply(lambda grp: grp.sample(n=n_min, random_state=RANDOM_STATE))
    
    print(f"Label distribution: {df_bal['label'].value_counts().to_dict()}")
    
    X = df_bal.drop(columns=['label']).values
    y = df_bal['label'].values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y
####### MLP ###########
class MLP(nn.Module):
    def __init__(self, capas):
        super(MLP, self).__init__()
        layers = []
        # Capas 
        for i in range(len(capas)-2):
            layers.append(nn.Linear(capas[i], capas[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(capas[-2], capas[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        
def data_to_tensor(
    data_path="../../../../data",
    emb_file="embbedings_khipu/tfidf_numeric_B.npy",
    dataset_file="spotify_dataset_sin_duplicados_4.csv",
    n_rows=False,
    scaled=True,
    batch_size=64,
    test_size=0.2,
    random_state=42
):
    # Paths
    path_lb_embb = os.path.join(data_path, emb_file)
    path_dataset = os.path.join(data_path, dataset_file)

    # Load embeddings
    embeddings = np.load(path_lb_embb, mmap_mode="r")

    if n_rows:
        nrows = n_rows
        embeddings = embeddings[:nrows]
        epochs = 30
    else:
        nrows = None
        epochs = 30

    # Scale embeddings
    if scaled:
        scaler = MinMaxScaler(feature_range=(0, 1))
        embeddings = scaler.fit_transform(embeddings)
    else:
        scaler = None

    # Load dataset
    df = pd.read_csv(path_dataset, nrows=nrows)
    df['Explicit_binary'] = df['Explicit'].map({'Yes': 1, 'No': 0})

    X = embeddings
    y = df['Explicit_binary']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convertir a tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

    # Crear datasets
    trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
    testDataset  = TensorDataset(X_test_tensor,  y_test_tensor)

    # Crear dataloaders
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader  = DataLoader(testDataset,  batch_size=batch_size, shuffle=False)

    return (
        trainLoader,
        testLoader,
        trainDataset,
        testDataset,
        X_train,
        X_test,
        y_train,
        y_test,
        epochs,
        scaler
    )

def train_deep_rn(net, trainLoader, testLoader, criterion, optimizer, device, n_epoch=20, target_f1=0.95, print_every=126):
    train_losses = []
    test_losses = []
    best_f1_score = 0.0
    best_pred = None
    best_ephoc = None
    best_labels = None
    AUC_according_best_f1 = 0.0

    for epoch in range(n_epoch):
        # Training
        net.train()
        total_train_loss = 0
        for embbedings, labels in trainLoader:
            embbedings, labels = embbedings.to(device), labels.to(device).float()
            outputs = net(embbedings).squeeze(1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(trainLoader)
        train_losses.append(avg_train_loss)

        # Evaluation
        net.eval()
        total_test_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for embbedings, labels in testLoader:
                embbedings, labels = embbedings.to(device), labels.to(device).float()
                outputs = net(embbedings).squeeze(1)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_test_loss = total_test_loss / len(testLoader)
        test_losses.append(avg_test_loss)

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs) 
        if f1 > best_f1_score:
            best_f1_score = f1
            best_pred = all_preds
            best_ephoc = epoch
            best_labels = all_labels.copy()
            AUC_according_best_f1 = auc
            epocas= epoch
        if f1 >= target_f1:
            print(f"Target F1-score {target_f1} alcanzado en la época {epoch}. Deteniendo entrenamiento.")
            break
        if epoch % print_every == 0:
            print(f"Epoch [{epoch}/{n_epoch}] "
                  f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
                  f"F1: {f1:.4f}, AUC: {auc:.4f}")

    print("Mejores resultados en la época: ", best_ephoc)
    print("f1-score", best_f1_score)
    print("AUC según el mejor F1-score", AUC_according_best_f1)
    return train_losses,test_losses, best_f1_score, best_pred, best_ephoc, best_labels, AUC_according_best_f1

def evaluar_modelo(y_true, y_pred, params=None, labels=('Not Explicit', 'Explicit'),
                save_dir="resultados"):
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = "Confusion Matrix"
    if params is not None:
        title += f" (Epoch = {params})"
    plt.title(title)

    # Guardar o mostrar
    # if save:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"confusion_matrix_param_{params if params is not None else 'final'}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Matriz de confusión guardada en: {filepath}")
    plt.close()
    # else:
    # plt.show()

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }
# evaluar_modelo(best_labels, best_pred, 20, labels=('Not Explicit', 'Explicit'), save_dir="resulta3dos")
 


####### END MLP ###########




# ['tfidf', 'lyrics_bert']
def train_models(X_train, X_test, y_train, y_test, dir_ = "output", embedding_type="tfidf", nets = None):
    models = {
        "Logistic Regression": LogisticRegression(
            penalty='l2',               # Regularización L2
            C=1,                   # Inverso de lambda_reg → C = 1 / λ
            max_iter=1000,              # Iteraciones máximas
            solver='lbfgs',             # Recomendado para L2 + datasets grandes
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            C=1.0,                  
            kernel='rbf',           
            max_iter=1000,          
            random_state=RANDOM_STATE 
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion='entropy',         
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE

        ), 
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    }

    if embedding_type == "tfidf":
        models["Naive Bayes"] = MultinomialNB(alpha=1.0)
    else:
        models["Naive Bayes"] = GaussianNB()

    resumen_metricas = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{'='*30}")
        print(f"Model: {name}")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, )
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        resumen_metricas[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Explicit', 'Explicit'],
            yticklabels=['Not Explicit', 'Explicit']
        )
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        # Params
        print(f"Hiperparámetros: {model.get_params()}")

        # Save
        os.makedirs(dir_, exist_ok=True)
        filename = os.path.join(dir_, f"conf_matrix_{name.replace(' ', '_').lower()}.png")
        plt.savefig(filename)
        print(f"Confusion matrix saved as: {filename}")
        plt.close()

        # Save model
        model_filename = os.path.join(dir_, f"{name.replace(' ', '_').lower()}_model.pkl")
        joblib.dump(model, model_filename)
        print(f"Modelo guardado como: {model_filename}")


    print("\n\nResumen de métricas:")
    for modelo, metricas in sorted(resumen_metricas.items(), key=lambda x: x[1]["f1_score"], reverse=True):
        print(f"{modelo}: {metricas}")

    return resumen_metricas


def train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_ = "output", embedding_type="tfidf"):
    param_grids = {
        "Logistic Regression": {
            "penalty": ["l2"],
            "C": [0.1, 1],
            "solver": ["lbfgs"],
            "max_iter": [1000]
        },
        "SVM": {
            "C": [0.1, 1],
            "kernel": [ "rbf"],
            "max_iter": [1000]
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [4,5,6,7, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    }

    if embedding_type == "tfidf":
        param_grids["Naive Bayes"] = {"alpha": [0.5, 1.0, 2.0]}
        base_models = {
            "Naive Bayes": MultinomialNB()
        }
    else:
        param_grids["Naive Bayes"] = {"var_smoothing": [1e-9, 1e-8]}
        base_models = {
            "Naive Bayes": GaussianNB()
        }
    base_models.update({
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE)
    })
    resumen_metricas = {}
    for name, model in base_models.items():
        print(f"\n{'='*30}\nOptimizando {name}...")
        if param_grids[name]:
            grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring="f1")
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Mejores parámetros: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)
            best_model = model
            print("Sin parámetros a optimizar para este modelo.")
        
        # Predicción
        y_pred = best_model.predict(X_test)
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        resumen_metricas[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "best_params": best_model.get_params()
        }

        # Reporte y matriz de confusión
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Explicit', 'Explicit'],
                    yticklabels=['Not Explicit', 'Explicit'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        os.makedirs(dir_, exist_ok=True)
        plt.savefig(os.path.join(dir_, f"conf_matrix_{name.replace(' ', '_').lower()}.png"))
        plt.close()

        # Guardar modelo
        joblib.dump(best_model, os.path.join(dir_, f"{name.replace(' ', '_').lower()}_model.pkl"))
    # Resumen final
    print("\n\nResumen final de métricas:")
    for modelo, metricas in sorted(resumen_metricas.items(), key=lambda x: x[1]["f1_score"], reverse=True):
        print(f"{modelo}: {metricas}")

def get_array(path):
    with open(path, "r") as f:
        array_ = json.load(f)  
    return array_

def elimnate_index(X, y, path= "faltantes_according_token.json"):
    elminar_ = get_array(path)
    print("Se eliminarán los siguientes indices", elminar_)

    if elminar_[-1] > X.shape[0]:
        return X, y

    mask = np.ones(len(X), dtype=bool)
    mask[elminar_] = False
    X_filtrado = X[mask]
    y_filtrado = y[mask]
    
    print(f"Shape original X: {X.shape}, y: {y.shape}")
    print(f"Shape filtrado  X: {X_filtrado.shape}, y: {y_filtrado.shape}")
    return X_filtrado, y_filtrado



def main():
    ### CONFIGURATIONS ###
    TESTING = False
    # Data 
    UNDERSAMPLING = True
    USE_SMOTE = False

    SCALED = True
    STEAMING = True
    REMOVESTW = True
    NUMERICCOlS = True
    MAX_FEATURES = 5000

    MLP_ = True

    _EPHOCS = 30





    path_DATA = "../../data"
    if USE_SMOTE and UNDERSAMPLING:
        print("FAIL: You are using smote and undersampling please check")
        print(f"\n")
    
    resultados_globales = {}
    # CONFIGURATIONS For text Columns
    A = ['text', 'song', 'Artist(s)', 'Album', 'Similar Artist 1', 'Genre']
    B = ['Artist(s)', 'song', 'emotion', 'Genre', 'Album', 'Similar Artist 1', 'Similar Song 1', 'Similar Artist 2', 'Similar Song 2', 'Similar Artist 3', 'Similar Song 3', 'song_normalized', 'artist_normalized']
    C = ['text', 'Artist(s)', 'song', 'emotion', 'Genre', 'Album', 'Similar Artist 1', 'Similar Song 1', 'Similar Artist 2', 'Similar Song 2', 'Similar Artist 3', 'Similar Song 3', 'song_normalized', 'artist_normalized']
    
    D = ['emotion', 'Time signature', 'Artist(s)', 'song', 'Genre', 'Album', 'Release Date', 'Key', 'Similar Artist 1', 'Similar Song 1', 'Similar Artist 2', 'Similar Song 2', 'Similar Artist 3', 'Similar Song 3', 'song_normalized', 'artist_normalized']
    T = ['text']

    COL_TF_IDF = D
    print("For TF-IDF embbedings you are selecteing this columns:" )
    print("-->", COL_TF_IDF)

    
    # CONFIGURATIONS for numeric Columns
    N_A =['Danceability', 'Loudness (db)']
    N_B = ['Tempo', 'Popularity', 'Energy', 'Danceability', 'Positiveness', 'Speechiness', 'Liveness', 'Acousticness', 'Instrumentalness', 'Good for Party', 'Good for Work/Study', 'Good for Relaxation/Meditation', 'Good for Exercise', 'Good for Running', 'Good for Yoga/Stretching', 'Good for Driving', 'Good for Social Gatherings', 'Good for Morning Routine']
    
    N_C = ['Tempo','Length',  'Loudness (db)', 'Popularity', 'Energy', 'Danceability', 'Positiveness', 'Speechiness', 'Liveness', 'Acousticness', 'Instrumentalness', 'Good for Party', 'Good for Work/Study', 'Good for Relaxation/Meditation', 'Good for Exercise', 'Good for Running', 'Good for Yoga/Stretching', 'Good for Driving', 'Good for Social Gatherings', 'Good for Morning Routine']
    N_cols = N_C
    if NUMERICCOlS:
        print("For both embbedings your are adding this columns: ")
        print("-->", N_cols)
    else:
        print("Your are not adding cols numercis")


    if TESTING:
        _SAMPLE_SIZE = 1000
        print(f"\nYou are executing with [EXAMPLE] of {_SAMPLE_SIZE} songs")
        # _EMBEDDINGS = _EMBEDDINGS[:1000]
    else:
        print("You are executing with [ALL] dataset")
        _SAMPLE_SIZE = None
        
    if COL_TF_IDF == B or COL_TF_IDF == D:
        # npy_path = "../../data/embbedings_khipu/lb_khipu_B.npy" 

        # npy_path = f"{path_DATA}/embbedings_khipu/LB_fuss/lb_khipu_B.npy" 
        npy_path = f"{path_DATA}/new_embbedings_khipu/LB_fuss/lb_khipu_D.npy" 


        cols_type = "D"
    elif COL_TF_IDF == A:
        cols_type = "A"
        # npy_path = "../../data/embbedings_khipu/lb_khipu_A.npy" 
        npy_path = f"{path_DATA}/embbedings_khipu/LB_fuss/lb_khipu_A.npy" 
    elif COL_TF_IDF == C:
        cols_type = "C"
        # npy_path = "../../data/embbedings_khipu/lb_khipu_A.npy" 
        npy_path = f"{path_DATA}/embbedings_khipu/LB_fuss/lb_khipu_C.npy" 
    elif COL_TF_IDF == T:
        cols_type = "T"
        # npy_path = "../../data/embbedings_khipu/lb_khipu_A.npy" 
        npy_path = f"{path_DATA}//lb_npy.npy" 

    print("--> PaTH: ",npy_path )
        
    csv_path = f"{path_DATA}/spotify_dataset_sin_duplicados_4.csv"



    # A) Embeddings types
    embb_types = ['tfidf', 'lyrics_bert', 'gpt']
    embb_types = ['tfidf', 'lyrics_bert'] # GPT no, since with this only we have the text feature extraction
    # embb_types = ['tfidf']

    # Only with tf_idf we have the best f1-score


    for embedding_type in embb_types:

        if embedding_type == 'lyrics_bert':
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")
            X, y = load_LB_embbedings(csv_path, npy_path, sample_size=_SAMPLE_SIZE)
            X, y = elimnate_index(X, y)

        elif embedding_type == 'gpt':
            
            npy_path_gpt = f"{path_DATA}/gpt_embd/gpt_fussioned/embeddings_fused.npy"
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")
            X, y = load_gpt_embbedings(csv_path, npy_path_gpt, sample_size=_SAMPLE_SIZE)

        else: 
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")
            X, y = load_TFIDF_embbedings(csv_path, sample_size=_SAMPLE_SIZE, columns_tf_idfizable = COL_TF_IDF,  max_features = MAX_FEATURES, steaming= STEAMING, remove_stopwords = REMOVESTW)   
            X, y = elimnate_index(X, y)

        if NUMERICCOlS: # !Numeric cols only works ok with all dataset since yes skdjsk
            df = pd.read_csv(csv_path, nrows=_SAMPLE_SIZE)
            # def elimnate_index(X, y, path= "faltantes_according_token.json"):
            indices_to_remove = get_array("faltantes_according_token.json")
            df = df.drop(indices_to_remove).reset_index(drop=True)
            df['Loudness (db)'] = df['Loudness (db)'].astype(str).str.replace('db', '', regex=False)
            df['Loudness (db)'] = pd.to_numeric(df['Loudness (db)'], errors='coerce')

            df["Length"] = pd.to_timedelta("00:" + df["Length"]).dt.total_seconds().astype(int)

            df[N_cols] = df[N_cols].fillna(0)
            X = np.concatenate([X, df[N_cols].to_numpy()], axis=1)


        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        print("\nData con el spliting...")
        print(f"Label distribution en TRAIN: {y_train.value_counts().to_dict()}")
        print(f"Label distribution en TEST: {y_test.value_counts().to_dict()}")
        print("\n")

        # Apliying Undersampling
        if UNDERSAMPLING:
            X_train, y_train = undersample(X_train, y_train)


        if USE_SMOTE: 
            print("\n")
            print("Aplicando SMOTE oversampling...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            print(f"Nueva distribución de clases: {y_train.value_counts().to_dict()}")
        
        
        if TESTING:
            out = "test"
        else:
            out=""

        
        # output_dir = f"outputs{out}/undersample_{UNDERSAMPLING}_scaled_{SCALED}_steaming_{STEAMING}_removestw_{REMOVESTW}_numeric_{NUMERICCOlS}_useSmote_{USE_SMOTE}_MLP_{MLP_}_+_{MAX_FEATURES}_tfidf_{cols_type}/{embedding_type}"
        output_dir = f"outputs_d/6/{embedding_type}"
        

        if MLP_: 
            nets = {
                # Pequeña y rápida
                "1": [X_train.shape[1], 32, 1],  
                
                # Mediana
                "2": [X_train.shape[1], 64, 32, 1],  
                
                # Más profunda
                "3": [X_train.shape[1], 128, 64, 32, 1],  
                
                # Red grande
                "4": [X_train.shape[1], 256, 128, 64, 32, 1],  
                
                # Variante con capas decrecientes más suaves
                "5": [X_train.shape[1], 512, 256, 128, 64, 1],
                "6": [X_train.shape[1], 1024, 512, 256, 128, 64, 32, 1],
            }

            print("Resultados con MLP")
            ## Usaremos su porip escalador
            # Convertir data a tensores
            # Convertir a tensores
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            # y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # labels como enteros
            y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
            y_test_tensor  = torch.tensor(np.array(y_test), dtype=torch.long)

            X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
            # y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

            # Crear datasets
            trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
            testDataset  = TensorDataset(X_test_tensor,  y_test_tensor)

            trainLoader = DataLoader(trainDataset, batch_size=64, shuffle= True)
            testLoader  = DataLoader( testDataset, batch_size=64, shuffle=False)
            
            veces = 4
            n_params = []
            f1_scores = []
            for name, capas in nets.items():
                print(f"\nEntrenando red {name} con capas {capas}")
                net_ = MLP(capas)

                real_trainable_params = sum(p.numel() for p in net_.parameters() if p.requires_grad)
                trainable_params = sum(p.numel() for p in net_.parameters() if p.requires_grad)

                n_params.append(trainable_params)

                
                start_time = time.time()  

                f1_score_result = 0.0
                for i in range(veces):
                    print(f"\n--- Iteración {i+1} de {veces} para la red {name} ---")

                    net = MLP(capas).to(device)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.AdamW(net.parameters(), lr=0.001)

                    train_losses,test_losses, best_f1_score, best_pred, best_ephoc, best_labels, AUC_according_best_f1 = train_deep_rn(
                        net, trainLoader, testLoader, criterion, optimizer, device, 
                        n_epoch=_EPHOCS, target_f1=0.95, print_every=10
                    )
                    f1_score_result+=best_f1_score

                    os.makedirs(output_dir, exist_ok=True)
                    resumencito = evaluar_modelo(best_labels, best_pred, trainable_params, labels=('Not Explicit', 'Explicit'), save_dir=output_dir)


                f1_scores.append(f1_score_result/veces)
                print("real_trainable_params: ",  real_trainable_params)
                # resultados_globales[str(trainable_params)] = resumencito
                resultados_globales.setdefault(embedding_type, {})
                resultados_globales[embedding_type][f"MLP_{trainable_params}"] = resumencito

                end_time = time.time() 
                elapsed_time = end_time - start_time
                print(f"Tiempo total para red {name}: {elapsed_time:.2f} segundos")
        if SCALED:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        


        print("Saved on:", output_dir)
        resumen = train_models(X_train, X_test, y_train, y_test, dir_=output_dir, embedding_type=embedding_type)
        # resultados_globales[embedding_type] = resumen
        resultados_globales.setdefault(embedding_type, {})
        resultados_globales[embedding_type].update(resumen)


        # train_models_with_gridsearch
        # train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_=output_dir, embedding_type=embedding_type)
    i = 0
    print("\n\nResumen GLOBAL de métricas:")
    
    for emb_type, models_dict in resultados_globales.items():
        print(f"\n\nEMBEDDINGS TYPE: {emb_type.upper()}")
        for modelo, metricas in sorted(models_dict.items(), key=lambda x: x[1]["f1_score"], reverse=True):
            print(f"{modelo}: {metricas}")

    # CONF = f"undersample_{UNDERSAMPLING}_scaled_{SCALED}_removestw_{REMOVESTW}_5000tfidf"
    # print("\nYou are executing with this configuration:", CONF)
    
    
    
    print("\n========== CONFIGURATIONS ==========")
    configs = {
        "TESTING": TESTING,
        "UNDERSAMPLING": UNDERSAMPLING,
        "USE_SMOTE": USE_SMOTE,
        "SCALED": SCALED,
        "STEAMING": STEAMING,
        "REMOVE_STOPWORDS": REMOVESTW,
        "NUMERIC_COLS": NUMERICCOlS,
        "MAX_FEATURES_TFIDF": MAX_FEATURES,
        "TF-IDF Columns": COL_TF_IDF,
        "Numeric Columns": N_cols if NUMERICCOlS else "Not used"
    }
    for k, v in configs.items():
        print(f"{k}: {v}")
    print("====================================\n")



if __name__ == "__main__":
    main()
