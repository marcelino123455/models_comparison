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
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
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

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

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




def load_TFIDF_embbedings(csv_path, sample_size=None, columns_tf_idfizable = ['text'], max_features = 5000, steaming= False, remove_stopwords = True, faltantes_path = "faltantes_according_token.json"):
    
    # Read dataset in csv
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, nrows=sample_size)
    indices_to_remove = get_array(faltantes_path)
    if indices_to_remove[-1]>df.shape[0]:
        print("You are executing in testing way")
    else:
        df = df.drop(indices_to_remove).reset_index(drop=True)

 
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

    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")

    return df['processed_text'], y, vectorizer



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
 


# ['tfidf', 'lyrics_bert']
def train_models(X_train, X_test, y_train, y_test, dir_ = "output", embedding_type="tfidf", nets = None):
    models = {
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

    # if embedding_type == "tfidf":
    #     models["Naive Bayes"] = MultinomialNB(alpha=1.0)
    # else:
    #     models["Naive Bayes"] = GaussianNB()

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


from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import os, joblib, numpy as np
def sparse_to_dense(X):
    return X.toarray()
def train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_="output", embedding_type="tfidf", cv=5, val_size=0.2, vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=3,
        max_df=0.92,
        ngram_range=(1, 2)
    )
 ):
    os.makedirs(dir_, exist_ok=True)
    model_base = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [6, 8],
        "classifier__learning_rate": [ 0.1],
        "classifier__subsample": [0.7, 0.8],
        "classifier__colsample_bytree": [ 0.8]
    }

    to_dense = FunctionTransformer(sparse_to_dense, accept_sparse=True)

    
    pipeline = ImbPipeline([
        ('tfidf', vectorizer),  
        ('to_dense', to_dense),
        ('scaler', MinMaxScaler()), 
        ("undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("classifier", model_base)
    ])

   

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"Mejores parámetros encontrados: {grid.best_params_}")

    y_pred_test = grid.predict(X_test)
    test_scores = evaluar_modelo(y_test, y_pred_test, save_dir=dir_)
    
    modelo_final_path = os.path.join(dir_, "best_model_XGBoost.pkl")
    joblib.dump(grid.best_estimator_, modelo_final_path)
    print(f"Modelo final guardado en: {modelo_final_path}")

    resultados = {
        "mejor_modelo": "XGBoost",
        "mejores_parametros": grid.best_params_,
        "cv_f1_mean": grid.best_score_,
        "cv_f1_std": grid.cv_results_['std_test_score'][grid.best_index_],
        "test_scores": test_scores
    }

    print("\nEntrenando modelo final con TODO el dataset...")
    # Combinar train + test
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    final_pipeline = ImbPipeline([
        ('tfidf', vectorizer.set_params(**grid.best_estimator_.named_steps['tfidf'].get_params())),
        ('to_dense', to_dense),
        ('scaler', MinMaxScaler()),
        ("undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("classifier", XGBClassifier(
            n_estimators=grid.best_params_["classifier__n_estimators"],
            max_depth=grid.best_params_["classifier__max_depth"],
            learning_rate=grid.best_params_["classifier__learning_rate"],
            subsample=grid.best_params_["classifier__subsample"],
            colsample_bytree=grid.best_params_["classifier__colsample_bytree"],
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])

    final_pipeline.fit(X_full, y_full)
    modelo_grid_path = os.path.join(dir_, "best_model_XGBoost_gridsearch.pkl")
    joblib.dump(grid.best_estimator_, modelo_grid_path)
    print(f"Modelo del GridSearch guardado en: {modelo_grid_path}")
 

    return grid.best_estimator_, resultados


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
    MAX_FEATURES = 5000


    # path_DATA = "../../../../data"
    path_DATA = "../../data"
    if USE_SMOTE and UNDERSAMPLING:
        print("FAIL: You are using smote and undersampling please check")
        print(f"\n")
    
    resultados_globales = {}

    T = ['text']

    COL_TF_IDF = T
    print("For TF-IDF embbedings you are selecteing this columns:" )
    print("-->", COL_TF_IDF)

    if TESTING:
        _SAMPLE_SIZE = 100
        print(f"\nYou are executing with [EXAMPLE] of {_SAMPLE_SIZE} songs")
        # _EMBEDDINGS = _EMBEDDINGS[:1000]
    else:
        print("You are executing with [ALL] dataset")
        _SAMPLE_SIZE = None
        

        
    csv_path = f"{path_DATA}/spotify_dataset_sin_duplicados_4.csv"



    # A) Embeddings types
    embb_types = ['tfidf']
    for embedding_type in embb_types:

        if embedding_type == 'tfidf':
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")
            processed_text, y, vectorizer  = load_TFIDF_embbedings(csv_path, sample_size=_SAMPLE_SIZE, columns_tf_idfizable = COL_TF_IDF,  max_features = MAX_FEATURES, steaming= STEAMING, remove_stopwords = REMOVESTW)   
            
            X_train, X_test, y_train, y_test = train_test_split(
                processed_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            print("\nData con el spliting...")
            print(f"Label distribution en TRAIN: {y_train.value_counts().to_dict()}")
            print(f"Label distribution en TEST: {y_test.value_counts().to_dict()}")
            print("\n")

           

        # Apliying Undersampling
        # if UNDERSAMPLING:
        #     X_train, y_train = undersample(X_train, y_train)
        
        if TESTING:
            out = "test"
        else:
            out=""
        # output_dir = f"outputs{out}/undersample_{UNDERSAMPLING}_scaled_{SCALED}_steaming_{STEAMING}_removestw_{REMOVESTW}_numeric_{NUMERICCOlS}_useSmote_{USE_SMOTE}_MLP_{MLP_}_+_{MAX_FEATURES}_tfidf_{cols_type}/{embedding_type}"
        output_dir = f"outputs_text_pytia{out}/27/{embedding_type}"
        

        best_model, resumen=train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_=output_dir, embedding_type=embedding_type)
        resultados_globales[embedding_type] = resumen
        print("\n\n===================== RESUMEN GLOBAL =====================")
        all_rows = []
        for emb_type, resumen_ in resultados_globales.items():
            all_rows.append({
                "Embedding": emb_type,
                "Modelo": resumen_["mejor_modelo"],
                "Accuracy": resumen_["test_scores"]["accuracy"],
                "Precision": resumen_["test_scores"]["precision"],
                "Recall": resumen_["test_scores"]["recall"],
                "F1-score": resumen_["test_scores"]["f1_score"],
                "CV_F1_mean": resumen_["cv_f1_mean"],
                "CV_F1_std": resumen_["cv_f1_std"],
                "Best_params": resumen_["mejores_parametros"]
            })

        df_resumen = pd.DataFrame(all_rows)
        print(df_resumen)
        output_csv = os.path.join(output_dir, "resumen_global.csv")
        df_resumen.to_csv(output_csv, index=False)
        print(f"Resumen global guardado en: {output_csv}")
       
    i = 0

    
    print("\n========== CONFIGURATIONS ==========")
    configs = {
        "TESTING": TESTING,
        "UNDERSAMPLING": UNDERSAMPLING,
        "USE_SMOTE": USE_SMOTE,
        "SCALED": SCALED,
        "STEAMING": STEAMING,
        "REMOVE_STOPWORDS": REMOVESTW,
        "MAX_FEATURES_TFIDF": MAX_FEATURES,
        "TF-IDF Columns": COL_TF_IDF,
    }
    for k, v in configs.items():
        print(f"{k}: {v}")
    print("====================================\n")



if __name__ == "__main__":
    main()
