import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

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





def load_TFIDF_embbedings(csv_path, sample_size=None, columns_tf_idfizable = ['text'], max_features = 5000, steaming= False, remove_stopwords = True):
    
    # Read dataset in csv
    df = pd.read_csv(csv_path)

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
# ['tfidf', 'lyrics_bert']
def train_models(X_train, X_test, y_train, y_test, dir_ = "output", embedding_type="tfidf"):
    models = {
        "Logistic Regression": LogisticRegression(
            penalty='l2',               # Regularización L2
            C=1/0.01,                   # Inverso de lambda_reg → C = 1 / λ
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
            criterion='gini',         
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1
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


def train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_ = "output", embedding_type="tfidf"):
    param_grids = {
        "Logistic Regression": {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
            "max_iter": [1000]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "max_iter": [1000]
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 20],
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
        param_grids["Naive Bayes"] = {} 
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
        cm = confusion_matrix(y_test, y_pred)
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

# # def train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_ = "output"):
#     models = {
#         "Logistic Regression": {
#             "model": LogisticRegression(max_iter=500),
#             "params": {
#                 "C": [0.01, 0.1, 1, 10],
#                 "solver": ['lbfgs', 'liblinear'],
#                 "penalty": ['l2']
#             }
#         },
#         "SVM": {
#             "model": SVC(),
#             "params": {
#                 "C": [0.1, 1, 10],
#                 "kernel": ['linear', 'rbf'],
#                 "gamma": ['scale', 'auto']
#             }
#         },
#         "Naive Bayes": {
#             "model": GaussianNB(),
#             "params": {
#                 "var_smoothing": [1e-9, 1e-8, 1e-7]
#             }
#         },
#         "Decision Tree": {
#             "model": DecisionTreeClassifier(),
#             "params": {
#                 "criterion": ['gini', 'entropy'],
#                 "max_depth": [3, 5, 10, None],
#                 "min_samples_split": [2, 5, 10]
#             }
#         }
#     }

#     for name, mp in models.items():
#         print(f"\n{'=' * 40}")
#         print(f"Training model: {name}")

#         model = mp["model"]
#         param_grid = mp["params"]

#         # Para Naive Bayes, GridSearch no acepta directamente -> usar pipeline
#         if isinstance(model, GaussianNB):
#             pipe = Pipeline([
#                 ("model", model)
#             ])
#             grid = GridSearchCV(pipe, {"model__" + k: v for k, v in param_grid.items()}, cv=5)
#         else:
#             grid = GridSearchCV(model, param_grid, cv=5)

#         grid.fit(X_train, y_train)  

#         best_model = grid.best_estimator_
#         y_pred = best_model.predict(X_test)

#         print(f"Best Params: {grid.best_params_}")

#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, )
#         rec = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)

#         print(f"Accuracy:  {acc:.4f}")
#         print(f"Precision: {prec:.4f}")
#         print(f"Recall:    {rec:.4f}")
#         print(f"F1-score:  {f1:.4f}")
#         print(classification_report(y_test, y_pred))
#         cm = confusion_matrix(y_test, y_pred)
#         print(cm)
#         plt.figure(figsize=(5, 4))
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt='d',
#             cmap='Blues',
#             xticklabels=['Not Explicit', 'Explicit'],
#             yticklabels=['Not Explicit', 'Explicit']
#         )
#         plt.title(f'Confusion Matrix - {name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.tight_layout()

#         # Save
#         os.makedirs(dir_, exist_ok=True)
#         filename = os.path.join(dir_, f"conf_matrix_{name.replace(' ', '_').lower()}.png")
#         plt.savefig(filename)
#         print(f"Confusion matrix saved as: {filename}")
#         plt.close()


def main():
    ### CONFIGURATIONS ###
    TESTING = False
    UNDERSAMPLING = True
    SCALED = True
    STEAMING = True
    REMOVESTW = False
    MAX_FEATURES = 5000
    if TESTING:
        _SAMPLE_SIZE = 1000
        print(f"You are executing with [EXAMPLE] of {_SAMPLE_SIZE} songs")
        # _EMBEDDINGS = _EMBEDDINGS[:1000]
    else:
        print("You are executing with [ALL] dataset")
        _SAMPLE_SIZE = None

    csv_path = "../data/spotify_dataset_sin_duplicados_4.csv"
    npy_path = "../data/lb_npy.npy" 

    # A) Embeddings types
    embb_types = ['tfidf', 'lyrics_bert']

    for embedding_type in embb_types:

        if embedding_type == 'lyrics_bert':
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")
            X, y = load_LB_embbedings(csv_path, npy_path, sample_size=_SAMPLE_SIZE)
        
        else: 
            print(f"\n{'#' * 50}")
            print(f"Running experiment with {embedding_type.upper()} embeddings")

            X, y = load_TFIDF_embbedings(csv_path, sample_size=_SAMPLE_SIZE, columns_tf_idfizable = ['text'],  max_features = MAX_FEATURES, steaming= STEAMING, remove_stopwords = REMOVESTW)    
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        print("\nData con el spliting...")
        print(f"Label distribution en TRAIN: {y_train.value_counts().to_dict()}")
        print(f"Label distribution en TEST: {y_test.value_counts().to_dict()}")
        
        
        if UNDERSAMPLING:
            X_train, y_train = undersample(X_train, y_train)

        if SCALED:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        if TESTING:
            out = "_tes_"
        else:
            out=""
        output_dir = f"outputs_grid{out}/undersample_{UNDERSAMPLING}_scaled_{SCALED}_steaming_{STEAMING}_removestw_{REMOVESTW}_{MAX_FEATURES}tfidf/{embedding_type}"
    
        # train_models(X_train, X_test, y_train, y_test, dir_=output_dir, embedding_type=embedding_type)
        # train_models_with_gridsearch
        train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_=output_dir, embedding_type=embedding_type)

    
    CONF = f"undersample_{UNDERSAMPLING}_scaled_{SCALED}_removestw_{REMOVESTW}_5000tfidf"
    print("You are executing with this configuration:", CONF)



if __name__ == "__main__":
    main()
