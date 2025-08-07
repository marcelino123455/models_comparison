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
RANDOM_STATE = 42

def load_LB_embbedings(csv_path, npy_path, sample_size=None):
    print("Loading Lb vectors from: ",csv_path)
    
    X_vec = np.load(npy_path)
    df = pd.read_csv(csv_path)
    df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)

    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)
        X = X_vec[sampled_idx]
    y = df['Explicit_binary']
    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y





def load_TFIDF_embbedings(csv_path, sample_size=None, columns_tf_idfizable = ['text'], max_features = 1000):
    
    # Read dataset in csv
    df = pd.read_csv(csv_path)

    if sample_size and sample_size < len(df):
        sampled_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        df = df.loc[sampled_idx].reset_index(drop=True)

 
    # Chose the columns to use TF-IDF
    df['combined_text'] = df[columns_tf_idfizable].fillna('').agg(' '.join, axis=1)
    
    df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)
    y = df['Explicit_binary']


    # Aplying TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df['combined_text'])

    X = X_tfidf.toarray()
    print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def undersample(X, y, random_state= RANDOM_STATE):
    print("="*50)
    print("Apliying UNDERSAMPLE")
    df = pd.DataFrame(X)
    df['label'] = y
    counts = df['label'].value_counts()
    n_min = counts.min()
    df_bal = df.groupby('label', group_keys=False) \
               .apply(lambda grp: grp.sample(n=n_min, random_state=RANDOM_STATE))
    
    print(f"Label distribution: {df_bal['label'].value_counts().to_dict()}")
    
    X = df_bal.drop(columns=['label']).values
    y = df_bal['label'].values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{'='*30}")
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred))



def train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_ = "output"):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=500),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ['lbfgs', 'liblinear'],
                "penalty": ['l2']
            }
        },
        "SVM": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ['linear', 'rbf'],
                "gamma": ['scale', 'auto']
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "criterion": ['gini', 'entropy'],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        }
    }

    for name, mp in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training model: {name}")

        model = mp["model"]
        param_grid = mp["params"]

        # Para Naive Bayes, GridSearch no acepta directamente -> usar pipeline
        if isinstance(model, GaussianNB):
            pipe = Pipeline([
                ("model", model)
            ])
            grid = GridSearchCV(pipe, {"model__" + k: v for k, v in param_grid.items()}, cv=5)
        else:
            grid = GridSearchCV(model, param_grid, cv=5)

        grid.fit(X_train, y_train)  

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"Best Params: {grid.best_params_}")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, )
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

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

        # Save
        os.makedirs(dir_, exist_ok=True)
        filename = os.path.join(dir_, f"conf_matrix_{name.replace(' ', '_').lower()}.png")
        plt.savefig(filename)
        print(f"Confusion matrix saved as: {filename}")
        plt.close()


def main():
    ### CONFIGURATIONS ###
    TESTING = True
    UNDERSAMPLING = True
    SCALED = True
    STEAMING = False
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

            X, y = load_TFIDF_embbedings(csv_path, sample_size=_SAMPLE_SIZE, columns_tf_idfizable = ['text'])    
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        if UNDERSAMPLING:
            X_train, y_train = undersample(X_train, y_train)

        if SCALED:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        output_dir = f"outputs/undersample_{UNDERSAMPLING}_scaled_{SCALED}_steaming_{STEAMING}/{embedding_type}"
        train_models_with_gridsearch(X_train, X_test, y_train, y_test, dir_=output_dir)




if __name__ == "__main__":
    main()
