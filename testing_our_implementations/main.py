import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import joblib
import os
from tqdm import tqdm

from text_preprocessing import TextPreprocessor, TFIDFVectorizer
from models import LogisticRegression, NaiveBayes, SVM

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def undersample(X, y, random_state=42):
    df = pd.DataFrame({'text': X, 'label': y})
    counts = df['label'].value_counts()
    n_min = counts.min()
    df_bal = df.groupby('label', group_keys=False) \
               .apply(lambda grp: grp.sample(n=n_min, random_state=random_state))
    X_bal = df_bal['text']
    y_bal = df_bal['label']

    print("Data Balanceada...")
    print(f"Label distribution: {y_bal.value_counts().to_dict()}")


    # print(X_bal.shape)
    return X_bal, y_bal

class ExplicitLyricsClassifier:
    """
    Main classifier for explicit lyrics detection
    """

    def __init__(self, model_type: str = 'logistic_regression', random_state: int = RANDOM_STATE):
        """
        Initialize the classifier

        Args:
            model_type: Type of model ('logistic_regression', 'naive_bayes', 'svm')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFVectorizer(
            max_features=5000,  # Mantener 5000 características
            # max_features = 1000, 
            min_df=3,  # Reducir min_df para más vocabulario
            max_df=0.92,  # Ajustar max_df
            ngram_range=(1, 2)  # Mantener bigramas
        )
        self.model = None
        self.is_fitted = False

        # Initialize model based on type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                learning_rate=0.01,  # Learning rate más conservador
                max_iterations=1000,  # Suficientes iteraciones
                regularization='l2',
                lambda_reg=0.01,  # Regularización moderada
                random_state=random_state
            )
        elif model_type == 'naive_bayes':
            self.model = NaiveBayes(alpha=1.0)  # Mantener suavizado estándar
        elif model_type == 'svm':
            self.model = SVM(
                C=1.0,                # Regularización
                learning_rate=0.01,   # Tasa de aprendizaje
                max_iterations=1000,  # Iteraciones
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported models: 'logistic_regression', 'naive_bayes'")

    def load_and_preprocess_data(self, csv_path: str, sample_size: int = None, balance_data: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the dataset

        Args:
            csv_path: Path to the CSV file
            sample_size: Number of samples to use (None for all)
            balance_data: Whether to balance the classes

        Returns:
            Preprocessed features and labels
        """
        print("Loading dataset...")
        df = pd.read_csv(csv_path)

        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Check if required columns exist
        if 'text' not in df.columns or 'Explicit' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'Explicit' columns")

        # Remove rows with missing text or labels
        df = df.dropna(subset=['text', 'Explicit'])

        # Sample data if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=self.random_state).reset_index(drop=True)

        print(f"Dataset shape after preprocessing: {df.shape}")

        # Convert explicit labels to binary (0: No, 1: Yes)
        df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)

        print(f"Label distribution:")
        print(df['Explicit_binary'].value_counts())

        # Preprocess text
        print("Preprocessing text...")
        tqdm.pandas(desc="Text preprocessing")
        df['processed_text'] = df['text'].progress_apply(
            lambda x: self.preprocessor.preprocess(x, remove_stopwords=True, apply_stemming=True)
        )

        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

        # Balance the dataset if requested
        if balance_data:
            print("Balancing dataset...")
            # Get counts for each class
            class_counts = df['Explicit_binary'].value_counts()
            min_class_count = min(class_counts)

            print(f"Original distribution: {dict(class_counts)}")
            print(f"Balancing to {min_class_count} samples per class")

            # Sample equal amounts from each class
            df_balanced = df.groupby('Explicit_binary').apply(
                lambda x: x.sample(n=min_class_count, random_state=self.random_state)
            ).reset_index(drop=True)

            df = df_balanced
            print(f"Balanced distribution: {dict(df['Explicit_binary'].value_counts())}")

        print(f"Final dataset shape: {df.shape}")

        return df['processed_text'], df['Explicit_binary']

    def load_precomputed_data(self, csv_path: str, npy_path: str, sample_size: int = None) -> Tuple[np.ndarray, pd.Series]:
        """
        Load precomputed LB vectors and corresponding labels.

        Args:
            csv_path: Path to the CSV file with the labels
            npy_path: Path to the .npy file containing precomputed TF-IDF vectors
            sample_size: Number of samples to use (None for all)

        Returns:
            Tuple of (LB matrix, labels)
        """
        print("Loading labels from CSV...")
        df = pd.read_csv(csv_path)

        print(f"Original dataset shape: {df.shape}")
        if 'Explicit' not in df.columns:
            raise ValueError("CSV must contain 'Explicit' column")

        # Remove missing labels
        df = df.dropna(subset=['Explicit'])

        # Convert to binary labels
        df['Explicit_binary'] = (df['Explicit'].str.lower() == 'yes').astype(int)

        print("Loading Lb vectors from .npy...")
        X_vec = np.load(npy_path)
        

        if len(df) != X_vec.shape[0]:
            raise ValueError(f"Mismatch between number of labels ({len(df)}) and number of TF-IDF vectors ({X_vec.shape[0]})")

        # Sample if requested
        if sample_size and sample_size < len(df):
            sampled_idx = df.sample(n=sample_size, random_state=self.random_state).index
            df = df.loc[sampled_idx].reset_index(drop=True)
            X_vec = X_vec[sampled_idx]

        df['processed_text'] = list(X_vec)

        print(f"Final shape of LB matrix: {X_vec.shape}")
        print(f"Label distribution: {df['Explicit_binary'].value_counts().to_dict()}")

        return df['processed_text'], df['Explicit_binary']


    def fit(self, X: pd.Series, y: pd.Series) -> 'ExplicitLyricsClassifier':
        """
        Fit the classifier

        Args:
            X: Text features
            y: Labels

        Returns:
            Self for chaining
        """
        print(f"Training {self.model_type} model...")

        first_elem = X.iloc[0]
        print("Input already vectorized (Precomputed Embeddings)")
        if isinstance(first_elem, np.ndarray):
            X_vec = np.stack(X.to_numpy())
        else:
            X_vec = self.vectorizer.fit_transform(X.tolist())

        # Vectorize text
        # print("Vectorizing text...")
        # X_vec = self.vectorizer.fit_transform(X.tolist())

        print(f"Feature matrix shape: {X_vec.shape}")

        # Train model
        self.model.fit(X_vec, y.values)
        self.is_fitted = True

        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Text features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        first = X.iloc[0]

        if isinstance(first, np.ndarray):
            X_vec = np.stack(X.to_numpy())
        else:
        # Preprocess and vectorize
            X_processed = X.apply(
                lambda x: self.preprocessor.preprocess(x, remove_stopwords=True, apply_stemming=True)
            )

            X_vec = self.vectorizer.transform(X_processed.tolist())

        return self.model.predict(X_vec)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Predict probabilities (only for models that support it)

        Args:
            X: Text features

        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_type} does not support probability prediction")

        # Preprocess and vectorize
        X_processed = X.apply(
            lambda x: self.preprocessor.preprocess(x, remove_stopwords=True, apply_stemming=True)
        )
        X_vec = self.vectorizer.transform(X_processed.tolist())

        return self.model.predict_proba(X_vec)

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'random_state': self.random_state
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'ExplicitLyricsClassifier':
        """Load a trained model"""
        model_data = joblib.load(filepath)

        classifier = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state']
        )

        classifier.model = model_data['model']
        classifier.vectorizer = model_data['vectorizer']
        classifier.preprocessor = model_data['preprocessor']
        classifier.is_fitted = True

        return classifier


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", filename: str = "modelo.png"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Explicit', 'Explicit'],
                yticklabels=['Not Explicit', 'Explicit'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def compare_models(X_train: pd.Series, X_test: pd.Series,
                  y_train: pd.Series, y_test: pd.Series,  output_dir: str = "matrices") -> Dict[str, Dict[str, Any]]:
    """
    Compare different models

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        Comparison results
    """
    models = ['logistic_regression', 'naive_bayes', 'svm']
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")

        # Train model
        classifier = ExplicitLyricsClassifier(model_type=model_type, random_state=RANDOM_STATE)
        classifier.fit(X_train, y_train)

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        results[model_type] = metrics

        # Print results
        print(f"\nResults for {model_type}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nMatriz de confusion: ")


        # Plot confusion matrix
        # Asegura que la carpeta de salida exista
        filename = f"{output_dir}/modelo_{model_type}.png"
        print(metrics['confusion_matrix'])
        plot_confusion_matrix(metrics['confusion_matrix'],
                            f"Confusion Matrix - {model_type.upper()}",
                            filename=filename)

    return results


def main():
    """Main function"""
    print("Explicit Lyrics Classification Project")
    print("="*50)

    ### CONFIGURACIONES ###
    TESTING = False
    # NPY_EMBB = False
    # If false we use TF-IDF vectors


    if TESTING:
        print("You are executing with a short example of the dataset")
        _SAMPLE_SIZE = 500
        # _EMBEDDINGS = _EMBEDDINGS[:1000]
    else:
        print("You are executing with all dataset")
        _SAMPLE_SIZE = None

    
    # Load and preprocess data
    csv_path = "./data/spotify_dataset_sin_duplicados_4.csv"


    ### LB AND TF-IDF

    for embedding_type in ['tfidf', 'lyrics_bert']:
        print(f"\n{'#' * 50}")
        print(f"Running experiment with {embedding_type.upper()} embeddings")
        print(f"{'#' * 50}")

        if embedding_type == 'lyrics_bert':
            print("Lyrics-Bert Embbedings will be used")
            # Use a sample for faster development (remove sample_size for full dataset)
            classifier = ExplicitLyricsClassifier()
            npy_path = "./data/lb_npy.npy" 
            X, y = classifier.load_precomputed_data(csv_path, npy_path, sample_size=_SAMPLE_SIZE)
        else: 
            print("TF-IDF Embbedings will be used")

            # import nltk
            # nltk.download('punkt_tab')
            classifier = ExplicitLyricsClassifier()
            # X, y = classifier.load_precomputed_data(csv_path, npy_path, sample_size=_SAMPLE_SIZE)
            X, y = classifier.load_and_preprocess_data(
                csv_path,
                sample_size=_SAMPLE_SIZE,
                balance_data=False
            )
    
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        print("\nData con el spliting...")
        print(f"Label distribution en TRAIN: {y_train.value_counts().to_dict()}")
        print(f"Label distribution en TEST: {y_test.value_counts().to_dict()}")

        # Aplicar Undersampling
        X_train_bal, y_train_bal = undersample(X_train, y_train, random_state=RANDOM_STATE)

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # Output directories
        _output_dir = f"outputs/conf_matrices_{embedding_type}"
        os.makedirs(_output_dir, exist_ok=True)

        # Compare models
        results = compare_models(X_train_bal, X_test, y_train_bal, y_test, output_dir=_output_dir)

        # Print summary
        print(f"\n{'='*50}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*50}")

        for model_type, metrics in results.items():
            print(f"{model_type.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print()

        # Train and save best model (highest F1-Score)
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        best_model_name = best_model[0]
        print(f"Training final model ({best_model_name.replace('_', ' ').title()})...")
        final_classifier = ExplicitLyricsClassifier(model_type=best_model_name, random_state=RANDOM_STATE)
        final_classifier.fit(X_train_bal, y_train_bal)

        # Save model
        os.makedirs('saved_models', exist_ok=True)
        save_path = f'saved_models/explicit_lyrics_classifier_{embedding_type}.pkl'
        final_classifier.save_model(save_path)
        

if __name__ == "__main__":
    main()
