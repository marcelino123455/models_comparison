#!/usr/bin/env python3
"""
Script simplificado para entrenar y guardar el modelo para la API
"""

import pandas as pd
import pickle
import os
from text_preprocessing import TextPreprocessor, TFIDFVectorizer
from models import NaiveBayes
import numpy as np

def train_api_model():
    """Entrenar y guardar el modelo para la API"""
    print("🔧 Entrenando modelo para API...")

    # Cargar dataset
    print("📂 Cargando dataset...")
    data = pd.read_csv('data/spotify_dataset.csv')

    # Tomar una muestra más pequeña para evitar problemas de memoria
    data = data.sample(n=min(10000, len(data)), random_state=42)

    # Preparar datos
    print("📊 Preparando datos...")
    data = data.dropna(subset=['text', 'Explicit'])

    # Convertir 'Yes'/'No' a binario
    data['Explicit_binary'] = (data['Explicit'] == 'Yes').astype(int)

    # Balancear clases
    explicit_data = data[data['Explicit_binary'] == 1]
    non_explicit_data = data[data['Explicit_binary'] == 0]

    min_class_size = min(len(explicit_data), len(non_explicit_data))
    min_class_size = min(min_class_size, 2000)  # Máximo 2000 por clase

    balanced_data = pd.concat([
        explicit_data.sample(n=min_class_size, random_state=42),
        non_explicit_data.sample(n=min_class_size, random_state=42)
    ])

    print(f"📈 Dataset balanceado: {len(balanced_data)} canciones")
    print(f"   Explícitas: {len(balanced_data[balanced_data['Explicit_binary'] == 1])}")
    print(f"   No explícitas: {len(balanced_data[balanced_data['Explicit_binary'] == 0])}")

    # Preparar textos y etiquetas
    texts = balanced_data['text'].tolist()
    labels = balanced_data['Explicit_binary'].tolist()

    # Dividir datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"📚 Conjunto de entrenamiento: {len(X_train)} canciones")
    print(f"🧪 Conjunto de prueba: {len(X_test)} canciones")

    # Crear y entrenar el preprocessor
    print("🔧 Entrenando preprocessor...")
    preprocessor = TextPreprocessor(language='english')
    vectorizer = TFIDFVectorizer(max_features=3000)

    # Preprocesar y vectorizar
    print("🔄 Procesando texto...")
    X_train_clean = [preprocessor.clean_text(text) for text in X_train]
    X_train_vectorized = vectorizer.fit_transform(X_train_clean)

    # Entrenar modelo Naive Bayes
    print("🧠 Entrenando modelo Naive Bayes...")
    model = NaiveBayes()
    model.fit(X_train_vectorized, np.array(y_train))

    # Evaluar modelo
    print("📊 Evaluando modelo...")
    X_test_clean = [preprocessor.clean_text(text) for text in X_test]
    X_test_vectorized = vectorizer.transform(X_test_clean)
    y_pred = model.predict(X_test_vectorized)

    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Resultados del modelo:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Crear pipeline completo
    class ModelPipeline:
        def __init__(self, preprocessor, vectorizer, model):
            self.preprocessor = preprocessor
            self.vectorizer = vectorizer
            self.model = model

        def predict(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            # Preprocesar y vectorizar
            cleaned = [self.preprocessor.clean_text(text) for text in texts]
            vectorized = self.vectorizer.transform(cleaned)
            return self.model.predict(vectorized)

        def predict_proba(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            # Preprocesar y vectorizar
            cleaned = [self.preprocessor.clean_text(text) for text in texts]
            vectorized = self.vectorizer.transform(cleaned)
            return self.model.predict_proba(vectorized)

    pipeline = ModelPipeline(preprocessor, vectorizer, model)

    # Crear directorio si no existe
    os.makedirs('saved_models', exist_ok=True)

    # Guardar pipeline
    print("💾 Guardando modelo...")
    with open('saved_models/explicit_lyrics_classifier.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("✅ Modelo guardado en 'saved_models/explicit_lyrics_classifier.pkl'")

    # Probar el modelo guardado
    print("🧪 Probando modelo guardado...")
    with open('saved_models/explicit_lyrics_classifier.pkl', 'rb') as f:
        loaded_pipeline = pickle.load(f)

    # Casos de prueba
    test_cases = [
        "I love you so much, you are beautiful and amazing",
        "This is some fucked up shit, damn this is crazy",
        "Happy birthday to you, happy birthday"
    ]

    for i, text in enumerate(test_cases, 1):
        prediction = loaded_pipeline.predict([text])[0]
        probabilities = loaded_pipeline.predict_proba([text])[0]
        status = "Explícita" if prediction == 1 else "No explícita"
        confidence = max(probabilities)

        print(f"   Prueba {i}: {status} (confianza: {confidence:.3f})")

    print("🎉 Modelo listo para la API!")
    return True

if __name__ == "__main__":
    train_api_model()
