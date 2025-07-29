"""
Pipeline para clasificación de letras explícitas
"""

class ModelPipeline:
    """Pipeline completo para clasificación de letras explícitas"""

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
