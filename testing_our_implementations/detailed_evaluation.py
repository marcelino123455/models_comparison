#!/usr/bin/env python3
"""
Evaluación detallada del modelo Naive Bayes
Este script realiza un análisis profundo del rendimiento del modelo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
from main import ExplicitLyricsClassifier
from text_preprocessing import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class DetailedEvaluator:
    """Evaluación detallada del modelo"""

    def __init__(self, model_path='saved_models/explicit_lyrics_classifier.pkl'):
        """
        Inicializar evaluador

        Args:
            model_path: Ruta al modelo guardado
        """
        self.model_path = model_path
        self.classifier = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None

    def load_model_and_data(self, csv_path='data/spotify_dataset.csv', sample_size=50000):
        """
        Cargar modelo y datos para evaluación

        Args:
            csv_path: Ruta al dataset
            sample_size: Número de muestras
        """
        print("🔄 Cargando modelo y datos...")

        # Cargar modelo entrenado
        try:
            self.classifier = ExplicitLyricsClassifier.load_model(self.model_path)
            print(f"✅ Modelo cargado desde {self.model_path}")
        except:
            print("❌ No se pudo cargar el modelo. Entrenando nuevo modelo...")
            self.classifier = self._train_new_model(csv_path, sample_size)

        # Cargar y preparar datos de prueba
        X, y = self._prepare_test_data(csv_path, sample_size)

        # División para evaluación
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        print(f"📊 Datos preparados - Test set: {len(self.X_test)} muestras")
        print(f"   Distribución: {Counter(self.y_test)}")

    def _train_new_model(self, csv_path, sample_size):
        """Entrenar nuevo modelo si no existe"""
        classifier = ExplicitLyricsClassifier(model_type='naive_bayes')
        X, y = self._prepare_test_data(csv_path, sample_size)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        classifier.fit(X_train, y_train)
        return classifier

    def _prepare_test_data(self, csv_path, sample_size):
        """Preparar datos de prueba"""
        # Usar el mismo método que en main.py
        temp_classifier = ExplicitLyricsClassifier(model_type='naive_bayes')
        return temp_classifier.load_and_preprocess_data(csv_path, sample_size, balance_data=True)

    def basic_evaluation(self):
        """Evaluación básica del modelo"""
        print("\n" + "="*60)
        print("🎯 EVALUACIÓN BÁSICA DEL MODELO")
        print("="*60)

        # Hacer predicciones
        self.y_pred = self.classifier.predict(self.X_test)

        if hasattr(self.classifier.model, 'predict_proba'):
            self.y_pred_proba = self.classifier.predict_proba(self.X_test)

        # Métricas básicas
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)

        print(f"📈 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"📈 Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"📈 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

        if self.y_pred_proba is not None:
            # AUC-ROC solo si tenemos probabilidades
            try:
                auc = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
                print(f"📈 AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")
            except:
                print("📈 AUC-ROC:   No disponible")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def confusion_matrix_analysis(self):
        """Análisis detallado de la matriz de confusión"""
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DE MATRIZ DE CONFUSIÓN")
        print("="*60)

        cm = confusion_matrix(self.y_test, self.y_pred)

        # Mostrar matriz con números y porcentajes
        print("\nMatriz de Confusión (números absolutos):")
        print(f"                 Predicho")
        print(f"               No    Sí")
        print(f"Real No     {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"     Sí     {cm[1,0]:4d}  {cm[1,1]:4d}")

        # Calcular porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print(f"\nMatriz de Confusión (porcentajes por fila):")
        print(f"                 Predicho")
        print(f"               No      Sí")
        print(f"Real No   {cm_percent[0,0]:5.1f}%  {cm_percent[0,1]:5.1f}%")
        print(f"     Sí   {cm_percent[1,0]:5.1f}%  {cm_percent[1,1]:5.1f}%")

        # Análisis de errores
        total_samples = len(self.y_test)
        true_positives = cm[1,1]
        true_negatives = cm[0,0]
        false_positives = cm[0,1]
        false_negatives = cm[1,0]

        print(f"\n📊 Análisis detallado:")
        print(f"   Verdaderos Positivos (TP): {true_positives:4d} ({true_positives/total_samples*100:.1f}%)")
        print(f"   Verdaderos Negativos (TN): {true_negatives:4d} ({true_negatives/total_samples*100:.1f}%)")
        print(f"   Falsos Positivos (FP):     {false_positives:4d} ({false_positives/total_samples*100:.1f}%)")
        print(f"   Falsos Negativos (FN):     {false_negatives:4d} ({false_negatives/total_samples*100:.1f}%)")

        # Guardar visualización
        self._plot_confusion_matrix(cm)

        return cm

    def _plot_confusion_matrix(self, cm):
        """Crear visualización de matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Explícito', 'Explícito'],
                   yticklabels=['No Explícito', 'Explícito'])
        plt.title('Matriz de Confusión - Naive Bayes')
        plt.ylabel('Valores Reales')
        plt.xlabel('Predicciones')

        # Guardar figura
        plt.tight_layout()
        plt.savefig('graphs/detailed_confusion_matrix_naive_bayes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("💾 Matriz de confusión guardada en graphs/detailed_confusion_matrix_naive_bayes.png")

    def cross_validation_analysis(self):
        """Análisis con validación cruzada"""
        print("\n" + "="*60)
        print("🔄 VALIDACIÓN CRUZADA (5-FOLD)")
        print("="*60)

        # Usar datos originales sin división para CV
        print("🔄 Preparando datos para validación cruzada...")
        X, y = self._prepare_test_data('data/spotify_dataset.csv', 15000)  # Muestra reducida para CV

        # Validación cruzada estratificada manual
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # Almacenar resultados
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        print("🔄 Ejecutando validación cruzada manual...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/5...")

            # Dividir datos
            x_train_fold = X.iloc[train_idx]
            x_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Crear y entrenar clasificador para este fold
            fold_classifier = ExplicitLyricsClassifier(model_type='naive_bayes')
            fold_classifier.fit(x_train_fold, y_train_fold)

            # Hacer predicciones
            y_pred_fold = fold_classifier.predict(x_val_fold)

            # Calcular métricas
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred_fold))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred_fold))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred_fold))

        # Mostrar resultados
        print("\n📊 Resultados de Validación Cruzada (5-fold):")
        for metric, scores in cv_scores.items():
            scores_array = np.array(scores)
            print(f"📊 {metric.capitalize():9}: {scores_array.mean():.4f} (±{scores_array.std()*2:.4f})")
            print(f"           Folds: {[f'{s:.3f}' for s in scores]}")

        return cv_scores

    def error_analysis(self):
        """Análisis de errores específicos"""
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DE ERRORES")
        print("="*60)

        # Encontrar errores
        errors = self.y_test != self.y_pred

        if not errors.any():
            print("✅ ¡No hay errores en el conjunto de prueba!")
            return

        # Obtener indices donde hay errores y filtrar los primeros 10
        error_mask = errors.values if hasattr(errors, 'values') else errors
        error_positions = np.nonzero(error_mask)[0][:10]

        print(f"📊 Total de errores: {errors.sum()}/{len(self.y_test)} ({errors.sum()/len(self.y_test)*100:.1f}%)")
        print("\n🔍 Ejemplos de errores (primeros 10):")
        print("-" * 50)

        for i, pos in enumerate(error_positions):
            actual = self.y_test.iloc[pos] if hasattr(self.y_test, 'iloc') else self.y_test[pos]
            predicted = self.y_pred[pos]
            text_sample = self.X_test.iloc[pos] if hasattr(self.X_test, 'iloc') else self.X_test[pos]

            # Truncar texto si es muy largo
            text_display = text_sample[:100] + "..." if len(str(text_sample)) > 100 else str(text_sample)

            print(f"\n{i+1}. Real: {'Explícito' if actual else 'No Explícito'} | "
                  f"Predicho: {'Explícito' if predicted else 'No Explícito'}")
            print(f"   Texto: {text_display}")

            if self.y_pred_proba is not None:
                conf = max(self.y_pred_proba[pos])
                print(f"   Confianza: {conf:.3f}")

    def feature_importance_analysis(self):
        """Análisis de importancia de características"""
        print("\n" + "="*60)
        print("🎯 ANÁLISIS DE CARACTERÍSTICAS IMPORTANTES")
        print("="*60)

        if not hasattr(self.classifier.model, 'feature_probs'):
            print("❌ El modelo no tiene información de características disponible")
            return

        try:
            # Obtener vocabulario del vectorizador
            if hasattr(self.classifier.vectorizer, 'feature_names_'):
                feature_names = self.classifier.vectorizer.feature_names_
            elif hasattr(self.classifier.vectorizer, 'get_feature_names'):
                feature_names = self.classifier.vectorizer.get_feature_names()
            else:
                print("❌ No se pudo obtener los nombres de las características del vectorizador")
                return

            # Calcular diferencia entre probabilidades de clases
            explicit_probs = self.classifier.model.feature_probs[1]
            non_explicit_probs = self.classifier.model.feature_probs[0]

            # Diferencia log de probabilidades (mayor = más indicativo de explícito)
            log_prob_diff = np.log(explicit_probs) - np.log(non_explicit_probs)

            # Top palabras para contenido explícito
            explicit_indices = np.argsort(log_prob_diff)[-20:][::-1]
            print("🔥 Top 20 palabras indicativas de CONTENIDO EXPLÍCITO:")
            for i, idx in enumerate(explicit_indices):
                word = feature_names[idx]
                score = log_prob_diff[idx]
                print(f"   {i+1:2d}. {word:<15} (score: {score:6.3f})")

            # Top palabras para contenido no explícito
            non_explicit_indices = np.argsort(log_prob_diff)[:20]
            print("\n✨ Top 20 palabras indicativas de CONTENIDO NO EXPLÍCITO:")
            for i, idx in enumerate(non_explicit_indices):
                word = feature_names[idx]
                score = log_prob_diff[idx]
                print(f"   {i+1:2d}. {word:<15} (score: {score:6.3f})")

        except Exception as e:
            print(f"❌ Error en análisis de características: {e}")

    def run_complete_evaluation(self):
        """Ejecutar evaluación completa"""
        print("🚀 INICIANDO EVALUACIÓN COMPLETA DEL MODELO NAIVE BAYES")
        print("="*70)

        # Cargar modelo y datos
        self.load_model_and_data()

        # Ejecutar todas las evaluaciones
        basic_results = self.basic_evaluation()
        self.confusion_matrix_analysis()
        self.cross_validation_analysis()
        self.error_analysis()
        self.feature_importance_analysis()

        # Resumen final
        print("\n" + "="*70)
        print("📋 RESUMEN FINAL DE EVALUACIÓN")
        print("="*70)
        print("✅ Modelo evaluado: Naive Bayes")
        print(f"📊 Accuracy: {basic_results['accuracy']:.4f}")
        print(f"📊 F1-Score: {basic_results['f1_score']:.4f}")
        # Determinar nivel de rendimiento
        f1_score = basic_results['f1_score']
        if f1_score > 0.75:
            performance_level = "BUEN"
        elif f1_score > 0.6:
            performance_level = "REGULAR"
        else:
            performance_level = "BAJO"

        print(f"🎯 El modelo muestra {performance_level} rendimiento")

        if basic_results['precision'] > 0.8:
            print("✅ Alta precisión - Pocas falsas alarmas")
        if basic_results['recall'] > 0.7:
            print("✅ Buen recall - Detecta la mayoría del contenido explícito")

        print("\n💾 Archivos generados:")
        print("   - graphs/detailed_confusion_matrix_naive_bayes.png")

def main():
    """Función principal"""
    evaluator = DetailedEvaluator()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
