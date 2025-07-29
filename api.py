"""
API FastAPI para clasificación de letras explícitas
Utiliza el modelo entrenado para predecir si una letra de canción es explícita o no.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Explicit Lyrics Classifier API",
    description="API para clasificar letras de canciones como explícitas o no explícitas",
    version="1.0.0"
)

# Modelos de datos para la API
class LyricsRequest(BaseModel):
    lyrics: str
    song_title: Optional[str] = None
    artist: Optional[str] = None

class PredictionResponse(BaseModel):
    is_explicit: bool
    confidence: float
    prediction_class: str
    probabilities: Dict[str, float]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Variable global para almacenar el modelo cargado
model_pipeline = None

def load_model():
    """Cargar el modelo entrenado desde el archivo pickle"""
    global model_pipeline

    model_path = "saved_models/explicit_lyrics_classifier.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model_pipeline = pickle.load(f)
        logger.info("Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        return False

# Cargar modelo al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando API de clasificación de letras explícitas...")
    success = load_model()
    if not success:
        logger.error("Fallo al cargar el modelo durante el inicio")

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint raíz para verificar el estado de la API"""
    return HealthResponse(
        status="running",
        model_loaded=model_pipeline is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud para verificar el estado del servicio"""
    return HealthResponse(
        status="healthy" if model_pipeline is not None else "unhealthy",
        model_loaded=model_pipeline is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_explicit_lyrics(request: LyricsRequest):
    """
    Predecir si unas letras de canción son explícitas

    Args:
        request: Objeto con las letras de la canción y metadatos opcionales

    Returns:
        Predicción con probabilidades y metadatos
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Intente recargar el servicio."
        )

    if not request.lyrics or not request.lyrics.strip():
        raise HTTPException(
            status_code=400,
            detail="Las letras de la canción no pueden estar vacías"
        )

    try:
        # Realizar predicción
        prediction = model_pipeline.predict([request.lyrics])
        probabilities = model_pipeline.predict_proba([request.lyrics])

        # El modelo devuelve 0 para no explícito, 1 para explícito
        is_explicit = bool(prediction[0])
        confidence = float(max(probabilities[0]))

        # Crear diccionario de probabilidades
        prob_dict = {
            "not_explicit": float(probabilities[0][0]),
            "explicit": float(probabilities[0][1])
        }

        # Metadatos adicionales
        metadata = {
            "lyrics_length": len(request.lyrics),
            "word_count": len(request.lyrics.split()),
            "timestamp": datetime.now().isoformat()
        }

        if request.song_title:
            metadata["song_title"] = request.song_title
        if request.artist:
            metadata["artist"] = request.artist

        response = PredictionResponse(
            is_explicit=is_explicit,
            confidence=confidence,
            prediction_class="explicit" if is_explicit else "not_explicit",
            probabilities=prob_dict,
            metadata=metadata
        )

        logger.info(f"Predicción realizada: {response.prediction_class} (confianza: {confidence:.3f})")
        return response

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor durante la predicción: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch_lyrics(lyrics_list: List[str]):
    """
    Predecir múltiples letras de canciones en lote

    Args:
        lyrics_list: Lista de letras de canciones

    Returns:
        Lista de predicciones
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Intente recargar el servicio."
        )

    if not lyrics_list or len(lyrics_list) == 0:
        raise HTTPException(
            status_code=400,
            detail="La lista de letras no puede estar vacía"
        )

    if len(lyrics_list) > 100:  # Límite de seguridad
        raise HTTPException(
            status_code=400,
            detail="Máximo 100 letras por lote"
        )

    try:
        predictions = []

        for i, lyrics in enumerate(lyrics_list):
            if not lyrics or not lyrics.strip():
                predictions.append({
                    "index": i,
                    "error": "Letras vacías",
                    "is_explicit": None,
                    "confidence": None
                })
                continue

            prediction = model_pipeline.predict([lyrics])
            probabilities = model_pipeline.predict_proba([lyrics])

            is_explicit = bool(prediction[0])
            confidence = float(max(probabilities[0]))

            predictions.append({
                "index": i,
                "is_explicit": is_explicit,
                "confidence": confidence,
                "prediction_class": "explicit" if is_explicit else "not_explicit",
                "probabilities": {
                    "not_explicit": float(probabilities[0][0]),
                    "explicit": float(probabilities[0][1])
                }
            })

        logger.info(f"Predicción en lote realizada para {len(lyrics_list)} elementos")
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Error durante la predicción en lote: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor durante la predicción en lote: {str(e)}"
        )

@app.post("/reload-model")
async def reload_model():
    """Recargar el modelo desde disco"""
    try:
        success = load_model()
        if success:
            return {"message": "Modelo recargado exitosamente"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Fallo al recargar el modelo"
            )
    except Exception as e:
        logger.error(f"Error al recargar modelo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al recargar el modelo: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
