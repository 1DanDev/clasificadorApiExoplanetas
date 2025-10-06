# api_exoplanet.py

import numpy as np
import joblib
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

warnings.filterwarnings("ignore")


# Definir el modelo de datos para la entrada
class ExoplanetFeatures(BaseModel):
    prad: float  # Planetary Radius (in Earth radii)
    period: float  # Orbital Period (in days)
    teq: float  # Equilibrium Temperature (in Kelvin)
    depth: float  # Transit Depth (light blocked, e.g., 0.001)


# Definir el modelo de respuesta
class PredictionResult(BaseModel):
    probability: float
    confidence: str
    classification: str
    recommendation: str
    is_exoplanet: bool


class ExoplanetPredictor:
    def __init__(self, model_path="exoplanet_model.pkl"):
        """
        Loads the model and scaler when the object is initialized.
        """
        self.model = None
        self.scaler = None
        self.is_ready = False
        try:
            data = joblib.load(model_path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_ready = True
            print(f"✅ Model successfully loaded from '{model_path}'")
        except FileNotFoundError:
            print(f"❌ Error: Model file not found at '{model_path}'.")
            print("   Please run 'train_model.py' first to generate the model.")
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading the model: {e}")

    def predict(self, features: ExoplanetFeatures) -> PredictionResult:
        """
        Makes a prediction for a single candidate.
        """
        if not self.is_ready:
            raise HTTPException(
                status_code=500, detail="Model is not ready for predictions."
            )

        try:
            # Prepare the data for the model
            new_X = np.array(
                [[features.prad, features.period, features.teq, features.depth]]
            )
            print(f"🔍 Datos para el modelo: {new_X}")

            new_X_scaled = self.scaler.transform(new_X)
            print(f"🔍 Datos escalados: {new_X_scaled}")

            # Make predictions
            probability = self.model.predict_proba(new_X_scaled)[0]
            confirmed_prob = probability[1]

            print(f"🔍 Probabilidades crudas: {probability}")
            print(f"🔍 Probabilidad de exoplaneta: {confirmed_prob}")

            # Determine classification and confidence
            if confirmed_prob >= 0.8:
                confidence = "HIGH"
                classification = "STRONG CANDIDATE"
                recommendation = "Prioritize for follow-up observations."
                is_exoplanet = True
            elif confirmed_prob >= 0.6:
                confidence = "MEDIUM-HIGH"
                classification = "PROBABLE CANDIDATE"
                recommendation = "Further observation recommended."
                is_exoplanet = True
            elif confirmed_prob >= 0.4:
                confidence = "LOW"
                classification = "WEAK CANDIDATE"
                recommendation = "Deeper analysis required."
                is_exoplanet = False
            else:
                confidence = "VERY LOW"
                classification = "LIKELY A FALSE POSITIVE"
                recommendation = "Deprioritize or review input data."
                is_exoplanet = False

            result = PredictionResult(
                probability=confirmed_prob,
                confidence=confidence,
                classification=classification,
                recommendation=recommendation,
                is_exoplanet=is_exoplanet,
            )

            print(f"✅ Predicción exitosa: {result}")
            return result

        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")
            print(f"🔍 Tipo de error: {type(e).__name__}")
            import traceback

            print(f"🔍 Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Inicializar FastAPI y el predictor
app = FastAPI(
    title="Exoplanet Prediction API",
    description="API para predecir si un candidato es un exoplaneta basado en sus características físicas",
    version="1.0.0",
)

# Configurar CORS
origins = [
    "*",
    "http://localhost:4321",
    "http://127.0.0.1:4321",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el predictor
predictor = ExoplanetPredictor()

# Verificación del estado del modelo al iniciar
if not predictor.is_ready:
    print("🚨 ADVERTENCIA: La API está corriendo SIN el modelo cargado")
    print("   Las predicciones fallarán hasta que el modelo esté disponible")
else:
    print("🚀 API Exoplanet Prediction iniciada correctamente")


@app.get("/")
async def root():
    return {
        "message": "🌌 Exoplanet Prediction API",
        "status": "active",
        "model_ready": predictor.is_ready,
        "endpoints": {"docs": "/docs", "health": "/health", "predict": "/predict"},
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_exoplanet(features: ExoplanetFeatures):
    """
    Predice si un candidato es un exoplaneta basado en sus características.
    """
    print(f"📥 Datos recibidos para predicción: {features.dict()}")

    try:
        result = predictor.predict(features)
        print(f"✅ Predicción exitosa: {result}")
        return result
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        print(f"🔍 Tipo de error: {type(e).__name__}")
        import traceback

        print(f"🔍 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Verifica el estado de la API y del modelo.
    """
    return {
        "status": "healthy" if predictor.is_ready else "unhealthy",
        "model_loaded": predictor.is_ready,
        "timestamp": np.datetime64("now").astype(str),
    }


@app.get("/model-info")
async def model_info():
    """
    Retorna información sobre el modelo cargado.
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=500, detail="Model not loaded")

    model_type = type(predictor.model).__name__
    scaler_type = type(predictor.scaler).__name__

    return {
        "model_type": model_type,
        "scaler_type": scaler_type,
        "features_used": ["prad", "period", "teq", "depth"],
        "model_ready": predictor.is_ready,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

