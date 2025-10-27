# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd  # ✅ Added import for DataFrame handling

# ------------------------------
# Paths and Model Loading
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Load model and preprocessing pipeline
MODEL = joblib.load(MODELS_DIR / "best_heart_model.pkl")
PREPROCESSOR = joblib.load(MODELS_DIR / "scaler (1).pkl")

# These are the original raw input features (not encoded)
RAW_FEATURES = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_induced_angina", "st_depression",
    "st_slope", "num_major_vessels", "thalassemia"
]

# ------------------------------
# FastAPI Setup
# ------------------------------
app = FastAPI(title="Heart Disease Predictor API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------
# Input Schema
# ------------------------------
class PredictionInput(BaseModel):
    data: Dict[str, float]

# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "API is running successfully"}

@app.get("/features")
def get_features():
    """Return required raw features for frontend."""
    return {"features": RAW_FEATURES}

@app.post("/predict")
def predict(payload: PredictionInput):
    """Make a prediction using the trained model."""
    try:
        # ✅ Convert input to a pandas DataFrame
        x_raw = pd.DataFrame([payload.data], columns=RAW_FEATURES)

        # ✅ Apply preprocessing (scaling + encoding)
        x_transformed = PREPROCESSOR.transform(x_raw)

        # ✅ Predict
        prob = float(MODEL.predict_proba(x_transformed)[0][1])
        pred = int(MODEL.predict(x_transformed)[0])

        return {
            "prediction": pred,
            "probability": round(prob, 4)
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed. Please check input feature format."
        }
