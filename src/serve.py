from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from .config import FEATURES, PROCESSED_DIR, MODELS_DIR, SCALING_COLS
from .utils import load_artifact

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load artifacts once
scaler = load_artifact(PROCESSED_DIR / "scaler.pkl")
model = load_artifact(MODELS_DIR / "best_model.pkl")

# Input model matching your CSV feature names
class Patient(BaseModel):
    age: float = Field(..., ge=0)
    sex: int
    chest_pain_type: int
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: float
    exercise_induced_angina: int
    st_depression: float
    st_slope: int
    num_major_vessels: int
    thalassemia: int

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(p: Patient):
    row = pd.DataFrame([p.model_dump()])[FEATURES].copy()
    row[SCALING_COLS] = scaler.transform(row[SCALING_COLS])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[:, 1][0])
    else:
        s = model.decision_function(row)
        prob = float((s - s.min()) / (s.max() - s.min() + 1e-8))[0]

    pred = int(model.predict(row)[0])
    return {"prediction": pred, "probability": prob}
