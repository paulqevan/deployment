import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from joblib import load

from app.schemas import DiabetesFeatures, PredictionOut

MODEL_PATH = Path("model/model.joblib")
META_PATH = Path("model/metadata.json")

app = FastAPI(title="Diabetes ML API", version="1.0.0")

model = None
metadata = {"threshold": 0.5, "model_type": "RandomForestClassifier"}


@app.on_event("startup")
def _load_model():
    global model, metadata
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train it first: python src/train.py")

    model = load(MODEL_PATH)

    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: DiabetesFeatures):
    x = np.array([[
        payload.Pregnancies,
        payload.Glucose,
        payload.BloodPressure,
        payload.SkinThickness,
        payload.Insulin,
        payload.BMI,
        payload.DiabetesPedigreeFunction,
        payload.Age,
    ]], dtype=float)

    proba = float(model.predict_proba(x)[0, 1])
    threshold = float(metadata.get("threshold", 0.5))
    pred = int(proba >= threshold)

    return PredictionOut(
        prediction=pred,
        probability=proba,
        threshold=threshold,
        model=str(metadata.get("model_type", "RandomForestClassifier")),
    )

@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API (RandomForest + FastAPI + Docker)",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
        },
    }

