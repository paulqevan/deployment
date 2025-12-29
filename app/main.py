import json
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import DiabetesFeatures, PredictionOut

MODEL_PATH = Path("model/model.joblib")
META_PATH = Path("model/metadata.json")

model = None
metadata = {"threshold": 0.5, "model_type": "RandomForestClassifier"}


def load_artifacts():
    global model, metadata

    if not MODEL_PATH.exists():
        model = None
        raise FileNotFoundError("Model not found. Train it first: python src/train.py")

    model = load(MODEL_PATH)

    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_artifacts()
    yield
    # Shutdown (rien à faire)


app = FastAPI(title="Diabetes ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://paulqevan.github.io"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API (RandomForest + FastAPI + Docker)",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "metadata": "/metadata",
        },
    }


@app.get("/metadata")
def get_metadata():
    return metadata



@app.post("/predict", response_model=PredictionOut)
def predict(payload: DiabetesFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_order = metadata.get(
        "features",
        [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    x = pd.DataFrame(
        [{
            "Pregnancies": payload.Pregnancies,
            "Glucose": payload.Glucose,
            "BloodPressure": payload.BloodPressure,
            "SkinThickness": payload.SkinThickness,
            "Insulin": payload.Insulin,
            "BMI": payload.BMI,
            "DiabetesPedigreeFunction": payload.DiabetesPedigreeFunction,
            "Age": payload.Age,
        }]
    )[feature_order]

    proba = float(model.predict_proba(x)[0, 1])
    threshold = float(metadata.get("threshold", 0.5))
    pred = int(proba >= threshold)

    return PredictionOut(
        prediction=pred,
        probability=proba,
        threshold=threshold,
        model=str(metadata.get("model_type", "RandomForestClassifier")),
    )

