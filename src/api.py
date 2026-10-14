
import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "mlflow.db"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DATABASE_PATH.as_posix()}",
)

MODEL_NAMES = {
    "heart_rf": "HeartDiseaseRandomForest",
    "heart_logistic": "HeartDiseaseLogisticRegression",
    "boston_rf": "BostonHousingRandomForestRegressor",
}

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]

BOSTON_FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="MLOps Model Deployment API",
    description="Pre-production API for the required MLflow models.",
    version="1.0.0",
)


class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class BostonInput(BaseModel):
    CRIM: float = Field(ge=0)
    ZN: float = Field(ge=0)
    INDUS: float = Field(ge=0)
    CHAS: float = Field(ge=0, le=1)
    NOX: float = Field(gt=0)
    RM: float = Field(gt=0)
    AGE: float = Field(ge=0, le=100)
    DIS: float = Field(gt=0)
    RAD: float = Field(ge=0)
    TAX: float = Field(ge=0)
    PTRATIO: float = Field(gt=0)
    B: float
    LSTAT: float = Field(ge=0)


def predict_with_model(model_name, alias, features):
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    frame = pd.DataFrame([features])
    prediction = model.predict(frame)

    if len(prediction) != 1:
        raise RuntimeError(
            f"Unexpected prediction count for {model_name}"
        )

    value = float(prediction[0])

    if not pd.notna(value):
        raise RuntimeError(
            f"Model {model_name} returned a non-finite prediction"
        )

    return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/heart/random-forest")
def predict_heart_rf(data: HeartInput):
    try:
        result = predict_with_model(
            MODEL_NAMES["heart_rf"],
            "Challenger-pre-test",
            data.model_dump(),
        )
        return {"model": MODEL_NAMES["heart_rf"], "prediction": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/heart/logistic")
def predict_heart_logistic(data: HeartInput):
    try:
        result = predict_with_model(
            MODEL_NAMES["heart_logistic"],
            "Challenger-pre-test",
            data.model_dump(),
        )
        return {
            "model": MODEL_NAMES["heart_logistic"],
            "prediction": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/boston")
def predict_boston(data: BostonInput):
    try:
        result = predict_with_model(
            MODEL_NAMES["boston_rf"],
            "Challenger-pre-test",
            data.model_dump(),
        )
        return {
            "model": MODEL_NAMES["boston_rf"],
            "prediction": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
