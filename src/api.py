
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# 1. Locate the saved model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "titanic_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run python src/train.py first."
    )

# 2. Load the trained model once when the API starts
model = joblib.load(MODEL_PATH)

# 3. Create the FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "An educational API that uses our trained "
        "machine-learning model to predict survival."
    ),
    version="1.0.0",
)


# 4. Define and validate the incoming passenger data
class Passenger(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["female", "male"]
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0, le=20)
    parch: int = Field(ge=0, le=20)
    fare: float = Field(ge=0)
    embarked: Literal["S", "C", "Q"]


# 5. Homepage
@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


# 6. Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# 7. Predict survival for a passenger
@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert the request into the format used during training.
    passenger_data = pd.DataFrame(
        [passenger.model_dump()]
    )

    # Generate the prediction and estimated probability.
    prediction = int(model.predict(passenger_data)[0])

    probability = float(
        model.predict_proba(passenger_data)[0][1]
    )

    return {
        "predicted_survival": prediction,
        "predicted_outcome": (
            "Survived" if prediction == 1
            else "Did not survive"
        ),
        "survival_probability": round(probability, 4),
        "note": (
            "Educational demonstration using synthetic data; "
            "not a reliable real-world survival estimate."
        ),
    }