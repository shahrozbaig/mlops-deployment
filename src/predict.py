
from pathlib import Path

import joblib
import pandas as pd


# Find the project directory and saved model.
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "titanic_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run python src/train.py first."
    )

# Load the already-trained model.
model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")

# Details for a new, fictional passenger.
passenger = pd.DataFrame([
    {
        "pclass": 1,
        "sex": "female",
        "age": 25.0,
        "sibsp": 0,
        "parch": 1,
        "fare": 80.0,
        "embarked": "S",
    }
])

# Make a prediction.
prediction = int(model.predict(passenger)[0])
survival_probability = float(
    model.predict_proba(passenger)[0][1]
)

result = "Survived" if prediction == 1 else "Did not survive"

print("\nPassenger details:")
print(passenger.to_string(index=False))

print("\nPrediction result:")
print(f"Predicted outcome: {result}")
print(
    f"Model-estimated survival probability: "
    f"{survival_probability:.2%}"
)