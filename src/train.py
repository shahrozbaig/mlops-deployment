
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "passengers.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "titanic_model.joblib"
MLFLOW_DB_PATH = PROJECT_DIR / "mlflow.db"

EXPERIMENT_NAME = "Titanic-Survival-Prediction"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run: python src/create_dataset.py"
    )


# --------------------------------------------------
# 2. Configure MLflow with SQLite
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 3. Load and validate the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    "passenger_id",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "survived",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing these columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("The dataset contains no records.")

if df["survived"].nunique() != 2:
    raise ValueError(
        "The target column 'survived' must contain both 0 and 1."
    )

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head().to_string(index=False))


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

# We exclude the ID because it is just an identifier.
X = df.drop(columns=["survived", "passenger_id"])
y = df["survived"].astype(int)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]


# --------------------------------------------------
# 5. Prepare numeric and categorical data
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# 6. Create the machine-learning pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)


# --------------------------------------------------
# 7. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 8. Train, evaluate, and track the experiment
# --------------------------------------------------

with mlflow.start_run() as run:

    # Train the model using only the training data.
    model.fit(X_train, y_train)

    # Predict survival for the unseen test records.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # Record model settings and dataset information in MLflow.
    mlflow.log_params(
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "random_state": 42,
            "test_size": 0.20,
            "class_weight": "balanced",
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "number_of_features": X.shape[1],
        }
    )

    # Record model evaluation metrics.
    mlflow.log_metrics(
        {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        }
    )

    # Save the full fitted pipeline for use outside MLflow.
    joblib.dump(model, MODEL_PATH)

    # Log the trained model to MLflow.
    # Trust only the specific types reported by skops
    # for this locally created Random Forest pipeline.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="titanic_model",
        serialization_format="skops",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
        ],
        input_example=X_train.head(5),
    )

    # Log the separate Joblib file as an MLflow artifact.
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="exports",
    )

    # Display evaluation results.
    print("\nModel evaluation results")
    print("------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nMLflow Run ID:", run.info.run_id)

# This section runs only after the MLflow run completes.
print("\nTrained model saved to:", MODEL_PATH)
print("MLflow database:", MLFLOW_DB_PATH)
print("Training and MLflow logging completed successfully!")