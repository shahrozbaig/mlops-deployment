
from pathlib import Path
import math

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and MLflow configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HEART_PATH = DATA_DIR / "heart_disease.csv"
BOSTON_PATH = DATA_DIR / "boston_housing.csv"
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

if not HEART_PATH.exists():
    raise FileNotFoundError(
        f"Missing {HEART_PATH}. Run src/ingest_datasets.py first."
    )

if not BOSTON_PATH.exists():
    raise FileNotFoundError(
        f"Missing {BOSTON_PATH}. Run src/ingest_datasets.py first."
    )

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)
mlflow.set_experiment("MLOps-Required-Datasets")


# --------------------------------------------------
# 2. Load the required datasets
# --------------------------------------------------

heart = pd.read_csv(HEART_PATH)
boston = pd.read_csv(BOSTON_PATH)

print("Heart Disease dataset:", heart.shape)
print("Boston Housing dataset:", boston.shape)


# --------------------------------------------------
# 3. Configure Heart Disease features
# --------------------------------------------------

heart = heart.dropna(subset=["target"]).copy()

X_heart = heart.drop(columns=["target"])
y_heart = heart["target"].astype(int)

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


def make_heart_preprocessor():
    """Create a fresh preprocessing pipeline for each model."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# --------------------------------------------------
# 4. Common function to train a classifier
# --------------------------------------------------

def train_classifier(model_name, classifier):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_heart_preprocessor()),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_heart,
        y_heart,
        test_size=0.20,
        random_state=42,
        stratify=y_heart,
    )

    with mlflow.start_run(run_name=model_name) as run:
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        }

        mlflow.log_params(
            {
                "dataset": "Heart Disease",
                "model_name": model_name,
                "test_size": 0.20,
                "random_state": 42,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
            }
        )
        mlflow.log_metrics(metrics)

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        # This is a model created by our own training script.
        # Pickle serialization avoids the skops trusted-type
        # error encountered in the earlier practice run.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=model_name,
            serialization_format="pickle",
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="exports",
        )

        print(f"\n{model_name}")
        print("-" * len(model_name))
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("Run ID:", run.info.run_id)
        print("Saved model:", model_path)


# --------------------------------------------------
# 5. Train Random Forest and Logistic Regression
# --------------------------------------------------

train_classifier(
    "heart_random_forest",
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
)

train_classifier(
    "heart_logistic_regression",
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
)


# --------------------------------------------------
# 6. Prepare Boston Housing regression data
# --------------------------------------------------

boston = boston.dropna(subset=["MEDV"]).copy()

X_boston = boston.drop(columns=["MEDV"])
y_boston = boston["MEDV"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_boston,
    y_boston,
    test_size=0.20,
    random_state=42,
)


# --------------------------------------------------
# 7. Train Random Forest Regressor
# --------------------------------------------------

housing_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            ),
        ),
    ]
)

with mlflow.start_run(
    run_name="boston_random_forest_regressor"
) as run:

    housing_model.fit(X_train, y_train)
    predictions = housing_model.predict(X_test)

    metrics = {
        "mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "rmse": float(
            math.sqrt(mean_squared_error(y_test, predictions))
        ),
        "r2_score": float(
            r2_score(y_test, predictions)
        ),
    }

    mlflow.log_params(
        {
            "dataset": "Boston Housing",
            "model_name": "RandomForestRegressor",
            "n_estimators": 200,
            "test_size": 0.20,
            "random_state": 42,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
        }
    )
    mlflow.log_metrics(metrics)

    model_path = MODEL_DIR / "boston_random_forest_regressor.joblib"
    joblib.dump(housing_model, model_path)

    mlflow.sklearn.log_model(
        sk_model=housing_model,
        name="boston_random_forest_regressor",
        serialization_format="pickle",
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="exports",
    )

    print("\nBoston Housing — Random Forest Regressor")
    print("-----------------------------------------")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("Run ID:", run.info.run_id)
    print("Saved model:", model_path)


print("\nAll required training experiments completed!")
print("MLflow database:", MLFLOW_DB)