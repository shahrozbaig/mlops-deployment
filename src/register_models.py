
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


# --------------------------------------------------
# 1. MLflow configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "mlflow.db"

# Jenkins supplies a common URI for all branches.
# Local development falls back to the project database.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DATABASE_PATH.as_posix()}",
)

EXPERIMENT_NAME = "MLOps-Required-Datasets"
CHALLENGER_ALIAS = "Challenger"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)


# --------------------------------------------------
# 2. Required registered models
# --------------------------------------------------

MODELS = [
    {
        "registered_name": "HeartDiseaseRandomForest",
        "logged_model_name": "heart_random_forest",
        "description": (
            "Heart Disease classification using Random Forest."
        ),
    },
    {
        "registered_name": "HeartDiseaseLogisticRegression",
        "logged_model_name": "heart_logistic_regression",
        "description": (
            "Heart Disease classification using Logistic Regression."
        ),
    },
    {
        "registered_name": "BostonHousingRandomForestRegressor",
        "logged_model_name": "boston_random_forest_regressor",
        "description": (
            "Boston Housing regression using Random Forest."
        ),
    },
]


# --------------------------------------------------
# 3. Find the latest successful training model
# --------------------------------------------------

def find_latest_model_uri(logged_name):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and tags.mlflow.runName = '{logged_name}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            f"No successful training run found for {logged_name}"
        )

    run = runs[0]

    print(f"\nTraining run: {logged_name}")
    print(f"Run ID: {run.info.run_id}")

    # MLflow 3 supports logged model IDs.
    if hasattr(mlflow, "search_logged_models"):
        logged_models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            max_results=1000,
            output_format="list",
        )

        matches = [
            model
            for model in logged_models
            if model.name == logged_name
            and model.source_run_id == run.info.run_id
        ]

        if matches:
            model = max(
                matches,
                key=lambda item: getattr(item, "creation_timestamp", 0)
                or 0,
            )
            model_uri = f"models:/{model.model_id}"

            print(f"Logged model ID: {model.model_id}")
            return run, model_uri

    # Compatibility fallback for older MLflow versions.
    return run, f"runs:/{run.info.run_id}/{logged_name}"


# --------------------------------------------------
# 4. Register model and assign Challenger alias
# --------------------------------------------------

def register_model(model_info):
    registered_name = model_info["registered_name"]
    logged_name = model_info["logged_model_name"]

    run, model_uri = find_latest_model_uri(logged_name)

    print(f"Registering: {registered_name}")
    print(f"Model URI: {model_uri}")

    versions = client.search_model_versions(
        filter_string=f"name='{registered_name}'"
    )

    # Avoid creating another version for the same training run.
    existing = [
        version
        for version in versions
        if version.run_id == run.info.run_id
    ]

    if existing:
        version = max(
            existing,
            key=lambda item: int(item.version),
        )
        print(f"Already registered as version {version.version}")
    else:
        version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_name,
            await_registration_for=300,
        )
        print(f"Created version {version.version}")

    client.update_model_version(
        name=registered_name,
        version=str(version.version),
        description=model_info["description"],
    )

    client.set_registered_model_alias(
        name=registered_name,
        alias=CHALLENGER_ALIAS,
        version=str(version.version),
    )

    verified = client.get_model_version_by_alias(
        name=registered_name,
        alias=CHALLENGER_ALIAS,
    )

    if (
        verified.run_id != run.info.run_id
        or str(verified.version) != str(version.version)
    ):
        raise RuntimeError(
            f"Alias verification failed for {registered_name}"
        )

    print(
        f"SUCCESS: {registered_name} "
        f"version {verified.version} -> {CHALLENGER_ALIAS}"
    )


# --------------------------------------------------
# 5. Register and verify all required models
# --------------------------------------------------

if __name__ == "__main__":
    if (
        TRACKING_URI.startswith("sqlite:///")
        and not os.getenv("MLFLOW_TRACKING_URI")
        and not DATABASE_PATH.exists()
    ):
        raise FileNotFoundError(
            f"MLflow database not found: {DATABASE_PATH}. "
            "Run src/train_required_models.py first."
        )

    print("MLflow tracking URI:", TRACKING_URI)

    for model_info in MODELS:
        register_model(model_info)

    print("\n" + "=" * 55)
    print("ALL THREE MODELS REGISTERED SUCCESSFULLY")
    print("All Challenger aliases were verified.")
    print("=" * 55)
