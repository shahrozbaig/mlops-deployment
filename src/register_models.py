
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# --------------------------------------------------
# 1. Configure MLflow
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "mlflow.db"

TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)


# --------------------------------------------------
# 2. Define the trained models
# --------------------------------------------------

# run_id: the exact run from successful model training
# logged_model_name: the name passed to mlflow.sklearn.log_model

MODELS = [
    {
        "registered_name": "HeartDiseaseRandomForest",
        "run_id": "d908e1829c91438baf87178ddaae1574",
        "logged_model_name": "heart_random_forest",
        "description": (
            "Heart Disease classification using Random Forest."
        ),
    },
    {
        "registered_name": "HeartDiseaseLogisticRegression",
        "run_id": "5fb4e4abe2ed4bc9b5db7e110617eb25",
        "logged_model_name": "heart_logistic_regression",
        "description": (
            "Heart Disease classification using Logistic Regression."
        ),
    },
    {
        "registered_name": "BostonHousingRandomForestRegressor",
        "run_id": "e8e58dad16574bb98ce273608ef7dbbe",
        "logged_model_name": "boston_random_forest_regressor",
        "description": (
            "Boston Housing regression using Random Forest."
        ),
    },
]


# --------------------------------------------------
# 3. Find the URI of a trained model
# --------------------------------------------------

def find_model_uri(run, logged_model_name):
    """
    Find the model logged by our training script.

    MLflow 3 uses a logged model ID.
    Older MLflow versions can use a runs:/ URI.
    """

    if hasattr(mlflow, "search_logged_models"):
        logged_models = mlflow.search_logged_models(
            experiment_ids=[run.info.experiment_id],
            max_results=1000,
            output_format="list",
        )

        matching_models = [
            item
            for item in logged_models
            if item.name == logged_model_name
            and item.source_run_id == run.info.run_id
        ]

        if not matching_models:
            raise RuntimeError(
                f"Could not find logged model "
                f"'{logged_model_name}' for run "
                f"{run.info.run_id}.\n"
                "Check the model name and run ID against "
                "the successful training output."
            )

        # This name was logged by our training script.
        logged_model = matching_models[-1]

        model_uri = f"models:/{logged_model.model_id}"

        print("Logged model ID:", logged_model.model_id)
        print("Model URI:", model_uri)

        return model_uri

    # Compatibility fallback for older MLflow versions.
    return (
        f"runs:/{run.info.run_id}/"
        f"{logged_model_name}"
    )


# --------------------------------------------------
# 4. Register one model and assign Challenger alias
# --------------------------------------------------

def register_model(model_info):
    name = model_info["registered_name"]
    run_id = model_info["run_id"]
    logged_name = model_info["logged_model_name"]

    print("\n" + "=" * 55)
    print("Processing:", name)
    print("=" * 55)

    # Verify that the training run exists.
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"Training run {run_id} is not FINISHED. "
            f"Current status: {run.info.status}"
        )

    # Check whether this run is already registered.
    try:
        all_versions = client.search_model_versions(
            filter_string=f"name='{name}'"
        )
    except MlflowException:
        all_versions = []

    existing_versions = [
        version
        for version in all_versions
        if version.run_id == run_id
    ]

    if existing_versions:
        version = max(
            existing_versions,
            key=lambda item: int(item.version),
        )

        print(
            "Existing registered version:",
            version.version,
        )

    else:
        # Find the model using the proper URI for our MLflow version.
        model_uri = find_model_uri(run, logged_name)

        # Create a registered model version.
        version = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            await_registration_for=300,
        )

        print(
            "New registered version:",
            version.version,
        )

    # Add a description to this registered model version.
    client.update_model_version(
        name=name,
        version=str(version.version),
        description=model_info["description"],
    )

    # Assign the assignment-required development alias.
    client.set_registered_model_alias(
        name=name,
        alias="Challenger",
        version=str(version.version),
    )

    # Verify that the alias resolves to the correct version.
    registered = client.get_model_version_by_alias(
        name=name,
        alias="Challenger",
    )

    print("\nRegistration results")
    print("--------------------")
    print("Registered name:", registered.name)
    print("Version:", registered.version)
    print("Alias: Challenger")
    print("Source run:", registered.run_id)

    if registered.run_id != run_id:
        raise RuntimeError(
            f"Alias verification failed for {name}. "
            f"Expected source run {run_id}, "
            f"got {registered.run_id}."
        )

    print("Verification: SUCCESS")


# --------------------------------------------------
# 5. Register and verify all three models
# --------------------------------------------------

if __name__ == "__main__":
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {DATABASE_PATH}"
        )

    for model_info in MODELS:
        register_model(model_info)

    print("\n" + "=" * 55)
    print("ALL THREE MODELS REGISTERED SUCCESSFULLY")
    print("All Challenger aliases were verified.")
    print("=" * 55)
