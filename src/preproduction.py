
import os
from pathlib import Path

from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "mlflow.db"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{DATABASE_PATH.as_posix()}",
)

MODELS = [
    "HeartDiseaseRandomForest",
    "HeartDiseaseLogisticRegression",
    "BostonHousingRandomForestRegressor",
]

SOURCE_ALIAS = "Challenger"
PRE_TEST_ALIAS = "Challenger-pre-test"


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    print("MLflow tracking URI:", TRACKING_URI)
    print("Preparing models for pre-production testing")

    # Verify all source aliases before changing any aliases.
    versions = {}

    for name in MODELS:
        version = client.get_model_version_by_alias(
            name=name,
            alias=SOURCE_ALIAS,
        )
        versions[name] = str(version.version)

        print(
            f"Verified {name}: "
            f"version {version.version} -> {SOURCE_ALIAS}"
        )

    # Assign the pre-test alias to each verified version.
    for name, version in versions.items():
        client.set_registered_model_alias(
            name=name,
            alias=PRE_TEST_ALIAS,
            version=version,
        )

        verified = client.get_model_version_by_alias(
            name=name,
            alias=PRE_TEST_ALIAS,
        )

        if str(verified.version) != version:
            raise RuntimeError(
                f"Alias verification failed for {name}"
            )

        print(
            f"SUCCESS: {name} version {version} "
            f"-> {PRE_TEST_ALIAS}"
        )

    print("\nAll three models are ready for pre-production.")


if __name__ == "__main__":
    main()
