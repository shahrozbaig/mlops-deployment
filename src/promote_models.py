
import os
from pathlib import Path

from mlflow import MlflowClient


project_dir = Path(__file__).resolve().parent.parent
database_path = project_dir / "mlflow.db"

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{database_path.as_posix()}",
)

models = [
    "HeartDiseaseRandomForest",
    "HeartDiseaseLogisticRegression",
    "BostonHousingRandomForestRegressor",
]

client = MlflowClient(tracking_uri=tracking_uri)

# Resolve all versions before promoting any model.
versions = {}
for name in models:
    version = client.get_model_version_by_alias(
        name=name,
        alias="Challenger-pre-test",
    )
    versions[name] = str(version.version)

# Promote only after the Jenkins API test stage succeeds.
for name, version in versions.items():
    client.set_registered_model_alias(
        name=name,
        alias="Challenger-post-test",
        version=version,
    )
    verified = client.get_model_version_by_alias(
        name=name,
        alias="Challenger-post-test",
    )
    if str(verified.version) != version:
        raise RuntimeError(f"Promotion verification failed: {name}")

    print(f"PASS: {name} version {version} -> Challenger-post-test")

print("ALL THREE MODELS PROMOTED AFTER SUCCESSFUL TESTING")
