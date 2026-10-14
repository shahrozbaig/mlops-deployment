
from mlflow import MlflowClient

client = MlflowClient()

models = client.search_registered_models()

print("\n===== MODEL REGISTRY VERIFICATION =====")

for model in models:
    print(f"\nModel: {model.name}")

    versions = client.search_model_versions(
        f"name='{model.name}'"
    )

    for version in versions:
        print(f"  Version: {version.version}")
        print(f"  Aliases: {version.aliases}")

print("\n===== VERIFICATION COMPLETE =====")
