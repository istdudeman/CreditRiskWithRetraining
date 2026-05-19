import mlflow
import os

mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()

# Get the latest version
latest_versions = client.get_latest_versions(name='Credit_Risk_Model')
latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]

print(f"Transitioning version {latest_version.version} to Production...")

client.transition_model_version_stage(
    name='Credit_Risk_Model', 
    version=latest_version.version, 
    stage='Production', 
    archive_existing_versions=True
)

print("Transition successful.")
