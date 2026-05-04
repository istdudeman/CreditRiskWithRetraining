import joblib
import pandas as pd

# Find the latest woe_transformer.pkl in mlartifacts (or I can just use mlflow to load it)
import mlflow
client = mlflow.tracking.MlflowClient()
MODEL_NAME = "Credit_Risk_Model"
latest_versions = client.get_latest_versions(name=MODEL_NAME)
latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
run_id = latest_version.run_id

artifact_path = client.download_artifacts(run_id, "preprocessing/woe_transformer.pkl")
woe_transformer = joblib.load(artifact_path)

if hasattr(woe_transformer, 'optbinners'):
    binner = woe_transformer.optbinners.get('age')
    if binner:
        print("WoE Bins for Age:")
        print(binner.splits)
        # also print the binning table if it exists
        if hasattr(binner, 'binning_table'):
            print(binner.binning_table.build(add_totals=False))
        else:
            print("No binning table found.")
    else:
        print("Age not found in optbinners.")
else:
    print("optbinners attribute not found on woe_transformer.")
