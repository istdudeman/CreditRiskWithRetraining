import os
import json
import time
import requests
import datetime
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "Credit_Risk_Model"

# Thresholds
MAX_MODEL_AGE_DAYS = 90
MACRO_UNEMPLOYMENT_THRESHOLD = 5.0 # Example: if unemployment > 5.0%, trigger retrain
MACRO_INFLATION_THRESHOLD = 8.0 # Example: if inflation > 8.0%, trigger retrain


def check_model_age() -> dict:
    """Checks the age of the latest MLflow model, returns a dict with status."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Try finding a Production model first, otherwise get latest versions
        latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
        if not latest_versions:
            latest_versions = client.get_latest_versions(name=MODEL_NAME)
            
        if not latest_versions:
            return {"triggered": True, "reason": f"No model named {MODEL_NAME} found in MLflow."}

        latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        
        # MLflow creation_timestamp is in milliseconds
        creation_time_ms = latest_version.creation_timestamp
        creation_date = datetime.datetime.fromtimestamp(creation_time_ms / 1000.0)
        
        age_days = (datetime.datetime.now() - creation_date).days
        
        if age_days > MAX_MODEL_AGE_DAYS:
            return {"triggered": True, "reason": f"Model is {age_days} days old (Threshold: {MAX_MODEL_AGE_DAYS})."}
        
        return {"triggered": False, "reason": f"Model is {age_days} days old, still fresh."}
        
    except Exception as e:
        return {"triggered": False, "reason": f"Error checking model age: {str(e)}"}


def check_macro_indicators() -> dict:
    """Simulates checking external APIs for macroeconomic indicators indicating a recession/crash."""
    # In a real scenario, you can replace this with an API call:
    # e.g., response = requests.get("https://api.worldbank.org/v2/country/us/indicator/SL.UEM.TOTL.ZS?format=json")
    # For demonstration, we simulate fetching fresh parameters...
    
    # Mock data to simulate API response:
    mock_api_response = {
        "unemployment_rate": 5.2, # Spiked!
        "inflation_rate": 4.5,
    }
    
    if mock_api_response["unemployment_rate"] > MACRO_UNEMPLOYMENT_THRESHOLD:
        return {"triggered": True, "reason": f"Unemployment spiked to {mock_api_response['unemployment_rate']}%"}
        
    if mock_api_response["inflation_rate"] > MACRO_INFLATION_THRESHOLD:
        return {"triggered": True, "reason": f"Inflation spiked to {mock_api_response['inflation_rate']}%"}
        
    return {"triggered": False, "reason": "Macroeconomic parameters normal."}


def check_drift_monitor() -> dict:
    """Hook for checking data drift metrics directly if we had inference storage available."""
    # Since drift_monitor relies on having new current data passed in, 
    # we would query recent database rows here and pass to the DriftMonitor class.
    # Currently omitted/bypassed until a database is connected, returning False.
    return {"triggered": False, "reason": "Drift monitoring requires current input database."}


def main():
    """Evaluates all triggers and outputs a JSON for n8n to parse"""
    
    triggers = []
    
    # 1. Check Model Age
    age_check = check_model_age()
    if age_check["triggered"]:
        triggers.append(age_check["reason"])
        
    # 2. Check Macro Economics
    macro_check = check_macro_indicators()
    if macro_check["triggered"]:
        triggers.append(macro_check["reason"])
        
    # 3. Check Data Drift
    drift_check = check_drift_monitor()
    if drift_check["triggered"]:
        triggers.append(drift_check["reason"])
        
    # Build final decision
    decision = {
        "should_retrain": len(triggers) > 0,
        "reason": " | ".join(triggers) if len(triggers) > 0 else "All checks pass. Model is stable."
    }
    
    # Print exactly one JSON line for n8n to parse natively
    print(json.dumps(decision))

if __name__ == "__main__":
    main()
