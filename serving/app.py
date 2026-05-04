from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import mlflow
import os
import sys
import joblib
import warnings
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure the root directory is in python path to load the preprocessing module properly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

app = FastAPI(title="Credit Risk Model API", version="1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "Credit_Risk_Model"

xgb_model = None
woe_transformer = None
shap_explainer = None

@app.on_event("startup")
def load_model_artifacts():
    global xgb_model, woe_transformer, shap_explainer
    client = mlflow.tracking.MlflowClient()
    
    try:
        print(f"Connecting to MLflow: {MLFLOW_TRACKING_URI}")
        # First try to fetch models explicitly promoted to Production
        latest_versions = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
        if not latest_versions:
            latest_versions = client.get_latest_versions(name=MODEL_NAME)
            
        if not latest_versions:
            print(f"Warning: No registered model found with name '{MODEL_NAME}'. API degraded.")
            return

        # Sort to get the absolute latest if multiple exist within the filtered list
        latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        run_id = latest_version.run_id
        
        print(f"Loading version {latest_version.version} from Run ID: {run_id}")
        
        # 1. Load the XGBoost Model
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        xgb_model = mlflow.xgboost.load_model(model_uri)
        
        # 2. Download the WoE transformer pickle
        artifact_path = client.download_artifacts(run_id, "preprocessing/woe_transformer.pkl")
        woe_transformer = joblib.load(artifact_path)
        
        # 3. Initialize SHAP explainer
        # Workaround for SHAP + XGBoost 2.0 list-like base_score parsing issue
        import builtins
        import shap.explainers._tree
        original_float = builtins.float
        def safe_float(val):
            try:
                return original_float(val)
            except ValueError:
                if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                    return original_float(val[1:-1])
                raise
        
        shap.explainers._tree.float = safe_float
        shap_explainer = shap.TreeExplainer(xgb_model)
        shap.explainers._tree.float = original_float
        
        print("Model, WoE Transformer, and SHAP Explainer successfully loaded into memory.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading model during startup: {e}")

class PredictRequest(BaseModel):
    # Dynamic list of dictionary objects to accommodate arbitrary input formats 
    # matching the original training data.
    data: List[Dict[str, Any]]

@app.get("/health")
def health_status():
    status = "healthy" if xgb_model and woe_transformer else "degraded (models not loaded)"
    return {"status": status, "model_name": MODEL_NAME}

@app.post("/predict")
def predict(request: PredictRequest):
    if not xgb_model or not woe_transformer:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    try:
        # Convert Request to DataFrame
        df_input = pd.DataFrame(request.data)
        
        # Default Features obtained from data mode
        DEFAULT_FEATURES = {
            'credit_history_length': 9.0, 'income_stability': 1.0, 'interest_rate': 0.05, 
            'salary_volatility': 0.05, 'avg_account_balance': 2816651.0, 'monthly_income': 3203345.0, 
            'macro_unemployment': 0.03, 'medical_check_grade': 'Hijau', 'pension_amount': 0.0, 
            'recent_cash_withdrawals': 0.0, 'outstanding_balance': 1834289.0, 'macro_property_index': 101.7, 
            'employer_risk_score': 0.03, 'digital_payment_frequency': 4.0, 'loan_amount': 5004438.0, 
            'macro_inflation': 0.015, 'macro_interest_rate': 0.054, 'card_spending': 0.0, 
            'total_existing_installments': 0.0, 'product_type': 'Vehicle Loan', 'ltv': 0.0,
            'marital_status': 'Menikah', 'employment_type': 'Pegawai Swasta', 'salary_source': 'Payroll',
            'medical_expense_ratio': 0.1, 'health_risk_score': 0.2, 'age': 40, 'life_insurance_coverage': 0,
            'insurance_status': 'Tidak ada', 'insurance_premium_monthly': 0, 'bureau_score': 650, 
            'past_due_months': 0, 'collateral_flag': '0'
        }
        
        # Fill missing required features
        required_features = getattr(woe_transformer, '_selected_features', [])
        for f in required_features:
            if f not in df_input.columns:
                df_input[f] = DEFAULT_FEATURES.get(f, 0)
        
        # Apply transformation
        df_woe = woe_transformer.transform(df_input, keep_target=False)
        
        # Predict Proba expects the correct feature shape
        probs = xgb_model.predict_proba(df_woe)[:, 1]
        
        # Compute SHAP values
        # shap_values from TreeExplainer on XGBoost returns an array of shape (n_samples, n_features)
        shap_vals = shap_explainer.shap_values(df_woe)
        
        # Format response
        results = []
        for i, p in enumerate(probs):
            # Create a dictionary of features and their SHAP contributions
            feature_contributions = []
            for j, col in enumerate(df_woe.columns):
                # Clean up WOE suffix if exists for better display, or keep as is
                display_name = col.replace('_woe', '').replace('_WOE', '')
                feature_contributions.append({
                    "feature": display_name,
                    "value": float(shap_vals[i][j])
                })
            
            # Sort by absolute SHAP value (most impactful first)
            feature_contributions.sort(key=lambda x: abs(x["value"]), reverse=True)
            
            results.append({
                "probability": float(p),
                "risk_class": "High Risk" if p > 0.50 else "Low Risk",
                "shap_values": feature_contributions
            })
            
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount static folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")
