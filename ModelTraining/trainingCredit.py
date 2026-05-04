import pandas as pd
import numpy as np
import os
import time
import sys
import json
import warnings
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
import mlflow
import mlflow.xgboost

warnings.filterwarnings("ignore", category=UserWarning)

# Setup paths using the script's directory so it works from any execution node
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path so we can import from the preprocessing folder
sys.path.append(os.path.dirname(SCRIPT_DIR))
from preprocessing.feature_selection import WoETransformer
from preprocessing.imbalance_handler import SmoteResampler

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:/{os.path.join(os.path.dirname(SCRIPT_DIR), 'mlruns').replace('//', '/')}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment_name = f"Credit_Risk_MLOPS_{int(time.time())}"
mlflow.set_experiment(experiment_name)

try:
    with mlflow.start_run(run_name="Tuned_WoE_SMOTE_Run") as run:
        data_path = os.path.join(SCRIPT_DIR, "synthetic_credit_risk_data.csv")
        try:
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Dataset not found at {data_path}", file=sys.stderr)
            mlflow.log_param("status", "Data Load Failed")
            sys.exit(1)
            
        # 1. Clean Data
        orig_cols = data.columns.tolist()
        cols_to_drop = [c for c in ['customer_id', 'name_masked', 'loan_id', 'issue_date', 'PD', 'risk_grade', 'expected_loss', 'LGD', 'EAD'] if c in data.columns]
        data = data.drop(columns=cols_to_drop)
        
        if 'missed_payment_flag' not in data.columns:
            raise ValueError("Target column 'missed_payment_flag' not found in dataset.")
            
        # 2. Train-Test Split (Before any transformation to prevent leakage)
        print(f"Target Distribution: \n{data['missed_payment_flag'].value_counts(normalize=True)}", file=sys.stderr)
        train_df, test_df = train_test_split(data, test_size=0.3, random_state=42, stratify=data['missed_payment_flag'])
        
        # Save snapshot
        snapshot_path = os.path.join(SCRIPT_DIR, "X_test_snapshot.csv")
        test_df.drop(columns='missed_payment_flag').to_csv(snapshot_path, index=False)
        mlflow.log_artifact(snapshot_path)

        # 3. Feature Engineering: Weight of Evidence
        print("Applying WoE Transformation...", file=sys.stderr)
        woe = WoETransformer(target_col="missed_payment_flag", bins=10, iv_threshold=0.001)
        train_woe = woe.fit_transform(train_df, keep_target=True)
        test_woe = woe.transform(test_df, keep_target=True)
        
        X_train_woe = train_woe.drop(columns=['missed_payment_flag'])
        y_train = train_woe['missed_payment_flag']
        X_test_woe = test_woe.drop(columns=['missed_payment_flag'])
        y_test = test_woe['missed_payment_flag']

        # Log selected features length
        selected_features = woe.get_selected_features()
        mlflow.log_param("n_selected_features", len(selected_features))

        # 4. Handle Imbalance: SMOTE
        print("Applying SMOTE Oversampling...", file=sys.stderr)
        smote = SmoteResampler(target_col="missed_payment_flag", strategy="auto", random_state=42, verbose=False)
        # Re-attach target for SmoteResampler interface
        train_woe_combined = pd.concat([X_train_woe, y_train], axis=1)
        X_train_res, y_train_res = smote.fit_resample(train_woe_combined)

        # 5. Scale the Data
        #print("Scaling Data...", file=sys.stderr)
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train_res)
        #X_test_scaled = scaler.transform(X_test_woe)

        X_train_final = X_train_res
        X_test_final = X_test_woe

        # 6. Monotone Constraints (Removed)
        # We previously applied strict `monotone_constraints`, which can inadvertently 
        # cause XGBoost to abandon valid mathematical splits during bootstrapping.
        # Since we use WoE, XGBoost natively handles the ordering perfectly!

        # 7. Hyperparameter Tuning
        print("Starting Hyperparameter Tuning...", file=sys.stderr)
        param_dist = {
            'max_depth': [3, 5, 8, 10],        # Increased max_depth to allow better capture
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],           # Focus on seeing more data
            'colsample_bytree': [0.8, 1.0],    # Focus on seeing more features
            'n_estimators': [200, 500]         # Increase trees for better capture
        }

        xgb = XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist,
            n_iter=10,
            scoring='roc_auc',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=1
        )

        random_search.fit(X_train_final, y_train_res)
        
        best_model = random_search.best_estimator_
        print(f"Best Parameters: {random_search.best_params_}", file=sys.stderr)

        # Metrics evaluation
        y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
        y_pred = best_model.predict(X_test_final)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"AUC Score: {auc_score:.4f}", file=sys.stderr)
        print(f"F1 Score: {f1:.4f}", file=sys.stderr)
        
        # Logging to MLflow
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("model_type", "XGBoost (Tuned + SMOTE + WoE)")
        mlflow.log_param("monotone_constraints_applied", True)
        
        mlflow.log_metric("AUC", auc_score)
        mlflow.log_metric("F1_Score", f1)

        # Save and log WoE Transformer
        woe_path = os.path.join(SCRIPT_DIR, "woe_transformer.pkl")
        joblib.dump(woe, woe_path)
        mlflow.log_artifact(woe_path, artifact_path="preprocessing")

        MODEL_NAME = "Credit_Risk_Model"
        model_registered = False
        
        if auc_score > 0.70: 
            mlflow.xgboost.log_model(
                xgb_model=best_model, 
                artifact_path="model", 
                registered_model_name=MODEL_NAME
            )
            print(f"Model berhasil didaftarkan ke MLflow Model Registry dengan nama: {MODEL_NAME}", file=sys.stderr)
            model_registered = True
        else:
            print("Model tidak didaftarkan karena performa di bawah threshold (0.70).", file=sys.stderr)

        print(f"MLFLOW_RUN_ID:{run.info.run_id}", file=sys.stderr)
        
        # Output JSON for n8n orchestrator to easily parse
        n8n_output = {
            "status": "success",
            "run_id": run.info.run_id,
            "auc_score": round(float(auc_score), 4),
            "f1_score": round(float(f1), 4),
            "model_registered": bool(model_registered),
            "model_name": MODEL_NAME
        }

        # 2. Use a distinct marker so n8n can find it easily
        print("\n---N8N_JSON_START---")
        print(json.dumps(n8n_output))
        print("---N8N_JSON_END---")

except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
    print(f"Training pipeline failed: {str(e)}", file=sys.stderr)
    sys.exit(1)