import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
import mlflow

# Add project root to path for importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.feature_selection import WoETransformer
from TSTR.utils_metrics import calculate_inverted_ks_statistic, calculate_dcr

def build_model():
    """Builds the XGBoost model with monotonic constraints."""
    # Example monotonic constraint: 
    # -1 means higher value corresponds to lower probability of default (e.g., income)
    #  1 means higher value corresponds to higher probability of default
    # For now, we will apply generic constraints if specified, but xgboost allows passing a dict.
    # If the user has a specific 'income' column transformed by WoE, we should be careful. 
    # In WoE, higher WoE means lower risk of default typically (since WoE = ln(P_good / P_bad)).
    # So if features are WoE substituted, monotonic decreasing (-1) against WoE is standard.
    # We will let XGBoost figure it out, but user asked for "Monotonic Constraints (higher income should not increase risk)".
    # To keep it generic yet fulfill requirements, we can add a constraint dictionary if 'income_woe' or similar exists.
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    return model

def main(real_data_path, synthetic_data_path, target_col):
    # mlflow.set_tracking_uri("http://localhost:5000") # Replace with your mlflow server
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("TSTR_Evaluation")
    
    with mlflow.start_run(run_name="TSTR_vs_TRTR_Run") as run:
        print(f"Loading datasets...")
        
        try:
            df_real = pd.read_csv(real_data_path)
            df_syn = pd.read_csv(synthetic_data_path)
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            mlflow.log_param("status", "Failed to load files")
            return
            
        print(f"Real data shape: {df_real.shape}")
        print(f"Synthetic data shape: {df_syn.shape}")
        
        # We need a Dreal set (Hold-out slice of original public dataset). 
        # Usually df_real is the entire real data. We will split it into train/test for TRTR.
        # Dreal_test will also be the test set for TSTR.
        X_real = df_real.drop(target_col, axis=1)
        y_real = df_real[target_col]
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )
        
        # Dsyn is used entirely for training
        X_syn = df_syn.drop(target_col, axis=1)
        y_syn = df_syn[target_col]
        
        # Calculate Data Privacy / Fidelity Metrics BEFORE WoE
        print("Calculating Fidelity & Privacy Constraints (DCR & KS)...")
        # Ensure we only compare matching numerical columns
        num_cols_real = set(X_real.select_dtypes(include=[np.number]).columns)
        num_cols_syn = set(X_syn.select_dtypes(include=[np.number]).columns)
        common_cols = list(num_cols_real.intersection(num_cols_syn))
        inverted_ks = calculate_inverted_ks_statistic(X_real, X_syn, columns=common_cols)
        dcr_metrics = calculate_dcr(X_real, X_syn, columns=common_cols)
        
        mlflow.log_metric("Inverted_KS_Statistic", inverted_ks)
        mlflow.log_metric("DCR_Mean", dcr_metrics["mean_dcr"])
        mlflow.log_metric("DCR_Min", dcr_metrics["min_dcr"])
        mlflow.log_metric("DCR_Perfect_Copies_Fraction", dcr_metrics["perfectly_copied_fraction"])
        
        print(f"- Inverted KS: {inverted_ks:.4f}")
        print(f"- Mean DCR: {dcr_metrics['mean_dcr']:.4f}")
        print(f"- Identical Record Fraction: {dcr_metrics['perfectly_copied_fraction']:.4%}")

        # -----------------------------
        # Preprocessing: WoE Transformation
        # -----------------------------
        # We must fit a separate WoETransformer for TRTR (on real_train) and TSTR (on syn).
        print("Applying WoE Transformations...")
        
        # For TRTR
        woe_trtr = WoETransformer(target_col=target_col, bins=10, iv_threshold=0.02)
        train_df_real = X_real_train.copy()
        train_df_real[target_col] = y_real_train
        X_real_train_woe = woe_trtr.fit_transform(train_df_real, keep_target=False)
        X_real_test_woe_trtr = woe_trtr.transform(X_real_test, keep_target=False)
        
        # For TSTR
        woe_tstr = WoETransformer(target_col=target_col, bins=10, iv_threshold=0.02)
        train_df_syn = X_syn.copy()
        train_df_syn[target_col] = y_syn
        X_syn_woe = woe_tstr.fit_transform(train_df_syn, keep_target=False)
        # Transform the exact SAME real hold-out test set using the synthetically-learned WoE mapper
        # Note: If some columns were dropped by IV thresholding in synthetic, they are also dropped here.
        # We need to make sure the model evaluates only on the intersection or handles missing cols.
        # It's usually safer to use the exact features returned by fit_transform.
        X_real_test_woe_tstr = woe_tstr.transform(X_real_test, keep_target=False)
        
        # Build models
        model_trtr = build_model()
        model_tstr = build_model()
        
        # Add Monotonic Constraints dynamically:
        # User requested: "higher income should not increase risk"
        # Let's see if income is one of the features. 
        # After WoE transformation, feature values are replaced by WoE scores.
        # Usually higher WoE means LOWER risk. If model is learning target_col=1 (Default), 
        # it should treat higher WoE as lower risk (-1 constraint).
        # We will apply -1 to ALL WoE transformed continuous columns just in case, or specific ones.
        # We'll apply -1 constraint to any column containing "income" or "pendapatan" if found, just as a safety.
        def apply_constraints(model_obj, features):
            constraints = []
            for col in features:
                if 'income' in col.lower() or 'pendapatan' in col.lower():
                    constraints.append(-1) # Increasing this feature should DECREASE prediction probability
                else:
                    constraints.append(0)  # No constraint
            model_obj.set_params(monotone_constraints=tuple(constraints))
            
        apply_constraints(model_trtr, X_real_train_woe.columns)
        apply_constraints(model_tstr, X_syn_woe.columns)

        # -----------------------------
        # TRTR Baseline Execution
        # -----------------------------
        print("Training TRTR (Train Real Test Real) Model...")
        model_trtr.fit(X_real_train_woe, y_real_train)
        
        trtr_preds = model_trtr.predict(X_real_test_woe_trtr)
        trtr_probs = model_trtr.predict_proba(X_real_test_woe_trtr)[:, 1]
        
        trtr_auc = roc_auc_score(y_real_test, trtr_probs)
        trtr_f1 = f1_score(y_real_test, trtr_preds)
        
        mlflow.log_metric("TRTR_AUC", trtr_auc)
        mlflow.log_metric("TRTR_F1", trtr_f1)
        
        print(f"--- TRTR Results ---")
        print(f"AUC: {trtr_auc:.4f} | F1: {trtr_f1:.4f}")
        
        # -----------------------------
        # TSTR Execution
        # -----------------------------
        print("Training TSTR (Train Synthetic Test Real) Model...")
        model_tstr.fit(X_syn_woe, y_syn)
        
        tstr_preds = model_tstr.predict(X_real_test_woe_tstr)
        tstr_probs = model_tstr.predict_proba(X_real_test_woe_tstr)[:, 1]
        
        tstr_auc = roc_auc_score(y_real_test, tstr_probs)
        tstr_f1 = f1_score(y_real_test, tstr_preds)
        
        mlflow.log_metric("TSTR_AUC", tstr_auc)
        mlflow.log_metric("TSTR_F1", tstr_f1)
        
        print(f"--- TSTR Results ---")
        print(f"AUC: {tstr_auc:.4f} | F1: {tstr_f1:.4f}")

        # -----------------------------
        # MLOps Quality Gate
        # -----------------------------
        print("\nEvaluating MLOps Quality Gate...")
        auc_drop = trtr_auc - tstr_auc
        auc_drop_pct = (auc_drop / trtr_auc) if trtr_auc > 0 else 0
        
        mlflow.log_metric("AUC_Drop_Absolute", auc_drop)
        mlflow.log_metric("AUC_Drop_Percentage", auc_drop_pct)
        
        utility_status = "Pass"
        if auc_drop_pct > 0.10: # > 10% drop
            utility_status = "Low Utility"
            print(f"❌ Quality Gate FAILED: TSTR AUC dropped by {auc_drop_pct:.2%} compared to TRTR (>10% threshold).")
        else:
            print(f"✅ Quality Gate PASSED: Synthetic data retains high utility.")
            
        mlflow.log_param("Utility_Status", utility_status)
        mlflow.log_param("Quality_Gate_Threshold", "10%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TSTR and TRTR Evaluation Pipeline.")
    parser.add_argument("--real_data", type=str, default="../ModelTraining/data_kredit_simulatif.csv", help="Path to real reference dataset")
    parser.add_argument("--syn_data", type=str, default="../ModelTraining/synthetic_credit_risk_data.csv", help="Path to synthetic dataset")
    parser.add_argument("--target", type=str, default="default_flag", help="Target column name")
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if they aren't absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    real_path = args.real_data if os.path.isabs(args.real_data) else os.path.join(script_dir, args.real_data)
    syn_path = args.syn_data if os.path.isabs(args.syn_data) else os.path.join(script_dir, args.syn_data)
    
    main(real_path, syn_path, args.target)
