# ITSNEEDS Final Project: Synthetic Data & Credit Risk Quality Gate

This documentation outlines the entire machine learning and MLOps pipeline for the Credit Risk Final Project (Tugas Akhir). The project leverages synthetic data generation, rigorous Weight of Evidence (WoE) preprocessing, robust model training (using XGBoost with Monotonic Constraints), and an automated MLOps Quality Gate (TSTR vs. TRTR) to ensure synthetic data utility and privacy.

---

## 🏗️ Project Architecture & Modules

The project is structured into three primary components:

### 1. `ModelTraining/` (Data Generation & Training)
This module acts as the foundation for the project by simulating Indonesian banking data and establishing the preliminary models.
- **`GenerateData.py`**: A robust synthetic data generator. It simulates demographical, financial, and behavioral credit risk parameters explicitly tailored to an Indonesian banking context (such as `'Pensiun'` statuses, BPJS/OJK behavior, and Taspen integrations). It computes a complex formula for Probability of Default (PD) and missed payment flags to serve as our target variable. Outputs `synthetic_credit_risk_data.csv`.
- **`trainingCredit.py`**: An XGBoost `XGBClassifier` training sandbox used to perform initial local evaluations. It tracks AUC and F1-score performance using MLflow to automatically register models that score above a predefined metric threshold (e.g., > 0.70 AUC).

### 2. `preprocessing/` (Data Engineering & Transformation)
This module securely and statistically prepares the data before it touches any model, specifically aimed at handling structural class imbalances and nonlinear feature mapping.
- **`feature_selection.py`**: Implements a modular `WoETransformer`. By binning continuous numerical features and comparing categorical classes to the target default rates, it replaces raw data with its theoretical Weight of Evidence (WoE). It also computes Information Value (IV) to drop features that fall below predicting thresholds (`> 0.02`), guaranteeing only highly predictive variables enter the XGBoost trees.
- **`imbalance_handler.py`**: *(Expected/Integrated Module)* Responsible for utilizing SMOTE (Synthetic Minority Over-sampling Technique) algorithms to handle extreme minority class populations (such as the 14.8% default vs. non-default skew), establishing an equilibrium during the model's cost-function updates.
- **`drift_monitor.py`**: *(Expected/Integrated Module)* Calculates the Population Stability Index (PSI) and Characteristic Stability Index (CSI) across numerical and categorical features to track and mathematically detect structural differences across unseen real-world data variants, triggering necessary retraining cycles.

### 3. `TSTR/` (The Quality Gate Validation)
The ultimate validation pipeline ensuring the generated synthetic data represents real-world dynamics without copying it, serving as a gatekeeper before production.
- **`evaluate_tstr.py`**: The orchestration file. Calculates two model baselines:
    1. **TRTR (Train Real Test Real)**: Baseline performance.
    2. **TSTR (Train Synthetic Test Real)**: Validated performance. 
   It rigorously applies Monotonic Constraints (`-1`) to logical banking variables (like income) ensuring the model behaves logically. All comparisons are tracked in MLflow. If the synthetic model's AUC drops `> 10%` from the real model, it is flagged as `"Low Utility"`.
- **`utils_metrics.py`**: Executes two major privacy and fidelity algorithms:
    1. **Inverted KS Statistic:** Mathematically measures distance between the Cumulative Distribution Function (CDF) of the `Dsyn` and `Dreal` arrays. ($1.0 = Perfect\ Distribution$).
    2. **Distance to Closest Record (DCR):** Generates $L^2$ Euclidean distance approximations using K-Nearest Neighbors to prove that the generative model synthesized new data, rather than illegally memorizing strictly confidential real-world rows.

---

## 🚀 Execution & Usage Guide

### 1. Generate Synthetic Data
Move to the `ModelTraining/` directory and execute the generator.
```bash
cd ModelTraining
python GenerateData.py
```
*This step produces the `synthetic_credit_risk_data.csv` mock sample.*

### 2. Run the Quality Gate (TSTR Evaluation)
To pass your synthetic tables through the validation algorithm, execute the evaluation script from the root context:
```bash
python TSTR/evaluate_tstr.py --real_data <path_to_real_data.csv> --syn_data <path_to_synthetic_data.csv> --target "missed_payment_flag"
```
*Wait for the Nearest Neighbor (DCR) evaluation to finish. The pipeline will output the Inverted KS, TRTR AUC, TSTR AUC, and finally determine if the Quality Gate is **PASSED** or **FAILED**.*

### 3. Review MLflow Logs
Since tracking operates locally natively (via `file:./mlruns`), all runs are recorded reliably.
```bash
mlflow ui --backend-store-uri mlruns
```
*Open `http://127.0.0.1:5000` in your browser to view run matrices, utility parameters, AUC graphs, and historical deployment statuses.*

---

## 📊 Model Evaluation & Project Findings

### 1. Data Distributions & Demographics (Synthetic Profile)
The dataset comprises **10,000 synthetic samples** specifically tailored to mirror Indonesian banking profiles (e.g., *Mandiri Taspen*, BPJS, ASN data). While the data pipeline is designed to manage true underlying real-world imbalances (such as a 14.8% default baseline) using **SMOTE**:
- **Target Variable (`missed_payment_flag`)**: Modeled based on a complex base formulation of `PD` (Probability of Default).
- **Age**: Ranges primarily between **45 to 69 years** (Average ~57 years), accurately reflecting Pensiun and ASN profiles.
- **Income (`monthly_income`)**: Averaging around **Rp 6.002.000**, with localized variations based on `employment_type` (Pensiunan vs. ASN Aktif vs. BUMN).

### 2. Interesting Findings
- **WoE Overcomes Strict Monotonic Constraints**: Implementing **Weight of Evidence (WoE)** successfully maps categorical data and non-linear boundaries cleanly along actual default rates. This organically structured the risks so well that hardcoded `monotone_constraints` in XGBoost were safely decoupled—letting the trees mathematically split risks optimally.
- **Data Utility Protection (TSTR)**: The Quality Gate rigorously validates synthetic utility without sacrificing privacy. It relies on the **Distance to Closest Record (DCR)** to prove that no true confidential data was memorized, and uses **Inverted KS** to compare synthetic distributions directly against real distributions.

### 3. Iteration Consistency & Pipeline Display
- **TRTR vs TSTR Validations**: The pipeline systematically compares dual iterations: the real baseline (TRTR) vs the synthetic deployment (TSTR). If the synthetic performance (AUC) degrades by **> 10%**, the MLflow pipeline flags the generation payload as `"Low Utility"`.
- **Hyperparameter Sweeps**: Executed sequentially via `RandomizedSearchCV` over **10 randomized iterations** (using 3 cross-validation folds). Each performance vector is tracked natively in MLflow tracking URLs.

### 4. Training Methods, Hyperparameters & Validation Metrics
- **Methodology**: The core pipeline unites `WoETransformer`, `SmoteResampler` for structural balance, and `XGBClassifier`.
- **Hyperparameter Space**: 
  - `max_depth`: `[3, 5, 8, 10]`
  - `learning_rate`: `[0.01, 0.05, 0.1]`
  - `n_estimators`: `[200, 500]`
  - `subsample` & `colsample_bytree`: `[0.8, 1.0]`
- **Loss Objective**: The estimator minimizes standard `logloss` as its foundational evaluation metric to heavily penalize over-confident false classifications.
- **Validation**: Key focus on **ROC AUC** metric (only iterations landing `> 0.70` become registered models) and **F1-Score** tracking.

### 5. SHAP Integration & Explainability
- **The Engine**: In the backend API inference (`app.py`), predictions call `shap.TreeExplainer` over the respective model paths against the WoE array mapping to evaluate the `shap_values`. The server converts these to absolute magnitudes, sorting the array dynamically to surface the heaviest determinants.
- **Explainable AI Focus**: *SHapley Additive exPlanations (SHAP)* fundamentally clarifies the Model's "black-box." It serves as a transparent view, assigning logical weights to *why* an applicant was scored as **High Risk** vs **Low Risk** by extracting exact contribution measures (e.g., negative bias heavily enforced by high *DTI* or low *bureau checking scores*).
