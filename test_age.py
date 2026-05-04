import requests
import json

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

for age in [30, 45, 60]:
    features = DEFAULT_FEATURES.copy()
    features['age'] = age
    try:
        response = requests.post("http://localhost:8000/predict", json={"data": [features]})
        print(f"--- AGE: {age} ---")
        if response.status_code == 200:
            res = response.json()["predictions"][0]
            print(f"Probability of Default: {res['probability']:.4f}")
            age_shap = next((s['value'] for s in res['shap_values'] if 'age' in s['feature'].lower()), None)
            print(f"SHAP contribution for age: {age_shap:.4f}")
        else:
            print(response.text)
    except Exception as e:
        print(e)
        
for cl_length in [2.0, 10.0]:
    features = DEFAULT_FEATURES.copy()
    features['credit_history_length'] = cl_length
    try:
        response = requests.post("http://localhost:8000/predict", json={"data": [features]})
        print(f"--- credit_history_length: {cl_length} ---")
        if response.status_code == 200:
            res = response.json()["predictions"][0]
            print(f"Probability of Default: {res['probability']:.4f}")
        else:
            print(response.text)
    except Exception as e:
        print(e)
